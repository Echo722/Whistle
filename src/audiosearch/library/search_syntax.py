"""REAPER Media Explorer search syntax parser compiled to trigram FTS5 + SQL."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r'"([^"]*)"|(\S+)')
_DRIVE_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")


def normalize_path_slashes(value: str) -> str:
    """Normalize Windows backslashes to POSIX forward slashes for rel_path matching."""
    return re.sub(r"\\+", "/", value)


def extract_absolute_path_query(query: str) -> str | None:
    """Return unquoted path if ``query`` looks like an absolute filesystem path."""
    s = query.strip()
    if not s:
        return None
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'") and s[0] not in s[1:-1]:
        s = s[1:-1].strip()
    if not s:
        return None
    if _DRIVE_ABS_RE.match(s):
        return s
    if s.startswith("\\\\") and len(s) >= 3 and s[2] != "\\":
        return s
    if os.name != "nt" and s.startswith("/") and len(s) >= 2:
        return s
    return None


def normalize_abs_path(raw: str) -> tuple[str, str, bool]:
    """Return ``(casefold_norm, original_case_norm, trailing_separator)`` for an absolute path."""
    trailing_sep = raw.endswith(("\\", "/"))
    if raw.startswith("\\\\") or raw.startswith("//"):
        rest = re.sub(r"[\\/]+", "/", raw[2:]).rstrip("/")
        norm = f"//{rest}" if rest else "//"
        return norm.casefold(), norm, trailing_sep
    if _DRIVE_ABS_RE.match(raw):
        norm = re.sub(r"[\\/]+", "/", raw)
        if len(norm) > 3:
            norm = norm.rstrip("/")
        return norm.casefold(), norm, trailing_sep
    norm = re.sub(r"/+", "/", raw.replace("\\", "/"))
    if len(norm) > 1:
        norm = norm.rstrip("/")
    return norm.casefold(), norm, trailing_sep


@dataclass(frozen=True, slots=True)
class _Term:
    raw: str
    quoted: bool
    negated: bool

    @property
    def _fts_core(self) -> str:
        if self.quoted:
            return normalize_path_slashes(self.raw)
        return normalize_path_slashes(self.raw.strip("*"))

    @property
    def is_fts_compatible(self) -> bool:
        core = self._fts_core
        if len(core) < 3:
            # SQLite FTS5 trigram requires at least 3 characters per token.
            return False
        if not self.quoted and "*" in core:
            # Internal wildcard (e.g. "fire*01") falls back to SQL LIKE.
            return False
        return True

    def to_fts_expr(self) -> str:
        escaped = self._fts_core.replace('"', '""')
        return f'"{escaped}"'

    def to_like_pattern(self) -> str:
        text = normalize_path_slashes(self.raw)
        if not self.quoted and "*" in text:
            escaped = _escape_like(text).replace("*", "%")
            if not escaped.startswith("%"):
                escaped = f"%{escaped}"
            if not escaped.endswith("%"):
                escaped = f"{escaped}%"
            return escaped
        return f"%{_escape_like(text)}%"


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    fts_match: str | None
    sql_clauses: tuple[str, ...]
    sql_params: tuple[str, ...]


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _tokenize(query: str) -> list[list[_Term]]:
    """Split query into OR-separated groups of AND-ed terms."""
    tokens: list[tuple[str, bool]] = []
    for match in _TOKEN_RE.finditer(query.strip()):
        if match.group(1) is not None:
            tokens.append((match.group(1), True))
        else:
            tokens.append((match.group(2), False))

    groups: list[list[_Term]] = [[]]
    pending_not = False
    for raw, quoted in tokens:
        if not quoted and raw.upper() == "OR":
            if groups[-1]:
                groups.append([])
            pending_not = False
            continue
        if not quoted and raw.upper() == "NOT":
            pending_not = True
            continue
        negated = pending_not
        pending_not = False
        if not quoted and raw.startswith("-") and len(raw) > 1:
            negated = True
            raw = raw[1:]
        if raw:
            groups[-1].append(_Term(raw=raw, quoted=quoted, negated=negated))
    return [group for group in groups if group]


def _term_like_sql(term: _Term, alias: str) -> tuple[str, list[str]]:
    pattern = term.to_like_pattern()
    fields = (
        f"{alias}.filename",
        f"{alias}.description",
        f"{alias}.package",
        f"{alias}.rel_path",
        f"{alias}.embedded_text",
    )
    ors = " OR ".join(f"{col} LIKE ? ESCAPE '\\'" for col in fields)
    return f"({ors})", [pattern] * len(fields)


def parse_me_query(query: str, *, alias: str = "f") -> ParsedQuery:
    """Parse ME syntax (AND, OR, NOT/-, "phrase", *) into trigram FTS5 + SQL."""
    groups = _tokenize(query)
    if not groups:
        return ParsedQuery(fts_match=None, sql_clauses=(), sql_params=())

    can_pure_fts = all(
        any(not t.negated for t in group) and all(t.is_fts_compatible for t in group)
        for group in groups
    )
    if can_pure_fts:
        group_exprs: list[str] = []
        for group in groups:
            positives = [t.to_fts_expr() for t in group if not t.negated]
            negatives = [t.to_fts_expr() for t in group if t.negated]
            expr = " AND ".join(positives) if len(positives) > 1 else positives[0]
            if len(positives) > 1 and negatives:
                expr = f"({expr})"
            for neg in negatives:
                expr = f"{expr} NOT {neg}"
            group_exprs.append(f"({expr})" if len(groups) > 1 else expr)
        return ParsedQuery(
            fts_match=" OR ".join(group_exprs),
            sql_clauses=(),
            sql_params=(),
        )

    if len(groups) == 1:
        group = groups[0]
        fts_pos = [t.to_fts_expr() for t in group if not t.negated and t.is_fts_compatible]
        fts_neg = [t for t in group if t.negated and t.is_fts_compatible]
        like_terms = [t for t in group if not t.is_fts_compatible]

        fts_match: str | None = None
        clauses: list[str] = []
        params: list[str] = []

        if fts_pos:
            expr = " AND ".join(fts_pos) if len(fts_pos) > 1 else fts_pos[0]
            if len(fts_pos) > 1 and fts_neg:
                expr = f"({expr})"
            for neg in fts_neg:
                expr = f"{expr} NOT {neg.to_fts_expr()}"
            fts_match = expr
        else:
            for neg in fts_neg:
                clauses.append(
                    f"{alias}.id NOT IN (SELECT rowid FROM files_fts WHERE files_fts MATCH ?)"
                )
                params.append(neg.to_fts_expr())

        for term in like_terms:
            sql, p = _term_like_sql(term, alias)
            clauses.append(f"NOT {sql}" if term.negated else sql)
            params.extend(p)

        return ParsedQuery(
            fts_match=fts_match,
            sql_clauses=tuple(clauses),
            sql_params=tuple(params),
        )

    or_sqls: list[str] = []
    params = []
    for group in groups:
        and_sqls: list[str] = []
        for term in group:
            if term.is_fts_compatible:
                sub = f"{alias}.id IN (SELECT rowid FROM files_fts WHERE files_fts MATCH ?)"
                and_sqls.append(f"NOT ({sub})" if term.negated else sub)
                params.append(term.to_fts_expr())
            else:
                sql, p = _term_like_sql(term, alias)
                and_sqls.append(f"NOT {sql}" if term.negated else sql)
                params.extend(p)
        or_sqls.append(f"({' AND '.join(and_sqls)})")
    return ParsedQuery(
        fts_match=None,
        sql_clauses=(f"({' OR '.join(or_sqls)})",),
        sql_params=tuple(params),
    )


def build_where_clause(query: str, *, alias: str = "f") -> tuple[bool, list[str], list[str]]:
    """Return ``(join_fts, where_clauses, params)`` for a ME search string."""
    parsed = parse_me_query(query, alias=alias)
    clauses: list[str] = []
    params: list[str] = []
    join_fts = parsed.fts_match is not None
    if parsed.fts_match is not None:
        clauses.append("files_fts MATCH ?")
        params.append(parsed.fts_match)
    clauses.extend(parsed.sql_clauses)
    params.extend(parsed.sql_params)
    return join_fts, clauses, params
