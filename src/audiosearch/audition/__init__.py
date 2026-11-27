"""Local audition: decode a fetched file into what the player serves.

Deliberately separate from ``audiosearch.integrations.reaper``: that package is
the REAPER bridge, which the desktop only calls for insert. Keeping the two
names apart means the bridge can change without touching local playback.
"""

from __future__ import annotations
