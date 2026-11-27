# 本机测试素材

`samples/audio/` 存放用户复制进来用于开发验证的音效包。保留每个包原来的目录名和内部结构；整段 `audio/` 不进入 Git，也不随软件发行包分发。

当前本机目录：

```text
samples/audio/
  Magic Bundle 魔术魔幻音效(Design+Construction Kit 共2专辑)/
    Magic - Bundle/
      boom34/
      boom35_1/ … boom35_6/
```

本次整理移动了 594 个文件，核对了移动前后的相对路径、大小及修改时间。已生成的 128 条固定样本引用仍有效，样本 ID、内容版本与清单 ID 保持一致。

已有清单在 `.local/bootstrap/manifest.json`，实际素材根目录以 `.local/bootstrap/roots.local.json` 为准。其他 Agent 的独立工作目录不包含这些音频，应该读取 Lead 提供的主目录清单与根映射。

以后新增音效包也放在 `audio/` 下。只有需要更换固定样本集合时才重新生成清单；移动已有素材根目录时更新根映射，并核对文件身份即可。

样本用于流程测试，跨类别检索质量仍需要准备其他类别的评测素材。
