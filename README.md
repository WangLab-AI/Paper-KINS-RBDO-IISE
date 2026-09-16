# Paper-KINS-RBDO-IISE

KINS RBDO IISE 论文代码复现、对照实验和结果管理。

**Repository:** https://github.com/WangLab-AI/Paper-KINS-RBDO-IISE

## 存储与归档原则

- **代码：** GitHub 管理
- **数据：** Google Drive 管理
- **最终版本：** Google Drive Release 归档

GitHub 只保存代码、脚本、配置和文档。不得上传原始/处理后数据、模型权重、检查点、仿真结果大文件、密钥或访问令牌。

## 目录结构

```text
Paper-KINS-RBDO-IISE/
├── src/              # 核心算法与可复用代码
├── experiments/      # 实验入口与实验清单
├── configs/          # 参数、场景和随机种子
├── results/          # 本地结果目录，仅跟踪 .gitkeep
├── requirements.txt  # Python 依赖
└── README.md
```

正式复现必须在 `REPRODUCIBILITY.md` 的基础上补充论文版本、基线实现来源、指标定义、随机种子和与论文表格/图的对应关系。

## 开始工作

1. 阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。
2. 从最新 `main` 创建独立工作分支。
3. 从 Google Drive 获取指定数据版本，不把数据复制到 Git 跟踪区域。
4. 在配置中固定环境、数据版本、随机种子和关键参数。
5. 运行相关训练、仿真、评估或产品集成检查。
6. 通过 Pull Request 提交代码、配置、复现命令和结果摘要。

## 统一协作规范

- Branch：`feature/<name>`、`fix/<name>`、`experiment/<name>`、`docs/<name>`、`chore/<name>`。
- Commit：使用 `feat:`、`fix:`、`docs:`、`experiment:`、`refactor:`、`test:`、`chore:` 前缀。
- Pull Request：说明目的、命令、环境、数据/配置版本、验证结果和已知限制。
- Version：开发阶段使用 `v0.x.y`，稳定可复现版本从 `v1.0.0` 开始。
- Reproduction：固定 commit SHA、数据版本、环境、配置、随机种子和指标。

详细规则见 [CONTRIBUTING.md](CONTRIBUTING.md)、[VERSION.md](VERSION.md) 和 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

## 数据与 Release

数据、模型、仿真结果和最终交付包进入有权限控制的 Google Drive。GitHub README 或 Release 只登记外部归档标识、版本、负责人、访问权限和 SHA-256 校验值；不要公开受限分享链接。
