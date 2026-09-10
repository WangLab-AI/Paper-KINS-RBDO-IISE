# 贡献与协作规范

本仓库采用分支开发和 Pull Request 审阅流程。禁止提交数据、模型权重、仿真结果大文件、密钥或个人绝对路径。

## Branch 规范

从最新的 `main` 创建分支，格式为：

```text
<类型>/<简短英文描述>
```

常用类型：

- `feature/`：新功能、算法、产品模块；
- `fix/`：错误修复；
- `experiment/`：实验或论文复现；
- `refactor/`：代码重构；
- `docs/`：文档；
- `chore/`：依赖和仓库维护。

示例：`feature/add-baseline-model`、`experiment/ablation-seed-42`、`fix/simulation-time-step`。

`main` 应始终保持可运行。一个分支只处理一项明确工作，不混入无关修改。

## Commit message 规范

格式：

```text
<类型>: <简洁的祈使句说明>
```

允许的常用类型包括 `feat`、`fix`、`docs`、`refactor`、`test`、`chore` 和 `experiment`。

示例：

```text
feat: add baseline controller
fix: make evaluation seed deterministic
docs: document dataset preprocessing
experiment: add ablation configuration
```

每个 commit 应小而完整。提交前检查 `git status`，确认没有数据、权重、结果、凭据和临时文件。

## Pull Request 流程

1. 推送工作分支并创建目标为 `main` 的 Pull Request。
2. 说明研究目的、主要改动、运行命令、配置、数据版本、验证结果和已知限制。
3. 实验类变更必须记录随机种子、环境版本、commit SHA 与关键指标。
4. 至少由一名非作者成员审阅；修改后重新运行相关检查。
5. 推荐使用 Squash merge，并保持最终 commit message 清晰。
6. 合并后删除工作分支；涉及用法变化时同步更新 README 或文档。

紧急修复也应通过 Pull Request。
