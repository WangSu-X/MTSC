# Contributing to MTSC

感谢您考虑为 MTSC 做出贡献！

## 开发环境设置

1. 克隆仓库：
```bash
git clone https://github.com/YOUR_USERNAME/MTSC.git
cd MTSC
```

2. 创建虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

3. 安装开发依赖：
```bash
pip install -e .
pip install pytest pytest-asyncio ruff mypy
```

## 运行测试

运行所有测试：
```bash
pytest tests/ -v
```

运行特定测试文件：
```bash
pytest tests/test_mtsc.py -v
```

运行特定测试：
```bash
pytest tests/test_mtsc.py::MLALayoutTest::test_mla_namespace_unifies_tp_sizes_when_pcp_and_dcp_are_one -v
```

## 代码风格

本项目使用 Ruff 进行代码检查和格式化。

检查代码风格：
```bash
ruff check mtsc/ proxy/ tests/
```

自动格式化代码：
```bash
ruff format mtsc/ proxy/ tests/
```

## 提交规范

提交信息遵循以下格式：

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型
- `feat`: 新功能
- `fix`: 修复 bug
- `refactor`: 重构（不改变功能）
- `docs`: 文档更新
- `test`: 测试相关
- `chore`: 构建、工具等

### Scope 范围
- `kv_cache_pool`: KV 缓存池相关
- `kv_transfer`: KV 传输相关
- `connector`: Connector 相关
- `scheduler`: 调度器相关
- `worker`: Worker 相关

### 示例
```
feat(kv_cache_pool): unify MLA namespace across TP sizes when PCP=DCP=1

Enable MLA deployments with PCP=1 and DCP=1 to share Store objects across
different TP sizes by normalizing namespace tp_size to 1.

Closes #123
```

## Pull Request 流程

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 测试要求

- 所有新功能必须包含测试
- 所有测试必须通过
- 代码覆盖率应保持或提高
- 遵循现有的测试模式和命名约定

## 文档要求

- 更新相关的设计文档（`docs/design/`）
- 更新 README 如果添加了新功能
- 为公共 API 添加 docstring
- 保持中文注释和文档的语言一致性

## 行为准则

- 尊重所有贡献者
- 接受建设性批评
- 专注于对项目最有利的事情
- 对新贡献者表现出同理心

## 问题反馈

如果您发现 bug 或有功能请求，请在 GitHub Issues 中创建 issue。

提供以下信息会很有帮助：
- 清晰的标题和描述
- 重现步骤（对于 bug）
- 预期行为 vs 实际行为
- 相关日志或错误消息
- 环境信息（Python 版本、vLLM 版本等）

## License

提交代码即表示您同意在 Apache 2.0 许可证下授权您的贡献。
