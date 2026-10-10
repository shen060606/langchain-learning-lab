# LangChain Learning Project

这是一个用于学习 Python 与 LangChain 的练习项目，内容从模型调用逐步扩展到 RAG、Agent、MCP 和轻量 Agent Harness。

项目以亲手编写和理解代码为目标。每个章节保存在独立的 Python 模块中，示例主要使用 DeepSeek 对话模型、硅基流动 Embedding 和 Qdrant 向量数据库。

## 学习内容

- 模型、消息、提示词和输出解析器
- Pydantic 与结构化输出
- LCEL 与 Runnable
- Tool、Agent、Memory、Checkpointer 和 Store
- Document、文本切分、Embedding、VectorStore 和 Retriever
- 完整 RAG、文件 RAG 与 Qdrant 持久化
- similarity、MMR、二阶段过滤、重排序和多查询检索
- 本地 stdio MCP Server 与 LangChain MCP Client
- 轻量 Agent Harness：超时、错误边界、日志和固定样例
- MCP、Agent 与 Harness 综合练习

## 环境要求

- Python 3.13 或更高版本
- [uv](https://docs.astral.sh/uv/) 或 pip
- DeepSeek API Key
- SiliconFlow API Key

调用真实模型或 Embedding API 会消耗对应平台的额度。

## 安装依赖

使用 uv：

```powershell
uv sync
```

或者在虚拟环境中使用 pip：

```powershell
pip install -e .
```

## 配置环境变量

复制环境变量示例文件：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写自己的 Key：

```dotenv
DEEPSEEK_API_KEY=your_deepseek_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

`.env` 已被 Git 忽略，请勿提交真实密钥。

## 运行示例

在项目根目录激活虚拟环境后，通过模块方式运行对应章节：

```powershell
python -m langchain_test.14rag_chain
python -m langchain_test.18mcp
python -m langchain_test.19harness
python -m langchain_test.20mcp_harness
```

第 18、20 章使用本地 stdio MCP Server。客户端会自动运行：

```powershell
python -m langchain_test.18mcp_server
```

通常不需要提前手动启动 MCP Server。

## 项目结构

```text
src/langchain_test/       各章节练习代码
data/langchain_notes.txt  文件 RAG 示例文档
data/qdrant/              Qdrant 本地数据，不提交到 Git
.env.example              环境变量示例
pyproject.toml            项目依赖与 Python 版本配置
```

## 当前进度

已完成第 01～19 章，目前正在进行第 20 章 MCP、Agent 与 Harness 综合练习。

## 说明

这是学习项目，不是生产级 Agent 平台。代码优先展示 LangChain 组件的数据流、输入输出类型和工程边界。
