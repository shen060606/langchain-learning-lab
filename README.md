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
- LangGraph 状态、节点、边、条件路由与持久化
- LangGraph 接入 ChatModel、工具调用与已有 MCP/Retriever
- Retriever、RAG 与 Agent 的固定样例评估和运行观测
- FastAPI 服务化、会话标识、错误边界与健康检查
- Qdrant RAG、MCP、LangGraph 和 Harness 最终综合项目

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
python -m langchain_test.21langgraph
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

已完成第 01～21 章，目前开始第 22 章 LangGraph 接入 ChatModel 与工具。

后续学习路线：

- 第 22 章：在 LangGraph 节点中调用 ChatModel，并逐步接入工具与已有 MCP/Retriever。
- 第 23 章：评估与可观测性，建立 Retriever、RAG、Agent 的固定回归样例。
- 第 24 章：使用 FastAPI 把已有能力包装为最小可调用服务。
- 第 25 章：整合 Qdrant RAG、MCP、LangGraph 与 Harness，完成最终练手项目。

第 25 章完成并通过正常路径、无检索结果、工具失败和会话隔离测试后，本学习项目结束。复杂多 Agent、LangGraph Platform、分布式调度和生产部署不纳入本轮练习。

## 说明

这是学习项目，不是生产级 Agent 平台。代码优先展示 LangChain 组件的数据流、输入输出类型和工程边界。
