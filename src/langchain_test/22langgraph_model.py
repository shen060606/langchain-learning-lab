"""第 22 章：LangGraph 接入模型与工具。"""
from typing import TypedDict
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient


class ChatState(TypedDict):
    question: str
    answer: str


def create_answer_node(model):
    """创建一个可以在图节点中调用模型的函数。"""
    def answer_node(state: ChatState) -> dict[str, str]:
        response = model.invoke(state["question"])
        return {"answer": response.content}

    return answer_node


def create_answer_prompt_node(model):
    """创建一个先构造提示词、再调用模型的图节点。"""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是 LangGraph 助教，请用简洁的中文回答。"),
            ("human", "{question}"),
        ]
    )

    chain = prompt | model

    def answer_node(state: ChatState) -> dict[str, str]:
        response = chain.invoke(
            {"question": state["question"]}
        )
        return {"answer": response.content}

    return answer_node


def langgraph_model01(model) -> None:
    """把 ChatModel 接入 LangGraph 节点并运行。"""
    builder = StateGraph(ChatState)

    # answer_node = create_answer_node(model)
    answer_node = create_answer_prompt_node(model)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)

    graph = builder.compile()
    result = graph.invoke({
        "question": "Langgraph中 的节点有什么作用",
        "answer": "",
    })

    print("[Model node answer]")
    print(result["answer"])


@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


def create_tool_chat_node(model, tools):
    """创建一个能请求调用 add_numbers 工具的模型节点。"""
    model_with_tools = model.bind_tools(tools)

    async def assistant_node(state: MessagesState) -> dict:
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    return assistant_node


def langgraph_model02(model) -> None:
    """构建模型与工具之间的调用循环。"""
    builder = StateGraph(MessagesState)

    builder.add_node("assistant", create_tool_chat_node(model, [add_numbers]))
    builder.add_node("tools", ToolNode([add_numbers]))

    builder.add_edge(START, "assistant")
    # tools_condition要是判断不要使用工具了，就会到 END，所用不用写 END 的节点连接
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    graph = builder.compile()
    result = graph.invoke({
        "messages": [
            HumanMessage(content="请调用 add_numbers 计算 123 + 456。")
        ]
    })

    print("[Tool loop messages]")
    for message in result["messages"]:
        print(type(message).__name__)

        if message.type == "ai" and message.tool_calls:
            print("模型请求调用：", message.tool_calls)
        elif message.type == "tool":
            print("工具执行结果：", message.content)
        else:
            print("消息内容：", message.content)


async def langgraph_model03() -> None:
    """连接本地 MCP Server 并查看它提供的工具。"""

    client = MultiServerMCPClient(
        {
            "learning_server": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "langchain_test.18mcp_server"],
            }
        }
    )

    tools = await client.get_tools()

    print("[MCP tools]")
    for tool in tools:
        print(tool.name, ": ", tool.description)


async def langgraph_model04(model) -> None:
    """连接本地 MCP Server 并查看它提供的工具。"""

    client = MultiServerMCPClient(
        {
            "learning_server": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "langchain_test.18mcp_server"],
            }
        }
    )

    tools = await client.get_tools()

    builder = StateGraph(MessagesState)

    builder.add_node("assistant", create_tool_chat_node(model, tools))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    graph = builder.compile()
    result = await graph.ainvoke({
        "messages": [
            # HumanMessage(content="请使用 add 工具计算 123 + 456。")
            HumanMessage(content="请简单解释一下 LangGraph 中 State 的作用。")
        ]
    })

    print("[MCP tool loop]")
    for message in result["messages"]:
        print(type(message).__name__)

        if message.type == "ai" and message.tool_calls:
            print("模型请求调用：", message.tool_calls)
        elif message.type == "tool":
            print("MCP 工具结果：", message.content)
        else:
            print("消息内容：", message.content)


def main() -> None:
    """初始化模型并运行当前小节。"""
    load_dotenv()
    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )
    # langgraph_model01(model)
    # langgraph_model02(model)
    # 运行当前小节，查看 MCP 工具列表。
    # asyncio.run(langgraph_model03())
    asyncio.run(langgraph_model04(model))


if __name__ == "__main__":
    main()
