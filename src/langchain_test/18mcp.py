import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model


async def mcp01() -> None:
    """连接本地 MCP Server，发现并手动调用工具。"""
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "langchain_test.18mcp_server"],
    )

    async with stdio_client(server_parameters) as streams:
        read_stream = streams[0]
        write_stream = streams[1]

        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            result = await session.list_tools()

            print("[MCP tools]")

            for tool in result.tools:
                print("名称：", tool.name)
                print("说明：", tool.description)
                print("输入结构：", tool.inputSchema)

            call_result = await session.call_tool(
                name="add",
                arguments={
                    "a": 7,
                    "b": 5,
                },
            )

            print("\n[MCP tool result]")
            print("结果类型：", type(call_result).__name__)
            print("是否错误：", call_result.isError)
            print("结构化结果：", call_result.structuredContent)

            for content in call_result.content:
                if content.type == "text":
                    print("文本结果：", content.text)


async def mcp02() -> None:
    """连接 MCP Server，并将其工具转换成 LangChain Tool。"""
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

    print("[LangChain MCP tools]")

    for tool in tools:
        print("对象类型：", type(tool).__name__)
        print("工具名称：", tool.name)
        print("工具说明：", tool.description)
        print("参数结构：", tool.args)

    add_tool = tools[0]

    tool_result = await add_tool.ainvoke(
        {
            "a": 10,
            "b": 20,
        }
    )

    print("\n[LangChain MCP tool result]")
    print("结果类型：", type(tool_result).__name__)
    print("执行结果：", tool_result)


async def mcp03(model) -> None:
    """把 MCP 工具交给 Agent，由模型判断并执行工具调用。"""
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

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一名计算助手，需要计算时必须使用工具。"
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请计算 123 加 456。",
                }
            ]
        }
    )

    print("[Agent messages]")

    for message in result["messages"]:
        print(type(message).__name__, message.content)


async def mcp04() -> None:
    """对比 MCP 工具正常执行与业务异常的返回结果。"""
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "langchain_test.18mcp_server"],
    )

    async with stdio_client(server_parameters) as streams:
        read_stream = streams[0]
        write_stream = streams[1]

        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            success_result = await session.call_tool(
                name="divide",
                arguments={
                    "dividend": 10,
                    "divisor": 2,
                },
            )

            error_result = await session.call_tool(
                name="divide",
                arguments={
                    "dividend": 10,
                    "divisor": 0,
                },
            )

            print("[Success result]")
            print("是否错误：", success_result.isError)
            print("内容：", success_result.content)

            print("\n[Error result]")
            print("是否错误：", error_result.isError)
            print("内容：", error_result.content)


async def mcp05(model) -> None:
    """观察 MCP 工具错误如何经过 LangChain Agent 返回给用户。"""
    client = MultiServerMCPClient(
        {
            "learning_server": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "langchain_test.18mcp_server"],
            }
        },
        handle_tool_errors=True,
    )

    tools = await client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=(
            "你是一名计算助手，需要计算时必须使用工具。"
            "如果工具返回错误，不要重复调用，直接解释错误原因。"
        ),
    )

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请使用除法工具计算 10 除以 0。",
                }
            ]
        }
    )

    print("[Agent error flow]")

    for message in result["messages"]:
        print(type(message).__name__, message.content)


def main() -> None:
    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )

    # asyncio.run(mcp01())
    # asyncio.run(mcp02())
    # asyncio.run(mcp03(model))
    # asyncio.run(mcp04())
    asyncio.run(mcp05(model))


if __name__ == "__main__":
    main()
