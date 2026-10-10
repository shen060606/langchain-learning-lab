import asyncio
import sys
import time
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dataclasses import dataclass, field
from langchain_core.messages import AIMessage, BaseMessage


@dataclass
class HarnessConfig:
    timeout_seconds: float = 20.0
    recursion_limit: int = 6


@dataclass
class HarnessResult:
    success: bool
    answer: str | None = None
    # 表示每次创建 HarnessResult 时，都执行一次,从而获得一个新的空列表
    used_tools: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0


def extract_used_tools(messages: list[BaseMessage]) -> list[str]:
    """从 Agent 消息中提取模型请求调用过的工具名称。"""
    used_tools = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                tool_name = tool_call["name"]
                used_tools.append(tool_name)

    return used_tools


async def run_with_harness(
    agent_function,
    question: str,
    config: HarnessConfig
) -> HarnessResult:
    """在运行边界内执行 Agent，并返回答案及工具调用记录。"""
    started_at = time.perf_counter()

    try:
        async with asyncio.timeout(config.timeout_seconds):
            answer, used_tools = await agent_function(question)

        result = HarnessResult(
            success=True,
            answer=answer,
            used_tools=used_tools,
            error=None,
        )

    except TimeoutError:
        result = HarnessResult(
            success=False,
            error="Agent 执行超时",
        )

    except Exception as error:
        result = HarnessResult(
            success=False,
            error=f"{type(error).__name__}: {error}",
        )

    finally:
        elapsed_seconds = time.perf_counter() - started_at

    result.elapsed_seconds = round(elapsed_seconds, 3)
    return result


def create_agent_function(agent, config: HarnessConfig):
    """创建符合 Harness 输入输出格式的 Agent 调用函数。"""

    async def invoke_agent(question: str) -> tuple[str, list[str]]:
        agent_state = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            config={
                "recursion_limit": config.recursion_limit,  # 从外部的配置对象中读取 recursion_limit 属性。
            },
        )

        messages = agent_state["messages"]
        final_message = messages[-1]

        answer = str(final_message.content)
        used_tools = extract_used_tools(messages)

        return answer, used_tools

    return invoke_agent


async def mcp_harness01() -> None:
    """连接本地 MCP Server，并取得可交给 Agent 的工具。"""
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

    print("[MCP tools for Agent]")

    for current_tool in tools:
        print(current_tool.name, type(current_tool).__name__)


async def mcp_harness02(model) -> None:
    """让 LangChain Agent 调用本地 MCP 工具完成计算。"""
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
        system_prompt="你是计算助手，进行计算时必须使用提供的工具。",
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

    used_tools = extract_used_tools(result["messages"])

    print("[MCP Agent messages]")
    print("[Used MCP tools]")
    print(used_tools)

    for message in result["messages"]:
        print(type(message).__name__, message.content)

async def mcp_harness03(model) -> None:
    """用轻量 harness 运行 Agent，并用固定样例检查工具选择与失败处理。"""
    client = MultiServerMCPClient(
        {
            "learning_server" : { 
                "transport" : "stdio" , 
                "command" : sys.executable, 
                "args" : [ 
                    "-m" , "langchain_test.18mcp_server" 
                ],
            }
        },
        handle_tool_errors=True,
    )
    tools = await client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=( "你是计算助手，进行计算时必须使用提供的工具。" "如果工具返回错误，不要重复调用，直接解释错误原因。" ),
    )

    config = HarnessConfig()

    run_agent = create_agent_function(agent, config)

    samples = [ "请计算 123 加 456。" , "请使用除法工具计算 10 除以 0。" ,]

    for question in samples:
        result = await run_with_harness(run_agent, question, config)
        print ( "\n[样例]" , question) 
        print ( "  成功：" , result.success) 
        print ( "  用到的工具：" , result.used_tools) 
        print ( "  答案：" , result.answer) 
        print ( "  错误：" , result.error) 
        print ( "  耗时(秒)：" , result.elapsed_seconds)


async def mcp_harness04 ( model ) -> None : 
    """用断言把固定样例收紧成可重复的测试用例。""" 
    client = MultiServerMCPClient(
        { "learning_server" : { "transport" : "stdio" , "command" : sys.executable, "args" : [ "-m" , "langchain_test.18mcp_server" ],
            }
        },
        handle_tool_errors= True ,
    )

    tools = await client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=( "你是计算助手，进行计算时必须使用提供的工具。" "如果工具返回错误，不要重复调用，直接解释错误原因。" ),
    )

    config = HarnessConfig()
    run_agent = create_agent_function(agent, config)

    cases = [
        { 
            "name" : "正常加法" , 
            "question" : "请计算 123 加 456。" , 
            "expect_success" : True , 
            "expect_tools" : [ "add" ], 
            "answer_contains" : "579" ,
        },
        { 
            "name" : "除零错误" , 
            "question" : "请使用除法工具计算 10 除以 0。" , 
            "expect_success" : True , 
            "expect_tools" : [ "divide" ], 
            "answer_contains" : "除数" ,
        },
    ]

    for case in cases:
        result = await run_with_harness(run_agent, case["question"], config)

        passed = (
            result.success == case["expect_success"]
            and result.used_tools == case [ "expect_tools" ] 
            and case [ "answer_contains" ] in result.answer
        )

        print ( "[PASS]" if passed else "[FAIL]" , case [ "name" ]) 
        if not passed: 
            print ( "  实际结果：" , result)



def main() -> None:
    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )

    # asyncio.run(mcp_harness01())
    # asyncio.run(mcp_harness02(model))
    # asyncio.run(mcp_harness03(model))
    asyncio.run(mcp_harness04(model))


if __name__ == "__main__":
    main()
