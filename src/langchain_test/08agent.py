from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


@tool
def multiply(a: int, b: int) -> int:
    """计算两数之积."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """计算两数之和."""
    return a + b


def agent01(model) -> None:
    """用 create_agent 封装工具，自动完成工具调用循环。"""
    agent = create_agent(
        model,
        tools=[multiply, add],
        system_prompt="你是一个专业的计算助手，使用工具完成计算。"
    )

    result = agent.invoke({"messages": "计算 23 乘以 17，然后再加 5。"})
    # print(result)
    print(result["messages"][-1].content)


def agent02(model) -> None:
    """查看 Agent 自动调用工具时产生的完整消息序列。"""
    agent = create_agent(
        model,
        tools=[multiply, add],
        system_prompt="你是一个专业的计算助手，需要计算时必须使用工具。",
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "计算 23 乘以 17，然后再加 5。",
                }
            ]
        }
    )

    # enumrate遍历时同时得到序号和元素,类似于go的range
    for index, message in enumerate(result["messages"], start=1):
        print(f"\n第 {index} 条：{type(message).__name__}")
        print("content:", repr(message.content))

        if getattr(message, "tool_calls", None):  # 尝试读取 message.tool_calls,如果没有这个属性，就返回 None
            print("tool_calls: ", message.tool_calls)

        if hasattr(message, "tool_call_id"):  # hasattr()：判断对象有没有某个属性
            print("tool_call_id:", message.tool_call_id)


def agent03(model) -> None:
    """使用 stream 实时观察 Agent 每一步产生的最新消息。"""
    agent = create_agent(
        model,
        tools=[multiply, add],
        system_prompt="你是一个专业的计算助手，需要计算时必须使用工具。",
    )

    states = agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "计算 12 乘以 8，然后加 4。",
                }
            ]
        },
        stream_mode="values",
    )

    for state in states:
        latest_message = state["messages"][-1]
        print(f"\n{type(latest_message).__name__}")
        print("content:", repr(latest_message.content))

        if getattr(latest_message, "tool_calls", None):
            print("tool_calls:", latest_message.tool_calls)


def agent04(model) -> None:
    """使用 checkpointer 和 thread_id 保存同一会话的消息历史。"""
    checkpointer = InMemorySaver()

    agent = create_agent(
        model,
        tools=[multiply, add],
        system_prompt="你是一个能记住对话内容的助手。",
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": "user-001",  # 会话编号
        }
    }

    # other_config = {
    #     "configurable": {
    #         "thread_id": "user-002",
    #     }
    # }

    agent.invoke(
        {"messages": [{"role": "user", "content": "请记住，我叫小明。"}]},
        config=config,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config=config,  # 这里如果写other_config就找不到之前的信息了
    )

    print(result["messages"][-1].content)


def main() -> None:
    load_dotenv()
    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )
    # agent01(model)
    # agent02(model)
    # agent03(model)
    agent04(model)


if __name__ == "__main__":
    main()
