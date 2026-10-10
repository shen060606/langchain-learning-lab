
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from typing import TypedDict
from langchain.tools import ToolRuntime, tool


class UserContext(TypedDict):
    """调用 Agent 时传入的用户上下文。"""

    user_id: str


@tool
def get_user_profile(runtime: ToolRuntime[UserContext]) -> str:
    """读取当前用户的长期资料。"""
    if runtime.store is None:
        return "长期记忆存储未配置。"

    user_id = runtime.context["user_id"]  # 本次调用的用户上下文
    namespace = ("users", user_id)
    item = runtime.store.get(namespace, "profile")  # Agent 配置的长期记忆 Store

    if item is None:
        return "没有当前用户资料"

    return str(item.value)


def memory01(model) -> None:
    """使用 get_state 查看 checkpointer 保存的会话状态。"""
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": "conversation-001",
        }
    }

    agent.invoke(
        {"messages": [{"role": "user", "content": "请记住，我喜欢 Go。"}]},
        config=config,
    )

    snapshot = agent.get_state(config)  # 从 checkpointer 中读取最近的状态快照。

    print(type(snapshot).__name__)
    print("消息数量：", len(snapshot.values["messages"]))  # 取得当前会话保存的消息数量

    for message in snapshot.values["messages"]:
        print(type(message).__name__, repr(message.content))


def memory02(model) -> None:
    """使用不同 thread_id 隔离两个会话的短期记忆。"""
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[],
        checkpointer=checkpointer,
    )

    config_a = {
        "configurable": {
            "thread_id": "conversation-a",
        }
    }
    config_b = {
        "configurable": {
            "thread_id": "conversation-b",
        }
    }

    agent.invoke(
        {"messages": [{"role": "user", "content": "我是小明，喜欢 Go。"}]},
        config=config_a,
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": "我是小红，喜欢 Python。"}]},
        config=config_b,
    )

    snapshot_a = agent.get_state(config_a)
    snapshot_b = agent.get_state(config_b)

    print("\n[会话 A]")
    for message in snapshot_a.values["messages"]:
        print(type(message).__name__, repr(message.content))

    print("\n[会话 B]")
    for message in snapshot_b.values["messages"]:
        print(type(message).__name__, repr(message.content))


def memory03() -> None:
    """使用 Store 按用户保存和读取长期资料。"""
    store = InMemoryStore()
    namespace = ("users", "user-001")

    store.put(
        namespace,
        "profile",
        {
            "name": "小明",
            "favorite_language": "Go",
        },
    )

    store.put(
        namespace,
        "settings",
        {
            "response_style": "简洁",
            "language": "中文",
        },
    )

    item = store.get(namespace, "profile")

    print(type(item).__name__)
    print(item.value)

    items = store.search(namespace)

    print("\n[用户的全部长期记忆]")
    for result in items:
        print(result.key, result.value)


def memory04(model) -> None:
    """让 Agent 通过 ToolRuntime 读取当前用户的长期记忆。"""
    store = InMemoryStore()

    store.put(
        ("users", "user-001"),
        "profile",
        {
            "name": "小明",
            "favorite_language": "Go",
        },
    )

    agent = create_agent(
        model=model,
        tools=[get_user_profile],
        system_prompt="回答用户资料问题时，必须先调用工具查询。",
        context_schema=UserContext,
        store=store,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "我叫什么名字，喜欢什么编程语言"
                }
            ]
        },
        context={"user_id": "user-001"},
    )

    print(result["messages"][-1].content)

    # for message in result["messages"]:
    #     print(type(message).__name__, repr(message.content))

    #     if getattr(message, "tool_calls", None):
    #         print("tool_calls:", message.tool_calls)


def main() -> None:
    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )

    # memory01(model)
    # memory02(model)
    # memory03()
    memory04(model)


if __name__ == "__main__":
    main()
