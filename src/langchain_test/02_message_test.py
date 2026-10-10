from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import message_to_dict

def main() -> None:
    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )

    system_message = SystemMessage(content="你是一名 Go 和 Python 教师，回答不超过 50 字。")
    human_message = HumanMessage(content="Python 的 list 和 Go 的 slice 有什么区别？")
    ai_message = AIMessage(content="list 是可变动态数组，slice 是底层数组的视图。")

    for msg in [system_message, human_message, ai_message]:
        print(f"type={msg.type:<8} content={msg.content!r}")

    # 1 单次构建角色与对话
    messages = [system_message, human_message, ai_message]

    response = model.invoke(messages)

    print("\n模型回答：")
    print(response.content)

    print("\n返回对象类型：")
    print(type(response))

    # 2手动构建多轮对话
    history = [
        SystemMessage(content="你是一名简洁的聊天助手。"),
        HumanMessage(content="我叫小明，我主要使用 Go 开发。"),
    ]

    first_response = model.invoke(history)
    history.append(first_response)

    print("\nfirst_response结果：")
    print(first_response.content)

    history.append(
        HumanMessage(content="我叫什么？主要使用什么语言？")
    )

    second_response = model.invoke(history)

    print("\n多轮对话结果：")
    print(second_response.content)

    # 3 toolmessage 和序列化
    #3.1 发一条带tool_calls的aimessage，意思是我要调用这个 工具
    ai_calling_tool = AIMessage(
        content = "",#此时模型不说话，只发出调用请求
        tool_calls = [
            # 工具名：get_current_time，参数为 city: 北京， 唯一编号： call_001
            {"name": "get_current_time", "args": {"city": "北京"}, "id": "call_999"},
        ],
    )

    tool_result = ToolMessage(
        content="2026-06-06 14:30:00",
        tool_call_id= "call_999",
    )

    #3.2 py去执行工具，把结果包进toolmessage
    tool_history = [
        SystemMessage(content="你是一名简介的助手。"),
        HumanMessage(content="现在北京几点了？"),
        ai_calling_tool,
        tool_result,
    ]

    final_response = model.invoke(tool_history)

    print("\n[模型基于工具结果的最终回答]")
    print(final_response.content)

    print("\n[AIMessage 发起的工具调用]")
    print(ai_calling_tool.tool_calls)

    print("\n[ToolMessage 的结构]")
    print(f"tool_call_id={tool_result.tool_call_id}  content={tool_result.content!r}")

    # ---------- 序列化：把消息变成可存储的字典 ----------
    print("\n[单条消息序列化]")
    print(message_to_dict(tool_result))

    print("\n[整段历史逐条序列化]")
    for msg in tool_history:
        print(message_to_dict(msg))


if __name__ == "__main__":
    main()
