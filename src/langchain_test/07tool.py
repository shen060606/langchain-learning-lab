from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from pydantic import ValidationError, Field, BaseModel
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


class MultiplyInput(BaseModel):
    a: int = Field(description="第一个整数", ge=1, le=100)
    b: int = Field(description="第二个整数", ge=1, le=100)


@tool(args_schema=MultiplyInput)  # 作为参数规则
def bounded_multiply(a: int, b: int) -> int:
    """计算两个1到100之间的整数的乘积。"""
    return a * b


@tool  # 将你定义的函数包装成 LangChain 工具对象
def multiply(a: int, b: int) -> int:
    """ 计算两数之积"""
    return a*b


def tool01() -> None:
    # 定义工具，并通过字典传入参数
    print("\n[01: 定义与调用工具]")
    print("工具名称： ", multiply.name)
    print("工具描述： ", multiply.description)

    result = multiply.invoke({"a": 6, "b": 7})
    print("执行成功：", result)


def tool02() -> None:
    """" 观察工具参数结构， 并测试参数校验"""
    print("\n[02: 工具参数与校验]")
    print("参数结构： ", multiply.args)

    try:
        result = multiply.invoke({"a": "hello", "b": 7})
        print("执行结果：", result)
    except ValidationError as error:
        for item in error.errors():
            print("错误位置：", item["loc"])
            print("错误原因：", item["msg"])


def tool03() -> None:
    """使用自定义 Schema 限制工具参数范围。"""
    print("\n[03：自定义工具参数]")
    print("正常结果：", bounded_multiply.invoke({"a": 6, "b": 7}))

    try:
        bounded_multiply.invoke({"a": 101, "b": 7})
    except ValidationError as error:
        for item in error.errors():
            print("错误位置：", item["loc"])
            print("错误原因：", item["msg"])


def tool04(model) -> None:
    """让模型选择工具并生成调用参数。"""
    model_with_tools = model.bind_tools([bounded_multiply])

    response = model_with_tools.invoke(
        "请使用乘法工具计算23乘以17。"
    )

    print("\n[04：模型生成工具调用请求]")
    print("正文：", response.content)

    for call in response.tool_calls:
        print("工具名称：", call["name"])
        print("工具参数：", call["args"])
        print("调用编号：", call["id"])

def tool05(model) -> None:
    """执行工具并构造 ToolMessage 回传，再让模型给出最终答案。"""
    model_with_tooles = model.bind_tools([bounded_multiply])

    # 1 模型生成工具调用请求
    ai_message = model_with_tooles.invoke("请使用乘法工具计算23乘以17。")
    print("\n[05：完整工具调用循环]")
    print("第①步 模型请求调用：", ai_message.tool_calls)

    # 每一个 tool_call 就是一个待执行的任务
    tool_calls = ai_message.tool_calls
    call = tool_calls[0]# 0是第一个工具调用请求

    # 2 程序执行工具：args 是从模型返回的参数里取的
    result = bounded_multiply.invoke(call["args"])
    print("第②步 工具执行结果：", result)

    # 3 构造 ToolMessage 回传，tool_call_id 必须等于 call["id"]（第2课配对知识）
    from langchain.messages import ToolMessage
    tool_message = ToolMessage(content=str(result), tool_call_id = call["id"])

    final_response = model_with_tooles.invoke([ai_message, tool_message])
    print("第④步 模型最终回答：", final_response.content)

def tool06() -> None:
    """对比 @tool 简写与 args_schema 自定义，观察工具元数据。"""
    print("\n[06：工具元数据与 Schema 对比]")

    # 工具"说明书"三大件：name / description / args
    print("multiply         ：名称=%s  描述=%s" % (multiply.name, multiply.description))
    print("bounded_multiply ：名称=%s  描述=%s" % (bounded_multiply.name, bounded_multiply.description))

    print("\nmultiply 的 args（@tool 自动推断）：")
    print(multiply.args)

    print("\nbounded_multiply 的 args（自定义 Schema）：")
    print(bounded_multiply.args)

def tool07(model) -> None:
    """注册两个工具，让模型根据问题自动选择并执行。"""
    model_with_tools = model.bind_tools([multiply,bounded_multiply])

    questions = [
        "请计算 42 乘以 33。",
        "请用尽量大的数相乘并给出结果。",
    ]

    for question in questions:
        print("\n[问题]", question)

        ai_message = model_with_tools.invoke(question)#ai已经知道要调用什么工具了
        # print(ai_message)  
        print("模型想调用：", ai_message.tool_calls)

        results = []
        for call in ai_message.tool_calls:
            # 工具名 + 参数 往下派发
            tool_amounts = {"multiply": multiply, "bounded_multiply": bounded_multiply}
            chosen = tool_amounts[call["name"]]
            result_value = chosen.invoke(call["args"])

            print("  执行 %s(%s) = %s" % (call["name"], call["args"], result_value))
            results.append(ToolMessage(content=str(result_value), tool_call_id=call["id"]))

        final_response = model_with_tools.invoke([ai_message] + results)
        print("最终回答：", final_response.content)

def tool08(model) -> None:
    """保留完整消息历史，完成标准的工具调用循环。"""
    tools = [multiply, bounded_multiply]
    tool_map = {item.name: item for item in tools}
    model_with_tools = model.bind_tools(tools)

    messages = [
        HumanMessage(content="请使用工具计算23乘以17。")
    ]

    ai_message = model_with_tools.invoke(messages)
    messages.append(ai_message)

    print("\n[08：标准工具调用消息链]")
    print("模型请求：", ai_message.tool_calls)

    for call in ai_message.tool_calls:
        selected_tool = tool_map[call["name"]]
        tool_result = selected_tool.invoke(call["args"])

        tool_message = ToolMessage(
            content=str(tool_result),
            tool_call_id=call["id"],
        )
        messages.append(tool_message)

    final_response = model_with_tools.invoke(messages)
    print("最终回答：", final_response.content)

if __name__ == "__main__":
    # tool01()
    # tool02()
    # tool03()
    load_dotenv()
    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )
    # tool04(model)
    # tool05(model)
    # tool06()
    # tool07(model)
    tool08(model)
