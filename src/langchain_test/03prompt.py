from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate


def main() -> None:
    load_dotenv()

    model=init_chat_model(
        model = "deepseek-chat",
        model_provider = "deepseek",
        temperature = 0,
    )

    # 1 prompttemplate 字符串模版,模版可以自动扫描出里面的{}变量
    template = PromptTemplate.from_template("给我将一个关于{topic}的{style}笑话。")

    #1.1 format() 直接得到填好的字符串，用于肉眼检查
    text = template.format(topic="程序员",style= "冷幽默")
    print(text)

    #1.2 invoke() 得到promptvalued对象，可以直接交给模型
    prompt_value = template.invoke({"topic":"程序员","style":"冷幽默"})
    print(prompt_value)
    print(type(prompt_value).__name__)#是 StringPromptValue

    response = model.invoke(prompt_value)
    print(response.content)

    # 2 chatpropmttemplate 聊天模版
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system","你是一名{role}，回答不超过50字。"),
            ("human","{question}"),
        ]
    )

    #2.1 invoke 出来的还是promptvalue，但是家族变了
    prompt_value = chat_prompt.invoke(
        {"role": "python 教师","question":"什么是装饰器"}
    )
    print(type(prompt_value))# ChatPromptValue，不是 StringPromptValue

    #2.2看最终会变成什么消息发送出去
    messages = prompt_value.to_messages()
    for msg in messages:
        print(f"type={msg.type:<8} content={msg.content!r}")

    #2.3 直接交给模型
    response = model.invoke(messages)
    print(response.content)

    # 3 messageplaceholder  历史消息占位符
    chat_prompt2 = ChatPromptTemplate.from_messages(
        [
            ("system", "你是一名简洁的助手。"),
            MessagesPlaceholder("history"),   # ← 一个"消息列表插槽"
            ("human", "{question}"),
        ]
    )

    history = [
        HumanMessage(content="我叫小明，正在学习 LangChain。"),
        AIMessage(content="你好小明！"),
    ]

    messages = chat_prompt2.invoke(
        {"history": history, "question": "我叫什么？"}
    ).to_messages()

    for msg in messages:
        print(f"type={msg.type:<8} content={msg.content!r}")

    response = model.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    main()
