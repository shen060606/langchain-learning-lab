from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv()

model = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    temperature=0,
)

# 正常输出


def invoke_demo() -> None:
    response = model.invoke("请用一句话解释 LangChain 是什么。")

    print("\n[invoke 完整响应]")
    print("回答内容：", response.content)
    print("消息类型：", type(response).__name__)
    print("响应元数据：", response.response_metadata)
    print("用量信息：", response.usage_metadata)

# 流式输出


def stream_demo() -> None:
    print("\n[stream 流式响应]")

    full_content = ""

    for chunk in model.stream("请列出学习 LangChain 的三个重点。"):
        print(chunk.content, end="", flush=True)
        full_content += chunk.content

    print()
    print("完整结果：", full_content)

# 批次输出


def batch_demo() -> None:
    questions = [
        "LangChain 是什么？",
        "RAG 是什么？",
        "向量数据库是什么？",
    ]

    responses = model.batch(questions)

    print("\n[batch 批量响应]")

    for index, response in enumerate(responses, start=1):
        print(f"\n问题 {index} 的答案：")
        print(response.content)


def main() -> None:
    print("模型对象类型：", type(model).__name__)

    invoke_demo()
    stream_demo()
    batch_demo()


if __name__ == "__main__":
    main()
