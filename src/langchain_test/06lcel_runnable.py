# LCEL:langchain expression language
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser


def clean_text(text: str) -> str:
    return text.strip().lower()


def add_prefix(text: str) -> str:
    return f"处理结果：{text}"


def runnable01() -> None:

    cleaner = RunnableLambda(clean_text)
    prefixer = RunnableLambda(add_prefix)

    chain = cleaner | prefixer

    result = chain.invoke("    LANGCHAIN    ")

    print(result)
    print(type(result).__name__)


def runnable02() -> None:
    # 第二部分：观察管道的执行顺序和中间数据。
    def trace_clean(text: str) -> str:
        result = clean_text(text)
        print(f"清理步骤：{text!r} -> {result!r}")
        return result

    def trace_prefix(text: str) -> str:
        result = add_prefix(text)
        print(f"前缀步骤：{text!r} → {result!r}")
        return result

    print("\n[02：观察管道执行过程]")
    chain = RunnableLambda(trace_clean) | RunnableLambda(trace_prefix)
    print("链已创建，还没有执行")

    result = chain.invoke("   LANGCHAIN   ")
    print("最终结果：", result)


def runnable03() -> None:
    # 批量处理多个独立输入。
    chain = RunnableLambda(clean_text) | RunnableLambda(add_prefix)

    inputs = ["   LANGCHAIN   ", "  RAG  ", "  PYTHON  "]

    results = chain.batch(
        inputs,
        config={"max_concurrency": 2},  # 表示：批量处理 inputs 时，最多同时执行 2 个任务
    )

    print("\n[03：批量执行]")
    # zip() 把两个列表按位置配对
    for original, result in zip(inputs, results):
        print(f"{original!r} -> {result!r}")


def runnable04() -> None:
    # 同一个输入交给多个分支，结果汇总为字典。
    parallel = RunnableParallel(  # 类似于并行的 | ,这个管道符是串行的
        cleaned=RunnableLambda(clean_text),
        prefixed=RunnableLambda(add_prefix),
    )

    result = parallel.invoke("   LANGCHAIN   ")

    print("\n[04：并行分支]")
    for key, value in result.items():  # 获取字典中的键和值
        print(f"{key}: {value!r}")


def runnable05() -> None:
    # 保留原始输入，同时生成处理后的结果。
    chain = RunnableParallel(
        original=RunnablePassthrough(),  # 相当于一个封装了runnable接口的函数，作用就是保留原始数据
        cleaned=RunnableLambda(clean_text),
    )

    result = chain.invoke("   LANGCHAIN   ")

    print("\n[05：保留原始输入]")
    for key, value in result.items():
        print(f"{key}: {value!r}")


def runnable06() -> None:
    # 保留输入字典的字段，并补充清理后的问题。
    def get_cleaned_question(data: dict) -> str:
        return clean_text(data["question"])

    chain = RunnablePassthrough.assign(
        # 保留输入字典的字段，执行你指定的计算，再把计算结果作为新字段放进输出字典
        cleaned_question=RunnableLambda(get_cleaned_question),
    )

    result = chain.invoke({"question": "   LANGCHAIN   "})

    print("\n[06：保留并扩展字典]")
    for key, value in result.items():
        print(f"{key}: {value!r}")


def retrieve_context(data: dict) -> str:
    notes = {
        "RAG": "RAG 先检索相关资料，再让模型根据资料回答。",
        "Agent": "Agent 根据任务选择工具并生成调用参数，由程序执行工具，再结合工具结果继续处理或给出答案。",
    }
    return notes.get(data["question"], "没有找到相关资料")  # 找到键就返回对应值，找不到就返回默认值


def runnable07() -> None:
    # 模拟检索，将问题和上下文一起填入提示词。

    prompt = ChatPromptTemplate.from_messages([
        ("system", "根据参考资料回答；资料不足时说明不知道。\n{context}"),
        ("human", "{question}"),
    ])

    chain = RunnablePassthrough.assign(
        context=RunnableLambda(retrieve_context),
    ) | prompt

    result = chain.invoke({"question": "RAG"})

    for message in result.to_messages():
        print(f"[{message.type}] {message.content}")


def runnable08(model) -> None:
    # 将模拟检索、提示词、模型和解析器组成问答链。
    prompt = ChatPromptTemplate.from_messages([
        ("system", "仅根据参考资料回答，不超过50字；资料不足时说明不知道。\n{context}"),
        ("human", "{question}"),
    ])
    chain = (
        RunnablePassthrough.assign(
            context=RunnableLambda(retrieve_context)
        )
        | prompt | model | StrOutputParser()
    )

    answer = chain.invoke({"question": "Agent"})
    print("\n[08：基于资料生成回答]")
    print(answer)


if __name__ == "__main__":
    # runnable01()
    # runnable02()
    # runnable03()
    # runnable04()
    # runnable05()
    # runnable06()
    # runnable07()

    load_dotenv()
    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )
    runnable08(model)
