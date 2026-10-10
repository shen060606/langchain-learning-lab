import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_documents(documents: list[Document]) -> str:
    """将多个 Document 整理成适合放入提示词的字符串。"""
    parts = []

    for document in documents:
        source = document.metadata.get("source", "未知")
        parts.append(
            f"来源：{source}\n内容：{document.page_content}"
        )

    return "\n\n".join(parts)  # 用两个换行把多段资料拼接起来


def rag01(embedding_model: Embeddings, model) -> None:
    """检索相关文档，并将结果整理成提示词上下文。"""
    documents = [
        Document(
            page_content="Go 语言适合开发高并发后端服务。",
            metadata={"source": "golang.txt"},
        ),
        Document(
            page_content="LangChain 可以编排模型、工具和检索流程。",
            metadata={"source": "langchain.txt"},
        ),
        Document(
            page_content="今天阳光很好，适合外出散步。",
            metadata={"source": "weather.txt"},
        ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2},
    )

    question = "LangChain 有什么作用？"
    retrieved_documents = retriever.invoke(question)
    # 把检索器返回的多个 Document 对象，整理成一段普通字符串，保存到 context 中
    context = format_documents(retrieved_documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个问答助手。只能根据下面的参考资料回答；"
                "资料不足时回答不知道。\n\n参考资料：\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    prompt_value = prompt.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print("\n提示词类型：", type(prompt_value).__name__)

    for message in prompt_value.messages:
        print(type(message).__name__, repr(message.content))

    response = model.invoke(prompt_value)

    print("\n模型返回类型：", type(response).__name__)
    print("最终答案：", response.content)

    print("检索结果类型：", type(retrieved_documents).__name__)
    print("上下文类型：", type(context).__name__)
    print("\n最终上下文：\n", context)


def rag02(embedding_model: Embeddings, model) -> None:
    """使用 LCEL 将检索、提示词、模型和解析器组成 RAG 链。"""
    documents = [
        Document(
            page_content="Go 语言适合开发高并发后端服务。",
            metadata={"source": "golang.txt"},
        ),
        Document(
            page_content="LangChain 可以编排模型、工具和检索流程。",
            metadata={"source": "langchain.txt"},
        ),
        Document(
            page_content="今天阳光很好，适合外出散步。",
            metadata={"source": "weather.txt"},
        ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个问答助手。只能根据下面的参考资料回答；"
                "资料不足时回答不知道。\n\n参考资料：\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    answer = chain.invoke("langchain有什么作用")

    print("\n[LCEL RAG]")
    print("返回类型：", type(answer).__name__)
    print("最终答案：", answer)


def rag03(embedding_model: Embeddings, model) -> None:
    """生成 RAG 答案，同时保留用于回答的来源文档。"""
    documents = [
        Document(
            page_content="Go 语言适合开发高并发后端服务。",
            metadata={"source": "golang.txt"},
        ),
        Document(
            page_content="LangChain 可以编排模型、工具和检索流程。",
            metadata={"source": "langchain.txt"},
        ),
        Document(
            page_content="今天阳光很好，适合外出散步。",
            metadata={"source": "weather.txt"},
        ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个问答助手。只能根据下面的参考资料回答；"
                "资料不足时回答不知道。\n\n参考资料：\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    retrieval_step = {
        "documents": retriever,
        "question": RunnablePassthrough(),
    }

    answer_chain = (
        {
            "context": lambda data: format_documents(data["documents"]),
            "question": lambda data: data["question"],
        }
        | prompt
        | model
        | StrOutputParser()
    )

    chain = retrieval_step | RunnablePassthrough.assign(
        answer=answer_chain
    )

    result = chain.invoke("LangChain 有什么作用？")

    print("\n[RAG 答案与来源]")
    print("答案：", result["answer"])

    for document in result["documents"]:
        print("来源：", document.metadata["source"])


def main() -> None:
    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )

    embedding_model = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )

    # rag01(embedding_model, model)
    # rag02(embedding_model, model)
    rag03(embedding_model, model)


if __name__ == "__main__":
    main()
