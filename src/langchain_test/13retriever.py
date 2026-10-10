import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


def retriever01(embedding_model: Embeddings) -> None:
    """将 VectorStore 包装成 Retriever，并使用 invoke 执行检索。"""
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
        search_kwargs={"k": 1},
    )

    results = retriever.invoke("什么框架可以编排大模型应用？")

    print("检索器类型：", type(retriever).__name__)
    print("返回类型：", type(results).__name__)

    for document in results:
        print("正文：", document.page_content)
        print("来源：", document.metadata["source"])


def retriever02(embedding_model: Embeddings) -> None:
    """使用 MMR 在相关性与结果多样性之间取得平衡。"""
    documents = [
        Document(page_content="LangChain 可以编排模型和工具。"),
        Document(page_content="LangChain 可以管理提示词和模型调用。"),
        Document(page_content="RAG 使用向量检索为模型补充外部资料。"),
        Document(page_content="今天阳光很好，适合外出散步。"),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance，最大边际相关性
        search_kwargs={
            "k": 2,
            "fetch_k": 4,  # 先从向量库取四条候选文档，再由 MMR 从中选择两条。
            "lambda_mult": 0.9,  # 控制相关性与多样性的权重：
        },
    )

    results = retriever.invoke("如何构建大模型应用？")

    for index, document in enumerate(results, start=1):
        print(f"第 {index} 条：", document.page_content)


def retriever03(embedding_model: Embeddings) -> None:
    """使用 batch 一次执行多个检索问题。"""
    documents = [
        Document(page_content="Go 语言适合开发高并发后端服务。"),
        Document(page_content="LangChain 可以编排大模型应用。"),
        Document(page_content="今天阳光很好，适合外出散步。"),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 1}
    )

    queries = [
        "什么语言适合开发高并发服务？",
        "什么框架可以编排大模型应用？",
    ]

    batch_results = retriever.batch(queries)

    for query, results in zip(queries, batch_results):  # zip() 会把两个列表相同位置的元素配对
        print("\n问题：", query)

        for document in results:
            print("检索结果：", document.page_content)


def main() -> None:
    load_dotenv()

    embedding_model = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )

    # retriever01(embedding_model)
    # retriever02(embedding_model)
    retriever03(embedding_model)


if __name__ == "__main__":
    main()
