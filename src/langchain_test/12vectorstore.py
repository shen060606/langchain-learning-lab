import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


def vector_store01(embedding_model: Embeddings) -> None:
    """将 Document 向量化后添加到内存向量库。"""
    documents = [
        Document(
            page_content="Go 语言适合开发高并发后端服务。",
            metadata={"source": "golang.txt", "topic": "编程"},
        ),
        Document(
            page_content="今天阳光很好，适合外出散步。",
            metadata={"source": "weather.txt", "topic": "天气"},
        ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    ids = vector_store.add_documents(documents)

    print("返回类型：", type(ids).__name__)
    print("文档编号：", ids)


def vector_store02(embedding_model: Embeddings) -> None:
    """使用自然语言查询向量库中最相关的文档。"""
    documents = [
        Document(
            page_content="Go 语言适合开发高并发后端服务。",
            metadata={"source": "golang.txt", "topic": "编程"},
        ),
        Document(
            page_content="今天阳光很好，适合外出散步。",
            metadata={"source": "weather.txt", "topic": "天气"},
        ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    results = vector_store.similarity_search(
        query="什么语言适合高并发服务",
        k=1,
    )

    print("返回类型：", type(results).__name__)
    print("结果数量：", len(results))

    for document in results:
        print("正文：", document.page_content)
        print("元数据：", document.metadata)


def vector_store03(embedding_model: Embeddings) -> None:
    """搜索相关文档，同时观察每条结果的相似度分数。"""
    documents = [
        Document(
            page_content="Go 语言适合开发高并发后端服务。",
            metadata={"source": "golang.txt", "topic": "编程"},
        ),
        Document(
            page_content="今天阳光很好，适合外出散步。",
            metadata={"source": "weather.txt", "topic": "天气"},
        ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    results = vector_store.similarity_search_with_score(  # 与02的区别就是返回的时候多了分数
        query="什么语言适合开发高并发服务？",
        k=2,
    )

    for document, score in results:
        print("正文：", document.page_content)
        print("相似度：", f"{score:.4f}")


def vector_store04(embedding_model: Embeddings) -> None:
    """先按 metadata 过滤文档，再执行向量相似度搜索。"""
    documents = [
        Document(
            page_content="Go 语言适合开发高并发后端服务。",
            metadata={
                "source": "golang.txt",
                "topic": "编程",
                "tenant_id": "tenant-001",
            }
        ),
        Document(
            page_content="今天阳光很好，适合外出散步。",
            metadata={
                "source": "weather.txt",
                "topic": "天气",
                "tenant_id": "tenant-002",
            }
        ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    results = vector_store.similarity_search(
        query="什么语言适合开发高并发服务？",
        k=2,
        # lambda 相当于一个临时函数
        # def is_weather(document: Document) -> bool:
        #     return document.metadata.get("topic") == "天气"
        filter=lambda document: document.metadata.get(
            "tenant_id") == "tenant-001",
    )

    for document in results:
        print("正文：", document.page_content)
        print("元数据：", document.metadata)


def main() -> None:
    load_dotenv()

    embedding_model = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )

    # vector_store01(embedding_model)
    # vector_store02(embedding_model)
    # vector_store03(embedding_model)
    vector_store04(embedding_model)


if __name__ == "__main__":
    main()
