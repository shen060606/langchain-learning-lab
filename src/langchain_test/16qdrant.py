import os

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


QDRANT_PATH = Path("data/qdrant")
COLLECTION_NAME = "langchain_notes"


def qdrant01(embedding_model: Embeddings) -> None:
    """创建本地 Qdrant 集合，并持久化文档和向量。"""

    documents = [
        Document(
            page_content="LangChain 可以组合模型、提示词和检索器。",
            metadata={"source": "langchain.txt", "topic": "框架"},
        ),
        Document(
            page_content="Go 语言适合开发高并发后端服务。",
            metadata={"source": "golang.txt", "topic": "编程"},
        ),
    ]

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embedding_model,
        path=str(QDRANT_PATH),  # 指定写入路径
        collection_name=COLLECTION_NAME,  # 指定把文档存入 Qdrant 的哪个集合
        # force_recreate=True,  # 当集合已经存在的时候仍然会重建
    )

    print("\n[创建 Qdrant 集合]")
    print("对象类型：", type(vector_store).__name__)
    print("集合名称：", COLLECTION_NAME)
    print("存储路径：", QDRANT_PATH)

    vector_store.client.close()


def qdrant02(embedding_model: Embeddings) -> None:
    """重新连接磁盘中的 Qdrant 集合，并检索已有文档。"""
    vector_store = QdrantVectorStore.from_existing_collection(  # 连接已存在的向量数据库，并封装成向量存储对象
        embedding=embedding_model,
        path=str(QDRANT_PATH),
        collection_name=COLLECTION_NAME,
    )

    results = vector_store.similarity_search_with_score(
        query="什么语言适合开发高并发后端服务？",
        k=2,
    )

    print("\n[重新连接 Qdrant]")
    for document, score in results:
        print("正文：", document.page_content)
        print("元数据：", document.metadata)
        print("相似度：", score)

    vector_store.client.close()


def qdrant03(embedding_model: Embeddings) -> None:
    """使用 metadata 条件限制 Qdrant 的向量检索范围。"""
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        path=str(QDRANT_PATH),
        collection_name=COLLECTION_NAME,
    )

    topic_filter = models.Filter(
        # must 类似 SQL 的 AND 条件
        must=[
            models.FieldCondition(
                key="metadata.topic",  # 过滤条件:metadata的topic这个里面包含编程
                match=models.MatchValue(value="编程"),
            )
        ]
    )

    results = vector_store.similarity_search_with_score(
        query="什么技术适合开发后端服务？",
        k=2,
        filter=topic_filter,
    )

    print("\n[metadata 过滤]")
    for document, score in results:
        print("正文：", document.page_content)
        print("topic：", document.metadata["topic"])
        print("相似度：", score)

    vector_store.client.close()


def format_documents(documents: list[Document]) -> str:
    """将 Qdrant 返回的文档整理成提示词上下文。"""
    return "\n\n".join(
        (
            f"来源：{document.metadata['source']}\n"
            f"内容：{document.page_content}"
        )
        for document in documents
    )


def qdrant04(embedding_model: Embeddings, model) -> None:
    """连接持久化 Qdrant，并构建完整 RAG 问答链。"""
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        path=str(QDRANT_PATH),
        collection_name=COLLECTION_NAME,
    )

    try:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 2},
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "只能根据参考资料回答；资料不足时回答不知道。"
                    "\n\n参考资料：\n{context}",
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

        answer = chain.invoke(
            "LangChain 和 Go 分别适合做什么？"
        )

        print("\n[Qdrant 持久化 RAG]")
        print(answer)
    finally:
        vector_store.client.close()


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

    # qdrant01(embedding_model)
    # qdrant02(embedding_model)
    # qdrant03(embedding_model)
    qdrant04(embedding_model, model)


if __name__ == "__main__":
    main()
