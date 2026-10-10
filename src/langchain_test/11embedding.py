import os

import numpy as np
from dotenv import load_dotenv
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_openai import OpenAIEmbeddings


def embedding01() -> None:
    """将一条查询文本转换成固定维度的测试向量。"""
    embedding_model = DeterministicFakeEmbedding(size=8)  # 输出向量包含 8 个数字

    vector = embedding_model.embed_query("langchain是什么")

    print("返回类型：", type(vector).__name__)
    print("向量维度：", len(vector))
    print("向量内容：", vector)


def embedding02() -> None:
    """批量将多段文档文本转换成二维向量列表。"""
    embedding_model = DeterministicFakeEmbedding(size=8)

    texts = [
        "LangChain 可以编排大模型应用。",
        "RAG 可以根据问题检索相关资料。",
        "今天天气很好。",
    ]

    vectors = embedding_model.embed_documents(texts)

    print("外层类型：", type(vectors).__name__)
    print("向量数量：", len(vectors))

    for index, vector in enumerate(vectors):
        print(
            f"第 {index + 1} 条：",
            texts[index],
            "维度：",
            len(vector),
            "前三维：",
            vector[:3],
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = np.dot(left, right)  # 计算两个向量的点积

    denominator = np.linalg.norm(left) * np.linalg.norm(right)  # 两个向量的长度相乘

    return float(numerator / denominator)


def embedding03() -> None:
    """比较查询向量与多个文档向量的余弦相似度。"""
    embedding_model = DeterministicFakeEmbedding(size=8)

    query_vector = embedding_model.embed_query("langchain是什么")
    document_vectors = embedding_model.embed_documents(
        [
            "LangChain 是什么？",
            "今天天气很好。",
        ]
    )

    for index, document_vector in enumerate(document_vectors, start=1):
        score = cosine_similarity(query_vector, document_vector)
        print(f"查询与文档 {index} 的相似度：{score:.4f}")


def embedding04(embedding_model: Embeddings) -> None:
    """调用真实 Embedding 模型比较文本语义相似度。"""
    query = "LangChain 有什么作用？"
    documents = [
        "LangChain 可以编排大模型、提示词和工具。",
        "今天天气很好，适合出去散步。",
    ]

    query_vector = embedding_model.embed_query(query)
    document_vectors = embedding_model.embed_documents(documents)

    print("向量维度：", len(query_vector))

    for index, document_vector in enumerate(document_vectors):
        score = cosine_similarity(query_vector, document_vector)
        print(documents[index], "相似度：", f"{score:.4f}")


def main() -> None:
    # embedding01()
    # embedding02()
    # embedding03()

    load_dotenv()

    embedding_model = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key=os.environ["SILICONFLOW_API_KEY"],
        base_url="https://api.siliconflow.cn/v1",
        check_embedding_ctx_length=False,
    )

    embedding04(embedding_model)


if __name__ == "__main__":
    main()
