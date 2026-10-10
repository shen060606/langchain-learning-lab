import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


def advanced_rag01(embedding_model: Embeddings) -> None:
    """比较 similarity 与 MMR 两种检索策略。"""
    documents = [
        Document(page_content="Go 使用 goroutine 实现轻量级并发。"),
        Document(page_content="Go 使用 channel 在 goroutine 之间通信。"),
        Document(page_content="Go 的标准库适合开发高并发网络服务。"),
        Document(page_content="LangChain 用于构建大模型和 RAG 应用。"),
    ]

    vector_store = InMemoryVectorStore(
        embedding=embedding_model,
    )
    vector_store.add_documents(documents)

    similarity_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    mmr_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 2,
            "fetch_k": 4,
            "lambda_mult": 0.3,
        },
    )

    question = "Go 如何进行并发编程？"

    print("[Similarity]")
    for document in similarity_retriever.invoke(question):
        print(document.page_content)

    print("\n[MMR]")
    for document in mmr_retriever.invoke(question):
        print(document.page_content)


def advanced_rag02(embedding_model: Embeddings) -> None:
    """先扩大向量召回范围，再按文档审核状态过滤候选结果。"""
    documents = [
        Document(
            page_content="Go 使用 goroutine 实现轻量级并发。",
            metadata={"status": "draft", "language": "go"},
        ),
        Document(
            page_content="Go 使用 channel 在 goroutine 之间安全通信。",
            metadata={"status": "verified", "language": "go"},
        ),
        Document(
            page_content="Go 的并发模型强调通过通信共享数据。",
            metadata={"status": "verified", "language": "go"},
        ),
        Document(
            page_content="Python 可以使用 asyncio 编写异步程序。",
            metadata={"status": "verified", "language": "python"},
        ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    broad_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    question = "Go 如何进行并发编程？"

    candidates = broad_retriever.invoke(question)

    # print("[Candidates]")

    # for document in candidates:
    #     print(document.metadata, document.page_content)

    filtered_documents = []

    for document in candidates:
        status = document.metadata.get("status")
        language = document.metadata.get("language")

        if status == "verified" and language == "go":
            filtered_documents.append(document)

    filtered_documents = filtered_documents[:2]

    print("[Filtered results]")
    for document in filtered_documents:
        print(document.page_content)


def advanced_rag03(embedding_model: Embeddings) -> None:
    """先向量召回候选文档，再根据关键词匹配情况重新排序。"""
    documents = [
        Document(page_content="Go 使用 goroutine 实现轻量级并发。"),
        Document(page_content="Go 使用 channel 在 goroutine 之间通信。"),
        Document(page_content="Go 的 channel 可以缓冲消息。"),
        Document(page_content="Python 使用 asyncio 实现异步并发。"),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    question = "Go 中如何使用 channel 在 goroutine 之间通信？"
    candidates = retriever.invoke(question)

    keywords = ["channel", "goroutine", "通信"]

    for document in candidates:
        score = 0

        for keyword in keywords:
            if keyword.lower() in document.page_content.lower():
                score += 1

        document.metadata["rerank_score"] = score

    def get_rerank_score(document):
        return document.metadata["rerank_score"]

    candidates.sort(
        key=get_rerank_score,
        reverse=True,
    )

    top_documents = candidates[:2]

    print("[Reranked results]")

    for document in top_documents:
        score = document.metadata["rerank_score"]
        print(score, document.page_content)


class DocumentRanking(BaseModel):
    document_index: int = Field(
        description="候选文档在输入列表中的索引",
        ge=0,
    )
    relevance_score: int = Field(
        description="文档与问题的相关性分数",
        ge=0,
        le=100,
    )
    reason: str = Field(
        description="给出该相关性分数的简短理由",
    )


class RerankResult(BaseModel):
    rankings: list[DocumentRanking] = Field(
        description="所有候选文档的评分结果"
    )


class QueryExpansion(BaseModel):
    queries: list[str] = Field(
        description="从不同角度改写得到的三个检索问题",
        min_length=3,
        max_length=3,
    )


def advanced_rag04(embedding_model: Embeddings, model) -> None:
    """先扩大向量召回范围，再使用大模型对候选文档进行语义重排。"""
    documents = [
        Document(page_content="Go 使用 goroutine 实现轻量级并发。"),
        Document(page_content="Go 使用 channel 在 goroutine 之间通信。"),
        Document(page_content="Go 的 channel 可以缓冲消息。"),
        Document(page_content="Python 使用 asyncio 实现异步并发。"),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    question = "Go 中如何使用 channel 在 goroutine 之间通信？"
    candidates = retriever.invoke(question)

    numbered_documents = []
    index = 0

    for document in candidates:
        numbered_document = f"[{index}] {document.page_content}"
        numbered_documents.append(numbered_document)
        index += 1

    candidate_text = "\n".join(numbered_documents)

    print("[Candidates for LLM reranker]")
    print(candidate_text)

    rerank_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 RAG 文档重排器。"
                "请判断每篇候选文档与问题的相关程度，"
                "为所有候选文档给出 0 到 100 的分数，不能遗漏。",
            ),
            (
                "human",
                "问题：{question}\n\n"
                "候选文档：\n{candidate_text}",
            ),
        ]
    )

    # 要求模型按照 RerankResult 的字段和类型返回结果
    structured_model = model.with_structured_output(RerankResult)
    rerank_chain = rerank_prompt | structured_model

    rerank_result = rerank_chain.invoke(
        {
            "question": question,
            "candidate_text": candidate_text,
        }
    )

    print("\n[LLM scores]")

    reranked_documents = []

    for ranking in rerank_result.rankings:
        document_index = ranking.document_index

        if document_index >= len(candidates):
            continue

        document = candidates[document_index]

        document.metadata["rerank_score"] = ranking.relevance_score
        document.metadata["rerank_reason"] = ranking.reason

        reranked_documents.append(document)

    print("\n[Mapped documents]")

    for document in reranked_documents:
        print(
            document.metadata["rerank_score"],
            document.page_content,
        )

    def get_llm_score(document):
        return document.metadata["rerank_score"]

    reranked_documents.sort(
        key=get_llm_score,
        reverse=True,
    )

    top_documents = reranked_documents[:2]

    print("\n[Final reranked documents]")

    for document in top_documents:
        score = document.metadata["rerank_score"]
        reason = document.metadata["rerank_reason"]

        print(score, document.page_content)
        print("理由：", reason)


def advanced_rag05(embedding_model: Embeddings, model) -> None:
    """使用大模型生成多个查询表达，为多查询检索准备输入。"""
    question = "Go 如何避免多个 goroutine 同时修改数据时发生问题？"

    query_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 RAG 查询改写器。"
                "请保留原始问题的含义，"
                "从不同术语和角度生成三个适合向量检索的问题。",
            ),
            (
                "human",
                "原始问题：{question}",
            ),
        ]
    )

    documents = [
        Document(page_content="Go 可以使用 sync.Mutex 保护共享数据。"),
        Document(page_content="多个 goroutine 无同步访问变量会产生数据竞争。"),
        Document(page_content="通过 channel 传递数据可以减少共享状态。"),
        Document(page_content="WaitGroup 用于等待一组 goroutine 执行完成。"),
        Document(page_content="Python 可以使用 Lock 保护线程共享数据。"),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    structured_model = model.with_structured_output(QueryExpansion)
    query_chain = query_prompt | structured_model

    query_result = query_chain.invoke(
        {
            "question": question,
        }
    )

    print("[Expanded queries]")

    all_retrieved_documents = []
    ranked_document_lists = []

    for query in query_result.queries:
        print("\n查询：", query)

        retrieved_documents = retriever.invoke(query)
        ranked_document_lists.append(retrieved_documents)

        for document in retrieved_documents:
            print("召回：", document.page_content)
            all_retrieved_documents.append(document)

    print("\n[Ranked document lists]")

    query_number = 1

    for document_list in ranked_document_lists:
        print("查询编号：", query_number)

        rank = 1

        for document in document_list:
            print("排名：", rank, document.page_content)
            rank = rank + 1

        query_number = query_number + 1

    print("\n[All retrieved documents]")

    seen_contents = set()
    unique_documents = []

    for document in all_retrieved_documents:
        content = document.page_content

        if content in seen_contents:
            continue

        seen_contents.add(content)
        unique_documents.append(document)

    print("\n[Unique documents]")

    recall_counts = {}

    for document in all_retrieved_documents:
        content = document.page_content

        if content not in recall_counts:
            recall_counts[content] = 0

        recall_counts[content] = recall_counts[content] + 1

    for document in unique_documents:
        content = document.page_content
        document.metadata["recall_count"] = recall_counts[content]

    def get_recall_count(document):
        return document.metadata["recall_count"]

    unique_documents.sort(
        key=get_recall_count,
        reverse=True,
    )

    print("\n[Documents ranked by recall count]")

    for document in unique_documents:
        count = document.metadata["recall_count"]
        print(count, document.page_content)


def advanced_rag06(embedding_model: Embeddings, model) -> None:
    """检索 top-k 文档并拼接上下文，为生成答案做准备。"""
    documents = [
        Document(page_content= "Go 使用 goroutine 实现轻量级并发。" ),
        Document(page_content= "Go 使用 channel 在 goroutine 之间通信。" ),
        Document(page_content= "多个 goroutine 无同步访问变量会产生数据竞争。" ),
        Document(page_content= "Python 使用 asyncio 实现异步并发。" ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_type= "similarity" ,
        search_kwargs={ "k" : 3 },
    )

    question = "Go 如何让 goroutine 之间可靠地传递数据？" 

    retrieved_documents = retriever.invoke(question)

    context_parts = []
    for index, document in enumerate(retrieved_documents):
        context_parts.append(f"[{index}] {document.page_content}")

    context = "\n".join(context_parts)

    print ( "[检索到的文档]" ) 
    for document in retrieved_documents: 
        print (document.page_content) 

    print ( "\n[拼接后的上下文]" ) 
    print (context)

    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是 Go 语言助手。请依据参考资料，用简洁连贯的语言总结回答，"
                "不要逐字复述原文；若资料确实不足以支撑回答，再说明资料有限。",
            ),
            (
                "human",
                "参考资料：\n{context}\n\n问题：{question}",
            ),
        ]
    )

    rag_chain = rag_prompt | model

    answer = rag_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print("\n[最终答案]")
    print(answer.content)



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
    )

    # advanced_rag01(embedding_model)
    # advanced_rag02(embedding_model)
    # advanced_rag03(embedding_model)
    # advanced_rag04(embedding_model, model)
    # advanced_rag05(embedding_model, model)
    advanced_rag06(embedding_model, model)


if __name__ == "__main__":
    main()
