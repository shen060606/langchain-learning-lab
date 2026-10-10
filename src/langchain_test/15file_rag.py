import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

KNOWLEDGE_PATH = Path("data/langchain_notes.txt")


def file_rag01() -> None:
    """读取本地知识文件，并转换成 LangChain Document。"""
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={"source": str(KNOWLEDGE_PATH)},
    )

    print("文件字符数：", len(text))
    print("对象类型：", type(document).__name__)
    print("来源：", document.metadata["source"])


def file_rag02() -> None:
    """读取真实文件，并切分成适合向量检索的文档块。"""
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={"source": str(KNOWLEDGE_PATH)},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=15,
    )

    chunks = splitter.split_documents([document])

    print("切块数量：", len(chunks))

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index
        print(f"\n第 {index + 1} 块：")
        print(repr(chunk.page_content))
        print(chunk.metadata)


def file_rag03(embedding_model: Embeddings) -> None:
    """将真实文件切块写入向量库，并检索相关文档。"""
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={"source": str(KNOWLEDGE_PATH)},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=15,
    )

    chunks = splitter.split_documents([document])

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    vector_store = InMemoryVectorStore(
        embedding=embedding_model
    )

    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2},
    )

    results = retriever.invoke(
        "Checkpointer 和 Store 分别保存什么数据？"
    )

    print("\n[真实文件检索]")
    print("检索数量：", len(results))

    for document in results:
        print(document.page_content)
        print(document.metadata)


def format_documents(documents: list[Document]) -> str:
    """将检索到的文件切块整理成提示词上下文。"""
    # 生成器表达式，与正常顺序相反
    return "\n\n".join(
        (
            f"来源：{document.metadata['source']}\n"
            f"切块：{document.metadata['chunk_index']}\n"
            f"内容：{document.page_content}"
        )
        for document in documents
    )


def file_rag04(embedding_model: Embeddings, model) -> None:
    """根据真实文件构建完整 RAG 问答链。"""
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={"source": str(KNOWLEDGE_PATH)},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=80,
        chunk_overlap=15,
    )

    chunks = splitter.split_documents([document])

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    vector_store = InMemoryVectorStore(
        embedding=embedding_model
    )

    vector_store.add_documents(chunks)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "只能根据参考资料回答问题；资料不足时回答不知道。"
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
        "Checkpointer 和 Store 分别保存什么数据？"
    )

    print("\n[真实文件 RAG]")
    print(answer)


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
    # file_rag01()
    # file_rag02()
    # file_rag03(embedding_model)
    file_rag04(embedding_model, model)


if __name__ == "__main__":
    main()
