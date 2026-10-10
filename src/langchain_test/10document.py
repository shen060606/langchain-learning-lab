from langchain_core.documents import Document
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


def document01() -> None:
    """创建 Document，理解正文与元数据的结构。"""
    documents = [
        Document(
            page_content="LangChain 可以组合模型、工具和 Agent。",
            metadata={
                "source": "langchain-note.txt",
                "page": 1,
                "category": "AI",
            },
        ),
    ]

    print(type(documents).__name__)

    for document in documents:
        print(type(document).__name__)
        print("正文：", document.page_content)
        print("元数据：", document.metadata)


def document02() -> None:
    """使用 pathlib 读取文本，并转换成 LangChain Document。"""
    path = Path("pyproject.toml")
    text = path.read_text(encoding="utf-8")

    documents = [
        Document(
            page_content=text,
            metadata={"source": str(path)},
        )
    ]

    print("返回类型：", type(documents).__name__)
    print("文档数量：", len(documents))

    for document in documents:
        print("正文：", document.page_content)
        print("元数据：", document.metadata)


def document03() -> None:
    """将长 Document 切分成多个带重叠内容的小 Document。"""
    path = Path("src/langchain_test/07tool.py")

    document = Document(
        page_content=path.read_text(encoding="utf-8"),
        metadata={"source": str(path)},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
    )

    chunks = splitter.split_documents([document])

    print("切块数量：", len(chunks))

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n第 {index} 块，长度：{len(chunk.page_content)}")
        print(repr(chunk.page_content))
        print("元数据：", chunk.metadata)


def document04() -> None:
    """切分文档，并在元数据中记录每块的原文起始位置。"""
    path = Path("pyproject.toml")

    document = Document(
        page_content=path.read_text(encoding="utf-8"),
        metadata={"source": str(path)},
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        add_start_index=True,
    )

    chunks = splitter.split_documents([document])

    for index, chunk in enumerate(chunks, start=1):
        print(
            f"第 {index} 块：",
            f"起点={chunk.metadata['start_index']}，",
            f"长度={len(chunk.page_content)}",
        )
        print(repr(chunk.page_content[:60]))


def main() -> None:
    # document01()
    # document02()
    # document03()
    document04()


if __name__ == "__main__":
    main()
