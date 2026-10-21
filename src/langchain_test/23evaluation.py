import os

import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStoreRetriever
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain.agents import create_agent
from langchain.tools import tool


def evaluation01(retriever) -> None:
    """用固定样例检查 Retriever 的 Hit@k。"""
    test_cases = [
        {
            "question": "什么框架可以编排大模型应用？",
            "expected_source": "langchain.txt",
        },
        {
            "question": "什么语言适合开发高并发后端服务？",
            "expected_source": "golang.txt",
        },
    ]

    hit_count = 0

    for case in test_cases:
        documents = retriever.invoke(case["question"])
        retrieved_sources = []

        for document in documents:
            retrieved_sources.append(document.metadata.get("source"))

        is_hit = case["expected_source"] in retrieved_sources
        if is_hit:
            hit_count += 1

        print(case["question"], "命中：", is_hit)

    hit_rate = hit_count / len(test_cases)
    print("Hit@k：", hit_rate)


def evaluation02(embedding_model: Embeddings) -> VectorStoreRetriever:
    """构造并返回评测用 Retriever。"""
    documents = [
        Document(
            page_content="LangChain 可以编排模型、工具和检索流程。",
            metadata={"source": "langchain.txt"},
        ),
        Document(
            page_content="Go 语言适合开发高并发后端服务。",
            metadata={"source": "golang.txt"},
        ),
    ]

    vector_store = InMemoryVectorStore(embedding=embedding_model)
    vector_store.add_documents(documents)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 1},
    )

    return retriever


def check_answer(
    answer: str,
    expected_points: list[str],
) -> tuple[float, list[str]]:
    """计算答案关键点覆盖率，并返回未命中的要点。"""
    matched_count = 0
    missing_points = []

    for point in expected_points:
        if point in answer:
            matched_count += 1
        else:
            missing_points.append(point)

    coverage = matched_count / len(expected_points)
    return coverage, missing_points


def evaluation03() -> None:
    """用多个固定样例检查答案关键点覆盖率。"""
    test_cases = [
        {
            "name": "完整回答",
            "answer": "LangChain 可以编排大模型、工具和检索流程。",
            "expected_points": ["编排大模型", "工具", "检索"],
        },
        {
            "name": "缺少关键点",
            "answer": "LangChain 可以编排大模型和工具。",
            "expected_points": ["编排大模型", "工具", "检索"],
        },
    ]

    for case in test_cases:
        coverage, missing_points = check_answer(
            case["answer"],
            case["expected_points"],
        )

        print(case["name"], "覆盖率：", coverage, "漏项：", missing_points)


def evaluation04(retriever: VectorStoreRetriever, model) -> None:
    """评测多个 RAG 问题的答案关键点覆盖率。"""
    test_cases = [
        {
            "question": "LangChain 有什么作用？",
            "expected_points": ["编排模型", "工具", "检索"],
        },
        {
            "question": "什么语言适合开发高并发后端服务？",
            "expected_points": ["Go", "高并发", "后端服务"],
        },
    ]

    total_coverage = 0.0

    for case in test_cases:
        retrieval_started = time.perf_counter()
        documents = retriever.invoke(case["question"])
        retrieval_seconds = time.perf_counter() - retrieval_started
        context_parts = []

        for document in documents:
            context_parts.append(document.page_content)

        context = "\n\n".join(context_parts)
        prompt = (
            "根据参考资料回答问题，不要使用资料之外的信息。\n"
            f"参考资料：\n{context}\n"
            f"问题：{case['question']}"
        )

        model_started = time.perf_counter()
        response = model.invoke(prompt)
        model_seconds = time.perf_counter() - model_started
        coverage, missing_points = check_answer(
            response.content,
            case["expected_points"],
        )
        total_coverage += coverage
        print("问题：", case["question"], "覆盖率：", coverage, "漏项：", missing_points)

        print("检索耗时：", round(retrieval_seconds, 3), "秒")
        print("模型耗时：", round(model_seconds, 3), "秒")

    average_coverage = total_coverage / len(test_cases)
    print("平均关键点覆盖率：", average_coverage)


@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


def check_tool_selection(
        messages: list[BaseMessage],
        expected_tools: list[str],
) -> tuple[bool, list[str]]:
    """检查 Agent 请求调用的工具是否符合预期。"""
    selected_tools = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls:
                selected_tools.append(tool_call["name"])

    passed = selected_tools == expected_tools
    return passed, selected_tools


def check_tool_execution(
        messages: list[BaseMessage],
        expected_result: str,
) -> tuple[bool, list[str]]:
    """检查工具执行结果是否符合预期。"""
    tool_results = []

    for message in messages:
        if isinstance(message, ToolMessage):
            tool_results.append(str(message.content))

    passed = expected_result in tool_results
    return passed, tool_results


def evaluation05(model) -> None:
    """评测 Agent 是否为计算问题选择了预期工具。"""
    agent = create_agent(
        model=model,
        tools=[add_numbers],
        system_prompt="进行加法计算时必须调用 add_numbers 工具。",
    )

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": "请用加法工具计算 123 加 456。"}
        ]
    })

    # print(result)

    passed, selected_tools = check_tool_selection(
        result["messages"],
        ["add_numbers"],
    )

    print("工具选择通过：", passed)
    print("实际选择的工具：", selected_tools)

    execution_passed, tool_results = check_tool_execution(
        result["messages"],
        "579",
    )
    print("工具执行结果：", tool_results)
    print("工具执行检查通过：", execution_passed)

    final_answer = str(result["messages"][-1].content)
    answer_passed = "579" in final_answer
    overall_passed = passed and execution_passed and answer_passed

    print("最终答案：", final_answer)
    print("最终答案检查通过：", answer_passed)
    print("本样例整体通过：", overall_passed)


def evaluation06() -> None:
    """用固定的错误工具结果测试失败路径。"""
    messages: list[BaseMessage] = [
        ToolMessage(content="578", tool_call_id="test-call")
    ]

    passed, tool_results = check_tool_execution(messages, "579")

    print("实际工具结果：", tool_results)
    print("错误结果被识别为失败：", not passed)

    selection_messages: list[BaseMessage] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "subtract_numbers",
                    "args": {"a": 123, "b": 456},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )
    ]

    selection_passed, selected_tools = check_tool_selection(
        selection_messages,
        ["add_numbers"],
    )

    print("实际选择的工具：", selected_tools)
    print("错误工具被识别为失败：", not selection_passed)

    failure_tests_passed = (not passed) and (not selection_passed)
    print("失败路径检查整体通过：", failure_tests_passed)


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

    # retriever = evaluation02(embedding_model)

    # evaluation01(retriever)
    # evaluation03()
    # evaluation04(retriever, model)
    evaluation05(model)
    # evaluation06()


if __name__ == "__main__":
    main()
