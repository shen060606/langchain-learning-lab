"""第 21 章：LangGraph 核心工作流。"""
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.errors import GraphRecursionError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt


class LearningState(TypedDict):
    text: str
    step_count: int


def add_prefix(state: LearningState) -> LearningState:
    """在文本前添加前缀，并增加步骤计数。"""
    new_text = "已处理: " + state["text"]
    new_step_count = state["step_count"] + 1

    return {
        "text": new_text,
        "step_count": new_step_count,
    }


class RouteState(TypedDict):
    question: str
    category: str
    answer: str


def classify_question(state: RouteState) -> dict[str, str]:
    """根据问题内容进行分类，并返回分类结果。"""
    question = state["question"]

    if "计算" in question:
        return {"category": "calculate"}

    return {"category": "knowledge"}


def calculate_node(state: RouteState) -> dict[str, str]:
    """处理需要计算的问题。"""
    return {"answer": "进入计算处理节点"}


def knowledge_node(state: RouteState) -> dict[str, str]:
    """处理普通知识类问题。"""
    return {"answer": "进入知识问答节点"}


def choose_route(state: RouteState) -> str:
    """读取分类结果，返回下一条路线的名称。"""
    return state["category"]


def langgraph01() -> None:
    """构建并运行 START 到单个节点再到 END 的基础图。"""
    # 创建一个图构建器，并规定整张图共享的 State 必须按照 LearningState 的结构传递。
    builder = StateGraph(LearningState)

    builder.add_node("add_prefix", add_prefix)

    builder.add_edge(START, "add_prefix")
    builder.add_edge("add_prefix", END)

    # 检查图的结构, 编译生成真正可执行的图对象
    graph = builder.compile()

    initial_state = {
        "text": "学习 langchain",
        "step_count": 0,
    }

    result = graph.invoke(initial_state)

    print("[LangGraph basic]")
    print(result)


def langgraph02() -> None:
    """根据问题分类结果，选择不同的处理节点。"""
    builder = StateGraph(RouteState)

    builder.add_node("classify_question", classify_question)
    builder.add_node("calculate_node", calculate_node)
    builder.add_node("knowledge_node", knowledge_node)

    builder.add_edge(START, "classify_question")

    # 添加条件边，三个参数分别为：从哪个节点开始判断,使用哪个函数判断,判断结果和节点的映射
    builder.add_conditional_edges(
        "classify_question",
        choose_route,
        {
            "calculate": "calculate_node",
            "knowledge": "knowledge_node",
        },
    )

    builder.add_edge("calculate_node", END)
    builder.add_edge("knowledge_node", END)

    graph = builder.compile()

    calculate_state: RouteState = {
        "question": "请计算 10 加 20",
        "category": "",
        "answer": "",
    }

    knowledge_state: RouteState = {
        "question": "LangGraph 是什么？",
        "category": "",
        "answer": "",
    }

    calculate_result = graph.invoke(calculate_state)
    knowledge_result = graph.invoke(knowledge_state)

    print("[Calculate route]")
    print(calculate_result)

    print("[Knowledge route]")
    print(knowledge_result)


class LoopState(TypedDict):
    count: int
    target: int


def increase_count(state: LoopState) -> dict[str, int]:
    """将当前计数增加一次。"""
    new_count = state["count"] + 1
    print("当前计数：", new_count)

    return {"count": new_count}


def choose_loop_route(state: LoopState) -> str:
    """判断继续循环还是结束图。"""
    if state["count"] < state["target"]:
        return "continue"

    return "stop"


def langgraph03() -> None:
    """让计数节点循环执行，达到目标次数后结束。"""
    builder = StateGraph(LoopState)

    builder.add_node("increase_count", increase_count)

    builder.add_edge(START, "increase_count")

    builder.add_conditional_edges(
        "increase_count",
        choose_loop_route,
        {
            "continue": "increase_count",
            "stop": END,
        },
    )

    graph = builder.compile()

    initial_state: LoopState = {
        "count": 0,
        "target": 3,
    }

    result = graph.invoke(initial_state)

    print("[LangGraph loop]")
    print(result)


def langgraph04() -> None:
    """使用递归限制阻止图无限循环。"""
    builder = StateGraph(LoopState)

    builder.add_node("increase_count", increase_count)
    builder.add_edge(START, "increase_count")

    builder.add_conditional_edges(
        "increase_count",
        choose_loop_route,
        {
            "continue": "increase_count",
            "stop": END,
        },
    )

    graph = builder.compile()

    initial_state: LoopState = {
        "count": 0,
        "target": 100,
    }

    try:
        graph.invoke(
            initial_state,
            config={"recursion_limit": 3},
        )
    except GraphRecursionError:
        print("[LangGraph recursion limit]")
        print("图超过最大执行步数，已被安全停止。")


class CheckpointState(TypedDict):
    count: int


def checkpoint_increment(state: CheckpointState) -> dict[str, int]:
    """读取上一次计数，将其增加一次。"""
    new_count = state["count"] + 1

    return {"count": new_count}


def langgraph05() -> None:
    """使用 Checkpointer 保存并延续同一会话的状态。"""
    builder = StateGraph(CheckpointState)

    builder.add_node("increment", checkpoint_increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    # 第一个会话：counter-session，first和 second, third 都是基于这个会话的操作
    config = {
        "configurable": {
            "thread_id": "counter-session",
        }
    }

    first_result = graph.invoke(
        {"count": 0},
        config=config,
    )

    # 虽然 count 传的是空，但是这个会话里面保存了上次画画的结果 count=1+1=2
    second_result = graph.invoke(
        {},
        config=config,
    )

    # 另一个会话，count的值与其他会话无关
    other_config = {
        "configurable": {
            "thread_id": "another-session",
        }
    }

    other_result = graph.invoke(
        {"count": 100},
        config=other_config,
    )

    third_result = graph.invoke(
        {},
        config=config,
    )

    counter_snapshot = graph.get_state(config)
    other_snapshot = graph.get_state(other_config)

    print("[Saved states]")
    print(type(counter_snapshot).__name__)
    print(counter_snapshot.values)
    print(other_snapshot.values)
    print("下一节点：", counter_snapshot.next)

    print("[First result]")
    print(first_result)

    print("[Second result]")
    print(second_result)

    print("[Other thread result]")
    print(other_result)

    print("[Original thread result]")
    print(third_result)


class ReviewState(TypedDict):
    draft: str
    approved: bool


def request_review(state: ReviewState) -> dict[str, bool]:
    """暂停工作流，等待外部提交审核结果。"""
    decision = interrupt(
        {
            "question": "是否批准这个草稿？",
            "draft": state["draft"],
        }
    )

    return {"approved": decision}


def choose_review_route(state: ReviewState) -> str:
    """根据审核结果选择后续路线。"""
    if state["approved"]:
        return "approved"

    return "rejected"


def revise_draft(state: ReviewState) -> dict[str, str]:
    """模拟将未通过审核的草稿退回修改。"""
    return {"draft": "退回修改：" + state["draft"]}


def langgraph06() -> None:
    """运行到人工审核节点并暂停工作流。"""
    builder = StateGraph(ReviewState)

    builder.add_node("request_review", request_review)
    builder.add_edge(START, "request_review")

    builder.add_node("revise_draft", revise_draft)
    builder.add_conditional_edges(
        "request_review",
        choose_review_route,
        {
            "approved": END,
            "rejected": "revise_draft",
        },
    )

    builder.add_edge("revise_draft", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "review-session",
        }
    }

    paused_result = graph.invoke(
        {
            "draft": "这是一份等待审核的回答。",
            "approved": False,
        },
        config=config,
    )

    snapshot = graph.get_state(config)

    print("[LangGraph interrupt]")
    print(paused_result)
    print("下一节点：", snapshot.next)

    resumed_result = graph.invoke(
        # Command 是 LangGraph 的流程控制对象
        # 即恢复刚才暂停的执行，并把 False 交给暂停点
        Command(resume=False),
        config=config,
    )

    resumed_snapshot = graph.get_state(config)

    print("[After resume]")
    print(resumed_result)
    print("下一节点：", resumed_snapshot.next)


def main() -> None:
    """选择并运行当前 LangGraph 学习小节。"""
    # langgraph01()
    # langgraph02()
    # langgraph03()
    # langgraph04()
    # langgraph05()
    langgraph06()


if __name__ == "__main__":
    main()
