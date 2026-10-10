from dataclasses import dataclass
import asyncio
import time
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool


@dataclass
class HarnessConfig:
    timeout_seconds: float = 15.0  # 限制整个 Agent 最多运行多少秒
    recursion_limit: int = 8  # 限制 Agent 图最多执行多少步


@dataclass  # 会自动生成初始化方法
class HarnessResult:
    success: bool
    answer: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


async def fake_agent(question: str) -> str:
    """模拟异步 Agent，根据问题返回固定格式的答案。"""
    await asyncio.sleep(0.1)
    return f"模拟回答：{question}"


async def failing_agent(question: str) -> str:
    """模拟执行过程中发生异常的 Agent。"""
    raise RuntimeError(f"无法处理问题：{question}")


async def run_with_harness(
        agent_function,
        question: str,
        config: HarnessConfig,
) -> HarnessResult:
    """在超时限制内执行 Agent，并返回统一结果。"""
    started_at = time.perf_counter()  # 计算一段代码经过了多长时间
    print("[Harness start]")
    print("问题：", question)
    print("超时限制：", config.timeout_seconds)

    try:
        async with asyncio.timeout(config.timeout_seconds):
            answer = await agent_function(question)

        result = HarnessResult(
            success=True,
            answer=answer,
            error=None,
        )

    except TimeoutError:
        result = HarnessResult(
            success=False,
            answer=None,
            error="Agent 执行超时",
        )

    except Exception as error:
        result = HarnessResult(
            success=False,
            answer=None,
            error=f"{type(error).__name__}: {error}",
        )

    finally:
        elapsed_seconds = time.perf_counter() - started_at
        elapsed_seconds = round(elapsed_seconds, 3)  # 保留三位小数
        print("[Harness end]")
        print("耗时：", elapsed_seconds, "秒")

    result.elapsed_seconds = elapsed_seconds
    return result


def harness01() -> None:
    """定义 Harness 的运行配置和统一结果结构。"""

    config = HarnessConfig(
        timeout_seconds=10.0,
        recursion_limit=6,
    )

    print("[Harness config]")
    print(config)


async def harness02() -> None:
    """调用模拟 Agent，并把答案包装成统一的 HarnessResult。"""
    question = "请计算 10 加 20。"

    answer = await fake_agent(question)

    result = HarnessResult(
        success=True,
        answer=answer,
        error=None,
    )

    print("[Harness result]")
    print(result)


async def harness03() -> None:
    """通过 Harness 执行模拟 Agent，并应用超时限制。"""
    config = HarnessConfig(
        timeout_seconds=1.0,
        recursion_limit=6,
    )

    result = await run_with_harness(
        agent_function=fake_agent,
        question="请计算 10 加 20。",
        config=config,
    )

    print("[Harness with timeout]")
    print(result)


async def harness04() -> None:
    """验证 Harness 能把 Agent 异常转换成统一失败结果。"""
    config = HarnessConfig(
        timeout_seconds=1.0,
        recursion_limit=6,
    )

    result = await run_with_harness(
        agent_function=failing_agent,
        question="异常测试",
        config=config,
    )

    print("[Harness error result]")
    print(result)


async def harness05(model) -> None:
    """通过harness允许真正的langchain agent"""
    config = HarnessConfig(
        timeout_seconds=20.0,
        recursion_limit=6,
    )

    agent = create_agent(
        model=model,
        tools=[add_numbers],
        system_prompt="你是计算助手，计算时必须使用工具。",
    )

    async def invoke_agent(question: str) -> str:
        agent_state = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            config={
                "recursion_limit": config.recursion_limit,
            },
        )

        final_message = agent_state["messages"][-1]
        return final_message.content

    result = await run_with_harness(
        agent_function=invoke_agent,
        question="请计算 10 加 20。",
        config=config,
    )

    print("[Real agent result]")
    print(result)


async def harness06() -> None:
    """使用固定样例自动检查 Harness 的成功和异常结果。"""
    config = HarnessConfig(
        timeout_seconds=1.0,
        recursion_limit=6,
    )

    success_result = await run_with_harness(
        agent_function=fake_agent,
        question="固定成功样例",
        config=config,
    )

    # assert类似于if，如果成功就没事，失败就报错
    assert success_result.success is True
    assert success_result.answer == "模拟回答：固定成功样例"
    assert success_result.error is None
    assert success_result.elapsed_seconds < config.timeout_seconds

    error_result = await run_with_harness(
        agent_function=failing_agent,
        question="固定异常案例",
        config=config,
    )

    assert error_result.success is False
    assert error_result.answer is None
    assert error_result.error is not None
    assert "RuntimeError" in error_result.error

    print("[Harness checks]")
    print("全部固定样例通过")


def main() -> None:
    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )

    # harness01()
    # asyncio.run(harness02())
    # asyncio.run(harness03())
    # asyncio.run(harness04())
    # asyncio.run(harness05(model))
    asyncio.run(harness06())


if __name__ == "__main__":
    main()
