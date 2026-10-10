from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate


class CourseAnalysis(BaseModel):  # 继承pydantic后，会帮助完成校验
    name: str = Field(description="课程名称")

    difficulty: int = Field(
        description="课程难度，取值为1-5",
        ge=1,
        le=5,
    )

    topics: list[str] = Field(description="课程的核心知识点")


def main() -> None:
    course = CourseAnalysis(
        name="langchain",
        difficulty=3,
        topics=["模型", "提示词", "输出解析器"],
    )

    print(course)
    # print(type(course).__name__)
    # print(course.name)
    # print(course.topics)
    # print(course.model_dump())

    # print("\n[测试错误的课程数据]")

    # try:
    #     invalid_course = CourseAnalysis(
    #         name="LangChain",
    #         difficulty=10,
    #         topics=["模型", "提示词"],
    #     )

    #     print(invalid_course)

    # except ValidationError as error:
    #     print("数据校验失败")
    #     print(error)
    #     print(error.errors())

    print("\n[模型结构化输出]")

    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )

    # 让后面的invoke返回的以courseanaysis的格式
    structured_model = model.with_structured_output(
        CourseAnalysis,
        include_raw=True,  # 把模型的原始消息和解析情况也一起返回
    )

    # result = structured_model.invoke(
    #     "请分析 LangChain 这门课程，"
    #     "给出课程名称、1到5的难度和三个核心知识点。"
    # )

    # print("\n[完整返回结果]")
    # print(result)
    # print(type(result).__name__)

    # raw_message = result["raw"]
    # parsed_course = result["parsed"]
    # parsing_error = result["parsing_error"]

    # print("\n[原始模型消息]")
    # print(type(raw_message).__name__)
    # print(raw_message.content)
    # print(raw_message.tool_calls)

    # print("\n[解析后的课程对象]")
    # print(type(parsed_course).__name__)
    # print(parsed_course)
    # print(parsed_course.name)
    # print(parsed_course.difficulty)
    # print(parsed_course.topics)

    # print("\n[解析错误]")
    # print(parsing_error)

    # 组合chatprompttemplate ，形成课复用的业务链
    analysis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名课程分析助手，"
                "请根据学习者背景客观分析课程。",
            ),
            (
                "human",
                "课程名称：{course_name}\n"
                "学习者背景：{learner_background}",
            ),
        ]
    )

    analysis_chain = analysis_prompt | structured_model
    result = analysis_chain.invoke(
        {
            "course_name": "LangChain",
            "learner_background": "有 Go 和 RAG 项目经验，正在学习 Python",
        }
    )

    parsed_course = result["parsed"]
    parsing_error = result["parsing_error"]

    print("\n[Prompt + 结构化模型]")
    print("完整结果类型：", type(result).__name__)

    if parsing_error is not None:
        print("结构化输出解析失败：")
        print(parsing_error)

    elif parsed_course is not None:
        print("解析结果类型：", type(parsed_course).__name__)
        print("课程数据：", parsed_course.model_dump())

    else:
        print("模型没有返回可用的结构化数据")


if __name__ == "__main__":
    main()
