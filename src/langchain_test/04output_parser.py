from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


def main() -> None:
    load_dotenv()

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        temperature=0,
    )

    # ---------- 第一部分：StrOutputParser ----------
    parser = StrOutputParser()

    # 不用 parser：拿到的是 AIMessage，要手动 .content
    # response = model.invoke("用一句话解释什么是提示词。")
    # print(type(response).__name__)   # AIMessage
    # print(response.content)

    # 用 parser：用 | 把模型和解析器串成链
    chain = model | parser
    text = chain.invoke("用一句话解释什么是提示词。")
    # print(type(text).__name__)       # TextAccessor（str 的兼容子类）
    # print(isinstance(text, str))     # True
    print(text)

    # ---------- 第二部分：JsonOutputParser ----------
    json_parser = JsonOutputParser()

    json_text = """
    {
        "name": "langchain",
        "language": "Python",
        "difficulty": 3
    }
    """

    json_result = json_parser.invoke(json_text)

    # print("\n[JsonOutputParser 手动解析]")
    print(json_result)
    # print(type(json_result).__name__)
    # print(json_result["name"])
    # print(json_result["difficulty"])

    # ---------- 第三部分：模型生成 JSON ----------
    json_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一名课程分析助手。\n"
                "请严格按照下面的格式要求回答：\n"
                "{format_instructions}",
            ),
            (
                "human",
                "请分析课程：{course_name}",
            ),
        ]
    )

    print("\n[模板变量]")
    print(json_prompt.input_variables)

    format_instructions = json_parser.get_format_instructions()

    print("\n[JSON 格式要求]")
    print(format_instructions)

    json_prompt = json_prompt.partial(
        format_instructions=format_instructions
    )
    print(json_prompt.input_variables)

    json_chain = json_prompt | model | json_parser

    course = json_chain.invoke(
        {
            "course_name": "LangChain",
        }
    )

    print("\n[模型生成并解析后的结果]")
    print(course)
    # print(type(course).__name__)

    # print("\n[读取字段]")
    # for key, value in course.items():
    #     print(f"{key} = {value}")

    # ---------- 第四部分：处理解析失败 ----------
    invalid_text = "这不是合法的json"

    try:
        result = json_parser.invoke(invalid_text)
        print(result)
    except OutputParserException as error:  # 是 LangChain 专门表示“输出解析失败”的异常类型
        print("\n[JSON 解析失败]")
        print(type(error).__name__)
        print(error)


if __name__ == "__main__":
    main()
