from mcp.server.fastmcp import FastMCP


mcp = FastMCP("learning-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """计算两个整数的和。"""
    return a + b


@mcp.tool()
def divide(dividend: float, divisor: float) -> float:
    """计算两个数的除法，除数不能为零。"""
    if divisor == 0:
        raise ValueError("除数不能为零")

    return dividend / divisor


def mcp_server01() -> None:
    """通过 stdio 启动包含计算工具的 MCP Server。"""
    mcp.run(transport="stdio")


def main() -> None:
    mcp_server01()


if __name__ == "__main__":
    main()
