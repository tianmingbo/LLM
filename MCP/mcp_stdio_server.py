from mcp.server.fastmcp import FastMCP

# 初始化MCP服务（默认stdio模式）
mcp = FastMCP("Stdio-Demo-Server")

# 自定义工具 - LLM可调用
@mcp.tool()
def calc_add(a: int, b: int) -> int:
    """简单加法计算
    Args:
        a: 数字1
        b: 数字2
    """
    return a + b

@mcp.tool()
def get_current_time() -> str:
    """获取当前系统时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 资源文件
@mcp.resource("note://info")
def read_note() -> str:
    """读取内置说明文档"""
    return "这是一个标准 Stdio 模式 MCP 服务，提供本地工具调用、资源读取、Prompt模板能力"

# 新增：Prompt模板（LLM可获取、复用的提示词模板）
@mcp.prompt()
def weather_report_prompt(city: str, time: str) -> str:
    """生成天气播报提示词模板
    用于让LLM根据城市和时间，生成自然、简洁的天气播报文案
    Args:
        city: 城市名称
        time: 当前时间（格式：YYYY-MM-DD HH:MM:SS）
    """
    return f"""请根据以下信息，生成一段友好的天气播报：
1. 城市：{city}
2. 当前时间：{time}
3. 要求：语气亲切，简洁明了，不超过30字，结尾加上一句温馨提示"""

@mcp.prompt()
def math_calc_prompt(num1: int, num2: int, operation: str) -> str:
    """生成数学计算提示词模板
    用于让LLM清晰理解计算需求，给出步骤和结果
    Args:
        num1: 第一个数字
        num2: 第二个数字
        operation: 运算类型（加、减、乘、除）
    """
    return f"""请完成以下数学计算，要求先写计算步骤，再写最终结果：
数字1：{num1}
数字2：{num2}
运算类型：{operation}
步骤清晰，结果准确，无需多余描述"""

# 启动：默认stdio运行
if __name__ == "__main__":
    mcp.run()