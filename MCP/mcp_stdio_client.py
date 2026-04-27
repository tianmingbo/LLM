import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

async def main():
    # 配置：启动本地python子进程运行mcp服务端
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_stdio_server.py"],
        env=None
    )

    async with stdio_client(server_params) as streams:
        async with ClientSession(*streams) as session:
            # 握手初始化
            await session.initialize()
            print("✅ MCP stdio 连接成功")

            # 1. 列举所有可用工具
            response = await session.list_tools()
            tools = response.tools
            print("\n📋 可用工具列表：")
            for t in tools.tools:
                print(f"- {t.name}: {t.description}")

            # 2. 调用加法工具
            # res1 = await session.call_tool("calc_add", {"a": 99, "b": 66})
            # print("\n🔧 calc_add(99,66) =", res1.content[0].text)
            #
            # # 3. 调用时间工具
            # current_time = await session.call_tool("get_current_time")
            # print("🔧 当前时间：", current_time.content[0].text)
            #
            # # 4. 读取资源
            # res3 = await session.read_resource("note://info")
            # print("\n📄 读取资源：", res3.contents[0].text)
            #
            # # 新增：5. 获取并使用Prompt模板（天气播报）
            # weather_prompt = await session.get_prompt(
            #     "weather_report_prompt",
            #     {"city": "北京", "time": current_time.content[0].text}
            # )
            # print("\n💬 天气播报Prompt：", weather_prompt.messages[0].content.text)
            #
            # # 新增：6. 获取并使用Prompt模板（数学计算）
            # math_prompt = await session.get_prompt(
            #     "math_calc_prompt",
            #     {"num1": 100, "num2": 25, "operation": "除"}
            # )
            # print("\n💬 数学计算Prompt：", math_prompt.messages[0].content.text)

if __name__ == "__main__":
    asyncio.run(main())