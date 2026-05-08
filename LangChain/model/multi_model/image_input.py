"""
图像输入
使用视觉模型处理图像

重要提示：
1. 需要支持视觉的模型（如 OpenAI 的 gpt-4o-mini）
2. 请在 images/ 目录下放置你自己的测试图片

使用前准备：
1. 在 images/ 目录下放置以下图片（或使用你自己的图片）:
   - sample.jpg: 任意测试图片
   - text_image.jpg: 包含文字的图片（用于OCR测试）
   - chart.png: 图表图片（用于图表分析）
"""

import os
import base64
from pathlib import Path

from dashscope import MultiModalConversation
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatTongyi

load_dotenv()

model = ChatTongyi(
    # model="qwen-plus",
    model="qwen-vl-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

IMAGES_DIR = "./images"


# ============================================================
# 辅助函数
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """将本地图像编码为 base64"""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def get_mime_type(image_path: str) -> str:
    """根据文件扩展名获取 MIME 类型"""
    ext = Path(image_path).suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return mime_types.get(ext, "image/jpeg")


def create_image_message(text: str, image_path: str):
    """
    创建包含本地图像的消息
    Args:
        text: 文字提示
        image_path: 本地图片路径
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    image_base64 = encode_image_to_base64(image_path)
    mime_type = get_mime_type(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"text": text},
                {"image": f"data:{mime_type};base64,{image_base64}"}
            ]
        }
    ]
    return messages


def check_image_exists(filename: str) -> str:
    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(image_path):
        print(f"\n⚠️ 图片不存在: {image_path}")
        print(f"请将图片 '{filename}' 放入 images/ 目录")
        print("或者修改代码使用你自己的图片路径\n")
        return None
    return str(image_path)


# 示例 1：基本图像描述
def example_1_image_description():
    image_path = check_image_exists("sample.png")
    if not image_path:
        print("跳过此示例")
        return None

    message = create_image_message(
        text="请详细描述这张图片中的内容。用中文回复。",
        image_path=image_path
    )

    print(f"📷 使用图片: {image_path}")
    print("正在分析图片...")
    response = MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen3.6-plus",
        messages=message,
    )
    res = response.output.choices[0].message.content[0]["text"]
    print("\n🤖 描述结果：", res)
    return res


# 示例 2：图像问答
def example_2_image_qa():
    image_path = check_image_exists("sample.jpg")
    if not image_path:
        print("跳过此示例")
        return None

    questions = [
        "图片中有什么主要物体？",
        "图片的整体色调是什么？",
        "这张图片给你什么感觉？"
    ]
    # 首先发送图片
    messages = create_image_message(
        text="我会问你关于这张图片的一些问题。",
        image_path=image_path
    )

    print(f"📷 已加载图片: {image_path}")

    for question in questions:
        print(f"\n❓ 问题: {question}")
        messages.append({'role': 'user', 'content': question})
        response = MultiModalConversation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen3.6-plus",
            messages=messages,
        )
        res = response.output.choices[0].message.content[0]["text"]

        print(f"💬 回答: {res}")


# 示例 3：OCR 文字识别
def example_3_ocr():
    image_path = check_image_exists("text_image.png")
    if not image_path:
        print("提示: 请准备一张包含文字的图片用于 OCR 测试")
        print("跳过此示例")
        return None

    message = create_image_message(
        text="""请仔细查看这张图片，执行以下任务：
1. 描述图片的主要内容
2. 提取图片中所有可见的文字
3. 说明这是什么类型的图片（照片、截图、文档等）

用中文回复。""",
        image_path=image_path
    )

    print(f"📷 使用图片: {image_path}")
    print("正在进行 OCR 识别...")

    response = MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen3.6-plus",
        messages=message,
    )
    res = response.output.choices[0].message.content[0]["text"]
    print("\n📝 识别结果：", res)


# 示例 4：图表分析
def example_4_chart_analysis():
    """
    分析图表数据
    """
    image_path = check_image_exists("chart.png")
    if not image_path:
        print("提示: 请准备一张图表图片（柱状图、折线图等）")
        print("跳过此示例")
        return None

    message = create_image_message(
        text="""请分析这个图表：
1. 这是什么类型的图表？
2. 图表展示了什么数据或信息？
3. 你能从图表中得出什么结论？
4. 如果有数值，请尽可能提取关键数据点

用中文详细回答。""",
        image_path=image_path
    )

    print(f"📷 使用图片: {image_path}")
    print("正在分析图表...")

    response = MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen3.6-plus",
        messages=message,
    )
    res = response.output.choices[0].message.content[0]["text"]
    print("\n📊 分析结果：", res)


if __name__ == "__main__":
    # example_1_image_description()
    # example_2_image_qa()
    # example_3_ocr()
    example_4_chart_analysis()
