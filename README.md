# LlangChain

一个最小可运行的 LangChain + Qwen 示例，密钥配置放在 `.env`。

## 1) 安装依赖

```bash
python3 -m pip install langchain langchain-openai langchain-community python-dotenv fastapi uvicorn
```

## 2) 配置环境变量

复制模板并填入你的 DashScope Key：

```bash
cp .env.example .env
```

`.env` 示例：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
QWEN_MODEL=qwen-plus
```

## 3) 运行示例

```bash
python3 qwen_langchain_example.py
```

默认会问模型一个问题并输出回答。

## 4) 启动 Qwen API 服务（支持流式）

```bash
python3 -m model.qwen.qwen_async
```

默认监听 `0.0.0.0:8000`。

- 健康检查：`GET /health`
- 普通回答：`POST /api/qwen/chat`
- 流式回答：`POST /api/qwen/chat/stream`

普通调用示例：

```bash
curl -X POST "http://127.0.0.1:8000/api/qwen/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_prompt":"你是谁？","system_prompt":"你是一个有帮助的助手。"}'
```

流式调用示例：

```bash
curl -N -X POST "http://127.0.0.1:8000/api/qwen/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"user_prompt":"介绍一下你自己","system_prompt":"你是一个有帮助的助手。"}'
```
