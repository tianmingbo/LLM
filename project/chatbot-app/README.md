# 星语助手

一个基于 `FastAPI + LangChain + Redis + 通义千问` 的前后端分离聊天应用。

## 功能特性

- 多角色聊天（通用助手 / 耐心老师 / 编程教练 / 翻译专家）
- 流式输出（SSE）
- Redis 会话记忆（`RedisChatMessageHistory`）
- 多会话管理（新建、切换、删除、搜索、重命名）
- 会话标题自动总结（每个会话只自动总结一次，手动重命名后不覆盖）
- 深色/浅色主题切换（持久化）
- 左侧会话栏可折叠（持久化）

## 项目结构

```text
chatbot-app/
  backend/
    main.py
    requirements.txt
    .env.example
    app/
      memory.py
      roles.py
      schemas.py
  frontend/
    index.html
    main.js
    style.css
    favicon.svg
```

## 一、启动 Redis

本地 Redis：

```bash
redis-server
```

## 二、启动后端

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY
uvicorn main:app --reload --port 8000
```

后端文档：`http://127.0.0.1:8000/docs`

## 三、启动前端

```bash
cd frontend
python3 -m http.server 5173
```

前端地址：`http://127.0.0.1:5173`

## 环境变量

`backend/.env`：

```env
DASHSCOPE_API_KEY=sk-xxxx
QWEN_MODEL=qwen-plus
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
REDIS_URL=redis://localhost:6379/0
MEMORY_TTL_SECONDS=86400
MEMORY_KEY_PREFIX=chat:
```

## API 概览

- `GET /health` 健康检查
- `GET /roles` 获取角色列表
- `POST /chat` 非流式聊天
- `POST /chat/stream` 流式聊天（SSE）
- `GET /sessions/{session_id}` 获取会话历史
- `DELETE /sessions/{session_id}` 删除会话历史
- `POST /sessions/{session_id}/summary` 生成会话标题
