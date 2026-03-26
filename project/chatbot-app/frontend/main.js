const API_BASE = window.CHATBOT_API_BASE || "http://127.0.0.1:8000";

const roleSelect = document.getElementById("roleSelect");
const chatWindow = document.getElementById("chatWindow");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const newSessionBtn = document.getElementById("newSessionBtn");
const sessionList = document.getElementById("sessionList");
const themeToggleBtn = document.getElementById("themeToggleBtn");
const sessionSearchInput = document.getElementById("sessionSearchInput");
const toggleSessionPanelBtn = document.getElementById("toggleSessionPanelBtn");
const containerEl = document.querySelector(".container");

let sessions = JSON.parse(localStorage.getItem("chat_sessions") || "[]");
let sessionNames = JSON.parse(localStorage.getItem("chat_session_names") || "{}");
let sessionNameManual = JSON.parse(localStorage.getItem("chat_session_name_manual") || "{}");
let sessionAutoSummarized = JSON.parse(localStorage.getItem("chat_session_auto_summarized") || "{}");
let sessionId = localStorage.getItem("chat_session_id") || "";
let sessionKeyword = "";
let sessionPanelCollapsed = localStorage.getItem("chat_session_panel_collapsed") === "true";

if (!Array.isArray(sessions)) {
  sessions = [];
}
if (!sessionNames || typeof sessionNames !== "object" || Array.isArray(sessionNames)) {
  sessionNames = {};
}
if (!sessionNameManual || typeof sessionNameManual !== "object" || Array.isArray(sessionNameManual)) {
  sessionNameManual = {};
}
if (!sessionAutoSummarized || typeof sessionAutoSummarized !== "object" || Array.isArray(sessionAutoSummarized)) {
  sessionAutoSummarized = {};
}
if (!sessionId) {
  sessionId = crypto.randomUUID();
}
if (!sessions.includes(sessionId)) {
  sessions.unshift(sessionId);
}

function saveSessionState() {
  localStorage.setItem("chat_session_id", sessionId);
  localStorage.setItem("chat_sessions", JSON.stringify(sessions));
  localStorage.setItem("chat_session_names", JSON.stringify(sessionNames));
  localStorage.setItem("chat_session_name_manual", JSON.stringify(sessionNameManual));
  localStorage.setItem("chat_session_auto_summarized", JSON.stringify(sessionAutoSummarized));
}

function shortId(id) {
  return id.length > 12 ? `${id.slice(0, 8)}...${id.slice(-4)}` : id;
}

function getSessionName(id) {
  return sessionNames[id] || `会话 ${shortId(id)}`;
}

function normalizeText(value) {
  return (value || "").toLowerCase().trim();
}

function setSessionName(id, name) {
  const clean = (name || "").trim();
  if (!clean) {
    delete sessionNames[id];
  } else {
    sessionNames[id] = clean;
  }
}

function ensureSessionName(id, fallbackText = "") {
  if (sessionNames[id]) return;
  const text = fallbackText.trim();
  if (text) {
    sessionNames[id] = text.slice(0, 24);
    return;
  }
  sessionNames[id] = `新会话 ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  sessionNameManual[id] = false;
}

function applyTheme(theme) {
  const target = theme === "dark" ? "dark" : "light";
  document.body.setAttribute("data-theme", target);
  localStorage.setItem("chat_theme", target);
  themeToggleBtn.textContent = target === "dark" ? "浅色模式" : "深色模式";
}

function applySessionPanelState(collapsed) {
  sessionPanelCollapsed = !!collapsed;
  localStorage.setItem("chat_session_panel_collapsed", String(sessionPanelCollapsed));
  if (sessionPanelCollapsed) {
    containerEl.classList.add("session-collapsed");
    toggleSessionPanelBtn.title = "展开会话";
    toggleSessionPanelBtn.setAttribute("aria-label", "展开会话");
  } else {
    containerEl.classList.remove("session-collapsed");
    toggleSessionPanelBtn.title = "收起会话";
    toggleSessionPanelBtn.setAttribute("aria-label", "收起会话");
  }
}

function renderSessionList() {
  sessionList.innerHTML = "";
  const keyword = normalizeText(sessionKeyword);
  const visibleSessions = sessions.filter((id) => {
    if (!keyword) return true;
    const name = normalizeText(getSessionName(id));
    return name.includes(keyword) || normalizeText(id).includes(keyword);
  });

  if (!visibleSessions.length) {
    const empty = document.createElement("div");
    empty.className = "session-empty";
    empty.textContent = "没有匹配的会话";
    sessionList.appendChild(empty);
    return;
  }

  visibleSessions.forEach((id) => {
    const item = document.createElement("div");
    item.className = `session-item ${id === sessionId ? "active" : ""}`;

    const switchBtn = document.createElement("button");
    switchBtn.type = "button";
    switchBtn.className = "session-switch-btn";
    switchBtn.textContent = getSessionName(id);
    switchBtn.title = id;
    switchBtn.addEventListener("click", async () => {
      if (id === sessionId) return;
      sessionId = id;
      saveSessionState();
      renderSessionList();
      await loadHistory();
    });

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "session-rename-btn";
    renameBtn.textContent = "重命名";
    renameBtn.title = `重命名会话 ${id}`;
    renameBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      const nextName = window.prompt("输入会话名称", sessionNames[id] || "");
      if (nextName === null) return;
      setSessionName(id, nextName);
      sessionNameManual[id] = true;
      sessionAutoSummarized[id] = true;
      saveSessionState();
      renderSessionList();
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "session-delete-btn";
    deleteBtn.textContent = "删除";
    deleteBtn.title = `删除会话 ${id}`;
    deleteBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      await deleteSession(id);
    });

    item.appendChild(switchBtn);
    item.appendChild(renameBtn);
    item.appendChild(deleteBtn);
    sessionList.appendChild(item);
  });
}

async function deleteSession(id) {
  try {
    await fetch(`${API_BASE}/sessions/${id}`, { method: "DELETE" });
  } catch (_) {
    // ignore backend failure, still update local state
  }

  sessions = sessions.filter((sid) => sid !== id);
  delete sessionNames[id];
  delete sessionNameManual[id];
  delete sessionAutoSummarized[id];

  if (!sessions.length) {
    sessionId = crypto.randomUUID();
    sessions.unshift(sessionId);
  } else if (id === sessionId) {
    sessionId = sessions[0];
  }

  saveSessionState();
  renderSessionList();
  await loadHistory();
}

function renderMessage(role, content) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = `${role === "user" ? "你" : role === "assistant" ? "AI" : "系统"}: ${content}`;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return div;
}

async function fetchRoles() {
  const res = await fetch(`${API_BASE}/roles`);
  if (!res.ok) {
    throw new Error("加载角色失败");
  }

  const data = await res.json();
  roleSelect.innerHTML = "";

  data.roles.forEach((role, idx) => {
    const option = document.createElement("option");
    option.value = role.key;
    option.textContent = role.name;
    if (idx === 0) {
      option.selected = true;
    }
    roleSelect.appendChild(option);
  });
}

async function loadHistory() {
  chatWindow.innerHTML = "";

  const res = await fetch(`${API_BASE}/sessions/${sessionId}`);
  if (!res.ok) {
    renderMessage("system", "历史记录加载失败，继续新会话。");
    return;
  }

  const data = await res.json();
  if (!data.messages.length) {
    renderMessage("system", `会话已创建：${sessionId}`);
    return;
  }

  data.messages.forEach((msg) => {
    renderMessage(msg.role, msg.content);
  });
}

function parseSSEChunk(raw, onEvent) {
  const lines = raw.split("\n");
  for (const line of lines) {
    if (!line.startsWith("data: ")) continue;
    const payload = line.slice(6).trim();
    if (!payload) continue;
    onEvent(JSON.parse(payload));
  }
}

async function sendMessageStream(text, onToken) {
  const payload = {
    message: text,
    role: roleSelect.value,
    session_id: sessionId,
  };

  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(errText || "请求失败");
  }

  if (!res.body) {
    throw new Error("浏览器不支持流式响应");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let donePayload = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      parseSSEChunk(part, (event) => {
        if (event.type === "token") {
          onToken(event.text);
        } else if (event.type === "done") {
          donePayload = event;
        } else if (event.type === "error") {
          throw new Error(event.message || "流式输出失败");
        }
      });
    }
  }

  if (!donePayload) {
    throw new Error("流式输出提前结束");
  }

  return donePayload;
}

async function refreshSessionTitleFromSummary(id) {
  if (sessionNameManual[id]) return;
  if (sessionAutoSummarized[id]) return;
  try {
    const res = await fetch(`${API_BASE}/sessions/${id}/summary`, { method: "POST" });
    if (!res.ok) return;
    const data = await res.json();
    const title = (data.title || "").trim();
    if (!title) return;
    sessionNames[id] = title.slice(0, 24);
    sessionNameManual[id] = false;
    sessionAutoSummarized[id] = true;
    saveSessionState();
    renderSessionList();
  } catch (_) {
    // ignore summarization failure
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  ensureSessionName(sessionId, text);
  sessionNameManual[sessionId] = false;
  saveSessionState();
  renderSessionList();

  renderMessage("user", text);
  messageInput.value = "";
  messageInput.focus();
  let assistantText = "";
  let hasToken = false;
  const assistantNode = renderMessage("assistant", "思考中");
  assistantNode.classList.add("thinking");

  try {
    const data = await sendMessageStream(text, (token) => {
      if (!hasToken) {
        hasToken = true;
        assistantNode.classList.remove("thinking");
        assistantNode.classList.add("streaming");
      }
      assistantText += token;
      assistantNode.textContent = `AI: ${assistantText || "..."}`;
      chatWindow.scrollTop = chatWindow.scrollHeight;
    });
    sessionId = data.session_id;
    if (!sessions.includes(sessionId)) {
      sessions.unshift(sessionId);
    }
    saveSessionState();
    renderSessionList();
    await refreshSessionTitleFromSummary(sessionId);
    assistantNode.classList.remove("streaming");
    assistantNode.classList.remove("thinking");
    assistantNode.textContent = `AI: ${assistantText || "（空响应）"}`;
  } catch (err) {
    assistantNode.remove();
    renderMessage("system", `发送失败：${err.message}`);
  }
});

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.requestSubmit();
  }
});

newSessionBtn.addEventListener("click", async () => {
  sessionId = crypto.randomUUID();
  if (!sessions.includes(sessionId)) {
    sessions.unshift(sessionId);
  }
  ensureSessionName(sessionId);
  sessionNameManual[sessionId] = false;
  sessionAutoSummarized[sessionId] = false;
  saveSessionState();
  renderSessionList();
  chatWindow.innerHTML = "";
  renderMessage("system", `已开启新会话：${sessionId}`);
});

sessionSearchInput.addEventListener("input", () => {
  sessionKeyword = sessionSearchInput.value || "";
  renderSessionList();
});

themeToggleBtn.addEventListener("click", () => {
  const current = document.body.getAttribute("data-theme") || "light";
  applyTheme(current === "dark" ? "light" : "dark");
});

toggleSessionPanelBtn.addEventListener("click", () => {
  applySessionPanelState(!sessionPanelCollapsed);
});

async function bootstrap() {
  try {
    applyTheme(localStorage.getItem("chat_theme") || "light");
    applySessionPanelState(sessionPanelCollapsed);
    ensureSessionName(sessionId);
    await fetchRoles();
    saveSessionState();
    renderSessionList();
    await loadHistory();
  } catch (err) {
    renderMessage("system", `初始化失败：${err.message}`);
  }
}

bootstrap();
