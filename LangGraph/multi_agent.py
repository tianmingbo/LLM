import os
import sys
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_community.chat_models import ChatTongyi
from typing import TypedDict, List, Literal, Optional
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


class WorkflowState(TypedDict):
    topic: str
    audience: str
    constraints: str

    research_notes: List[str]
    outline: str
    draft: str
    review_feedback: str

    quality_score: int
    revision_count: int
    approved: bool
    need_human_approval: bool
    human_comment: Optional[str]


llm = ChatTongyi(model=os.getenv("QWEN_MODEL", "qwen-plus"), api_key=os.getenv("DASHSCOPE_API_KEY"), streaming=True)


def call_llm(system: str, user: str, stage: str = "LLM") -> str:
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]

    print(f"\n[{stage}] 开始流式输出...")
    chunks: List[str] = []
    for chunk in llm.stream(messages):
        text = getattr(chunk, "content", "") or ""
        if text:
            chunks.append(text)
            print(text, end="", flush=True)

    print(f"\n[{stage}] 输出结束。")
    sys.stdout.flush()
    return "".join(chunks)


def planner_node(state: WorkflowState) -> WorkflowState:
    prompt = f"""
主题: {state['topic']}
受众: {state['audience']}
约束: {state['constraints']}

请给出一个高质量文章提纲（含一级/二级标题）并说明写作策略。
"""
    outline = call_llm(
        "你是资深内容策略师，擅长把复杂主题结构化。",
        prompt,
        stage="planner",
    )
    return {**state, "outline": outline}


def researcher_node(state: WorkflowState) -> WorkflowState:
    prompt = f"""
主题: {state['topic']}
当前提纲: {state['outline']}

请输出 3-5 条高价值研究要点（事实框架、争议点、反例、最佳实践）。
每条尽量简洁、可直接用于写作。
"""
    new_notes = call_llm(
        "你是研究分析师，输出必须信息密度高、避免空话。",
        prompt,
        stage="researcher",
    )
    notes = state["research_notes"] + [new_notes]
    return {**state, "research_notes": notes}


def outline_refiner_node(state: WorkflowState) -> WorkflowState:
    notes_text = "\n\n".join(state["research_notes"][-2:])
    prompt = f"""
主题: {state['topic']}
旧提纲:
{state['outline']}

最新研究笔记:
{notes_text}

请输出改进后的提纲，要求：
1) 结构更清晰
2) 每个一级标题下给出“核心论点”
3) 标注最容易写弱的部分
"""
    new_outline = call_llm(
        "你是总编，擅长提纲打磨和信息架构优化。",
        prompt,
        stage="outline_refiner",
    )
    return {**state, "outline": new_outline}


def writer_node(state: WorkflowState) -> WorkflowState:
    notes_text = "\n\n".join(state["research_notes"])
    prompt = f"""
主题: {state['topic']}
受众: {state['audience']}
约束: {state['constraints']}

提纲:
{state['outline']}

研究笔记:
{notes_text}

请写一篇完整文章，要求：
- 逻辑严谨，案例具体
- 避免套话
- 对受众可执行
"""
    draft = call_llm(
        "你是资深技术写作者，文风专业但不晦涩。",
        prompt,
        stage="writer",
    )
    return {**state, "draft": draft}


def reviewer_node(state: WorkflowState) -> WorkflowState:
    prompt = f"""
请审校以下文章并返回：

1) 总分(0-100)
2) 三个最关键问题
3) 可执行修改建议（按优先级）

文章:
{state['draft']}
"""
    review = call_llm(
        "你是挑剔的审稿人，只给具体可执行建议。",
        prompt,
        stage="reviewer",
    )

    score_text = call_llm(
        "从用户文本中提取一个0到100整数，只输出数字。",
        review,
        stage="score_extractor",
    )
    try:
        score = int("".join(ch for ch in score_text if ch.isdigit())[:3] or "0")
        score = max(0, min(score, 100))
    except Exception:
        score = 0

    return {
        **state,
        "review_feedback": review,
        "quality_score": score,
    }


def revision_node(state: WorkflowState) -> WorkflowState:
    prompt = f"""
你需要根据审稿意见修改文章。

当前文章:
{state['draft']}

审稿意见:
{state['review_feedback']}

请输出改写后的完整文章。
"""
    new_draft = call_llm(
        "你是专业编辑，严格落实反馈，不要只做表面润色。",
        prompt,
        stage="revision",
    )
    return {
        **state,
        "draft": new_draft,
        "revision_count": state["revision_count"] + 1,
    }


def human_approval_node(state: WorkflowState) -> WorkflowState:
    if not state["need_human_approval"]:
        return {**state, "approved": True, "human_comment": "无需人工审批，自动通过"}

    human_input = interrupt(
        {
            "action": "human_approval",
            "message": "请人工审批当前稿件：输入 approve 或 reject，并可附带 comment。",
            "quality_score": state["quality_score"],
            "review_feedback": state["review_feedback"],
            "draft_preview": state["draft"][:800],
        }
    )

    decision = str(human_input.get("decision", "")).strip().lower()
    comment = str(human_input.get("comment", "")).strip() or "无备注"
    approved = decision == "approve"
    return {**state, "approved": approved, "human_comment": comment}


def router_after_review(state: WorkflowState) -> Literal["human_approval", "revision", "end"]:
    if state['revision_count']==0:
        return "human_approval"
    if state["quality_score"] >= 90:
        return "end"

    if state["quality_score"] >= 80 and state["need_human_approval"]:
        return "human_approval"

    if state["revision_count"] >= 2:
        return "end"

    return "revision"


def router_after_human(state: WorkflowState) -> Literal["end", "revision"]:
    if state["approved"]:
        return "end"
    if state["revision_count"] >= 2:
        return "end"
    return "revision"


def build_graph():
    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("outline_refiner", outline_refiner_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("revision", revision_node)
    graph.add_node("human_approval", human_approval_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "outline_refiner")
    graph.add_edge("outline_refiner", "writer")
    graph.add_edge("writer", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        router_after_review,
        {
            "human_approval": "human_approval",
            "revision": "revision",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "human_approval",
        router_after_human,
        {
            "revision": "revision",
            "end": END,
        },
    )

    graph.add_edge("revision", "reviewer")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def run_with_progress(app, input_data, config):
    """流式执行图，实时打印节点进度；结束后返回最终状态。"""
    final_state = None
    for event in app.stream(input_data, config=config, stream_mode="updates"):
        if "__interrupt__" in event:
            final_state = event
            break

        for node_name, delta in event.items():
            print(f"\n[graph] 节点完成: {node_name}")
            if "quality_score" in delta:
                print(f"[graph] 当前质量分: {delta['quality_score']}")
            if "revision_count" in delta:
                print(f"[graph] 当前修订轮次: {delta['revision_count']}")

            if final_state is None:
                final_state = {}
            if isinstance(delta, dict):
                final_state.update(delta)
    print("final_state: ", final_state)
    return final_state or {}


def main():
    app = build_graph()
    png_data = app.get_graph().draw_mermaid_png()

    # 将数据写入当前目录的文件
    with open("./multi_agent.png", "wb") as f:
        f.write(png_data)

    config = {"configurable": {"thread_id": "article-workflow-demo"}}

    initial_state: WorkflowState = {
        "topic": "如何在企业中落地多Agent系统",
        "audience": "有工程背景的技术负责人",
        "constraints": "1500-2200字，强调工程可实施性，不要空泛理论",
        "research_notes": [],
        "outline": "",
        "draft": "",
        "review_feedback": "",
        "quality_score": 0,
        "revision_count": 0,
        "approved": False,
        "need_human_approval": True,
        "human_comment": None,
    }

    print("=== WORKFLOW START (streaming updates) ===")
    result = run_with_progress(app, initial_state, config)

    # 命中人工审批节点时，图会暂停并返回 __interrupt__，然后 resume 继续执行。
    if "__interrupt__" in result:
        print("=== INTERRUPTED FOR HUMAN APPROVAL ===")
        interrupt_payload = result["__interrupt__"][0].value
        print(interrupt_payload["message"])
        print(f"score: {interrupt_payload['quality_score']}")
        print("draft preview:")
        print(interrupt_payload["draft_preview"])
        print("-----")
        print("review feedback:")
        print(interrupt_payload["review_feedback"])

        decision = input("请输入 decision (approve/reject): ").strip().lower()
        comment = input("请输入 comment (可留空): ").strip()

        print("=== RESUME WORKFLOW (streaming updates) ===")
        result = run_with_progress(
            app,
            Command(
                resume={
                    "decision": decision,
                    "comment": comment,
                }
            ),
            config,
        )

    print("=== FINAL SCORE ===")
    print(result["quality_score"])
    print("=== HUMAN COMMENT ===")
    print(result["human_comment"])
    print("=== FINAL DRAFT (head) ===")
    print(result["draft"][:1200])


if __name__ == "__main__":
    main()
