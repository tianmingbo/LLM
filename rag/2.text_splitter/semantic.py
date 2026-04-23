"""Semantic text splitter demo for RAG.

本实现不依赖 langchain-experimental，核心思路：
1) 先按中英文标点切句
2) 用句向量计算相邻句子的余弦相似度
3) 在语义突变点断开并组装 chunk
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


DEFAULT_TEXT = """
自然语言处理是让计算机理解人类语言的技术。
它广泛用于机器翻译、情感分析、智能问答等场景。
通过词向量和大模型，可以实现文本语义的精准表示。

苹果公司最新发布了新款智能手机。
该手机搭载了更强的处理器，拍照能力也大幅提升。
市场分析师预计销量将创下新高。

向量数据库用于高效存储和检索 embedding。
在 RAG 场景中，它负责召回与问题语义最接近的文本片段。
""".strip()


def split_sentences(text: str) -> list[str]:
    """按中英文句号/问号/感叹号切句，保留句内内容。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    merged = " ".join(lines)
    parts = re.split(r"(?<=[。！？.!?])\s+", merged)
    return [p.strip() for p in parts if p.strip()]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def adjacent_similarities(vectors: np.ndarray) -> list[float]:
    return [
        cosine_similarity(vectors[i], vectors[i + 1])
        for i in range(len(vectors) - 1)
    ]


def pick_breakpoints(
    sims: list[float],
    percentile: float,
    min_chunk_sentences: int,
) -> set[int]:
    """返回断点索引集合（断点 i 表示在句子 i 与 i+1 之间切分）。"""
    if not sims:
        return set()
    threshold = float(np.percentile(sims, percentile))
    raw_points = {i for i, sim in enumerate(sims) if sim <= threshold}

    # 过滤过密断点，保证 chunk 至少包含 min_chunk_sentences 句
    filtered_points: set[int] = set()
    last_start = 0
    for i in sorted(raw_points):
        chunk_len = i - last_start + 1
        if chunk_len >= min_chunk_sentences:
            filtered_points.add(i)
            last_start = i + 1
    return filtered_points


def build_chunks(
    sentences: list[str],
    breakpoints: set[int],
    overlap_sentences: int,
) -> list[str]:
    if not sentences:
        return []

    chunks: list[str] = []
    start = 0
    for i in range(len(sentences) - 1):
        if i in breakpoints:
            end = i + 1
            chunk = " ".join(sentences[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            start = max(0, end - overlap_sentences)

    tail = " ".join(sentences[start:]).strip()
    if tail:
        chunks.append(tail)
    return chunks


def get_embeddings(model_name: str):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise SystemExit(
                "缺少 embedding 依赖，请先安装: "
                "pip install langchain-huggingface sentence-transformers"
            )

    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception:
        raise SystemExit(
            "初始化 HuggingFaceEmbeddings 失败。请检查依赖版本兼容性："
            "sentence-transformers、transformers、torch、numpy。"
            "常见修复：升级 torch，或降级 numpy 到 <2。"
        )


def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic splitter demo (RAG)")
    parser.add_argument("--text-file", type=str, default=None, help="输入文本文件（UTF-8）")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="HuggingFace embedding 模型",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=25.0,
        help="相似度分位数阈值（0~100，越小切分越少）",
    )
    parser.add_argument(
        "--min-chunk-sentences",
        type=int,
        default=2,
        help="每个 chunk 至少包含的句子数",
    )
    parser.add_argument(
        "--overlap-sentences",
        type=int,
        default=1,
        help="chunk 间句子重叠数",
    )
    args = parser.parse_args()

    if not (0 <= args.percentile <= 100):
        raise SystemExit("--percentile 必须在 0~100 之间")
    if args.min_chunk_sentences < 1:
        raise SystemExit("--min-chunk-sentences 必须 >= 1")
    if args.overlap_sentences < 0:
        raise SystemExit("--overlap-sentences 不能为负数")

    text = load_text(args.text_file)
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        print("文本句子数不足，无法进行语义切分。")
        print(text.strip())
        return

    embeddings = get_embeddings(args.model_name)
    vectors = np.array(embeddings.embed_documents(sentences))
    sims = adjacent_similarities(vectors)
    breakpoints = pick_breakpoints(
        sims=sims,
        percentile=args.percentile,
        min_chunk_sentences=args.min_chunk_sentences,
    )
    chunks = build_chunks(
        sentences=sentences,
        breakpoints=breakpoints,
        overlap_sentences=args.overlap_sentences,
    )

    print(f"句子数: {len(sentences)}")
    print(f"相邻相似度数: {len(sims)}")
    print(f"断点数: {len(breakpoints)}")
    print(f"chunk 数: {len(chunks)}")

    for idx, chunk in enumerate(chunks, start=1):
        print(f"\n=== Chunk {idx} | 字符数: {len(chunk)} ===")
        print(chunk)


if __name__ == "__main__":
    main()
