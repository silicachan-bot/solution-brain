from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brain.compose.menu import build_menu
from brain.config import LANCEDB_DIR, RETRIEVAL_TOP_K
from brain.store.pattern_db import PatternDB
from brain.store.retriever import retrieve_patterns
from brain.viewer import filter_patterns, format_pattern_summary, group_origins_by_example, sort_patterns


st.set_page_config(page_title="solution-brain 模式库", layout="wide")
st.title("solution-brain 模式库查看器")
st.caption("查看 PatternCard 样本，检查提取质量，并测试检索效果")


@st.cache_resource
def get_db() -> PatternDB:
    return PatternDB(LANCEDB_DIR)


def render_pattern_card(card) -> None:
    with st.expander(format_pattern_summary(card), expanded=False):
        origins_by_example = group_origins_by_example(card)
        st.markdown(f"**ID**: `{card.id}`")
        st.markdown(f"**模板**: `{card.template}`")
        st.markdown(f"**描述**: {card.description}")
        st.markdown(f"**来源**: {card.source}")
        st.markdown(
            "**频率**: "
            f"recent={card.frequency.recent}, "
            f"medium={card.frequency.medium}, "
            f"long_term={card.frequency.long_term}, "
            f"total={card.frequency.total}, "
            f"freshness={card.frequency.freshness:.2f}"
        )
        st.markdown(
            f"**时间**: created_at={card.created_at.isoformat()} | "
            f"updated_at={card.updated_at.isoformat()}"
        )
        st.markdown("**例句**")
        for example in card.examples:
            st.markdown(f"- {example}")
            origins = origins_by_example.get(example, [])
            if not origins:
                st.caption("来源缺失")
                continue
            for origin in origins:
                title = origin.video_title or "无标题"
                st.caption(f"{title} ({origin.bvid})")
                st.markdown(f"上文评论：{origin.parent_message}")
                st.markdown(f"回复评论：{origin.reply_message}")


def main() -> None:
    try:
        db = get_db()
        all_patterns = db.list_all()
    except Exception as exc:
        st.error(f"读取数据库失败：{exc}")
        return

    st.metric("PatternCard 总数", len(all_patterns))

    if not all_patterns:
        st.info("当前还没有 PatternCard。请先运行 scripts/run_pipeline.py 生成数据。")
        return

    browse_tab, retrieve_tab = st.tabs(["样本浏览", "检索测试"])

    with browse_tab:
        search_query = st.text_input(
            "搜索 PatternCard",
            placeholder="按模板、描述或例句搜索",
        )
        sort_by = st.selectbox(
            "排序方式",
            options=["updated_at", "freshness", "template"],
            format_func=lambda x: {
                "updated_at": "最近更新",
                "freshness": "热度 freshness",
                "template": "模板",
            }[x],
        )

        visible_patterns = sort_patterns(
            filter_patterns(all_patterns, search_query),
            sort_by,
        )
        st.write(f"当前显示 {len(visible_patterns)} 条")

        for card in visible_patterns:
            render_pattern_card(card)

    with retrieve_tab:
        query_text = st.text_area(
            "输入检索文本",
            placeholder="例如：这也太离谱了吧，我直接好家伙",
            height=120,
        )
        top_k = st.number_input(
            "返回条数 top_k",
            min_value=1,
            max_value=20,
            value=RETRIEVAL_TOP_K,
        )

        if st.button("执行检索"):
            if not query_text.strip():
                st.warning("请输入检索文本。")
            else:
                try:
                    results = retrieve_patterns(db, query_text, top_k=int(top_k))
                except Exception as exc:
                    st.error(f"检索失败：{exc}")
                else:
                    if not results:
                        st.info("没有匹配到模式。")
                    else:
                        st.subheader("检索结果")
                        for card in results:
                            render_pattern_card(card)

                        st.subheader("菜单预览")
                        st.code(build_menu(results), language="text")


if __name__ == "__main__":
    main()
