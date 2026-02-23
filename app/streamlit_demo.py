"""GibsGraph — Streamlit demo UI."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="GibsGraph",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("🕸️ GibsGraph")
    st.caption("GraphRAG + LangGraph agent for Neo4j knowledge graph reasoning")

    # Sidebar — config
    with st.sidebar:
        st.header("Configuration")
        neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        neo4j_user = st.text_input("Username", value="neo4j")
        neo4j_pass = st.text_input("Password", type="password")
        st.divider()
        st.caption("Built by [gibbrdev](https://gibs.dev)")

    # Tabs
    tab_ask, tab_ingest, tab_viz = st.tabs(["💬 Ask", "📥 Ingest", "🗺️ Visualize"])

    with tab_ask:
        st.subheader("Ask the knowledge graph")
        query = st.text_area(
            "Your question",
            placeholder="What regulations apply to fintech companies in the EU?",
            height=100,
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            run = st.button("Ask", type="primary", use_container_width=True)

        if run and query:
            with st.spinner("Reasoning over knowledge graph..."):
                try:
                    from gibsgraph import Graph
                    g = Graph(neo4j_uri, password=neo4j_pass or None)
                    result = g.ask(query)

                    st.success("Answer")
                    st.write(result.answer)

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Confidence", f"{result.confidence:.0%}")
                        st.metric("Nodes retrieved", result.nodes_retrieved)
                    with col_b:
                        if result.bloom_url:
                            st.link_button("Open in Neo4j Bloom", result.bloom_url)

                    with st.expander("Cypher used"):
                        st.code(result.cypher or "N/A", language="cypher")

                    if result.visualization:
                        with st.expander("Mermaid diagram"):
                            st.code(result.visualization, language="mermaid")

                    if result.errors:
                        st.warning(f"Warnings: {result.errors}")

                except Exception as exc:
                    st.error(f"Error: {exc}")

    with tab_ingest:
        st.subheader("Ingest text into the knowledge graph")
        st.warning("⚠️ Requires NEO4J_READ_ONLY=false in your .env")
        uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
        text_input = st.text_area("Or paste text directly", height=200)

        if st.button("Ingest", type="primary"):
            content = None
            if uploaded:
                content = uploaded.read().decode("utf-8")
            elif text_input:
                content = text_input

            if content:
                with st.spinner("Building knowledge graph..."):
                    try:
                        from gibsgraph import Graph
                        g = Graph(neo4j_uri, password=neo4j_pass or None, read_only=False)
                        result = g.ingest(content, source=uploaded.name if uploaded else "manual")
                        st.success(f"✓ {result}")
                    except Exception as exc:
                        st.error(f"Error: {exc}")
            else:
                st.info("Please upload a file or paste text.")

    with tab_viz:
        st.subheader("Graph visualization")
        st.info("Run a query in the Ask tab first, then return here to visualize.")
        st.markdown("""
        **Coming soon:**
        - Interactive PyVis graph rendering
        - Mermaid diagram export
        - Neo4j Bloom deep links
        """)


if __name__ == "__main__":
    main()
