from dataclasses import replace

import streamlit as st

from nl2sql.config import BGE, GEMINI_EMBED, MINILM, Config
from nl2sql.llm import Budget, MissingKey, get_api_key
from nl2sql.pipeline import Pipeline
from shared.ui import show_error_analysis, show_experiments, show_result

st.set_page_config(
    page_title="NLP TO SQL QUERY", page_icon=":material/database:", layout="wide"
)
st.title("NLP TO SQL QUERY")
st.caption(
    "Kashish Rata · Natural language, schema retrieval and guarded SQLite execution"
)
if "budget" not in st.session_state:
    st.session_state.budget = Budget()
with st.sidebar:
    st.subheader("Pipeline settings")
    implementation = st.selectbox("Implementation", ["scratch", "langchain"])
    generation_model = st.text_input(
        "Gemini generation model",
        value=Config().model,
        help="Use a generation model accessible to your API key. The connection check below lists models from the first catalog page.",
    )
    retrieval = st.selectbox("Retrieval", ["hybrid", "dense", "bm25"])
    embedding = st.selectbox("Embedding model", [MINILM, BGE, GEMINI_EMBED])
    schema_mode = st.selectbox("Schema context", ["retrieved", "full"])
    examples_k = st.select_slider("Solved examples", options=[0, 2, 4, 8], value=4)
    rerank = st.toggle("Cross-encoder reranking", value=False)
    fk = st.toggle("Foreign-key expansion", value=True)
    corrections = st.slider("Maximum corrections", 0, 2, 2)
    noise = st.selectbox("Synthetic distractor tables", [0, 25, 50, 100])
    candidates = st.slider("Retrieval candidate count", 8, 40, 20)
    schema_k = st.slider("Final schema document count", 2, 20, 8)
    quota = st.number_input(
        "Session API call limit", min_value=1, max_value=1000, value=20
    )
    st.session_state.budget.limit = int(quota)
    st.caption(f"API calls used: {st.session_state.budget.calls} / {quota}")
    st.link_button("GitHub · Kashish Rata", "https://github.com/kranaskt21-hash")
config = Config(
    implementation=implementation,
    model=generation_model.strip(),
    retrieval=retrieval,
    embedding=embedding,
    schema_mode=schema_mode,
    examples_k=examples_k,
    rerank=rerank,
    fk_expansion=fk,
    corrections=corrections,
    distractors=noise,
    candidates=candidates,
    schema_k=schema_k,
)
with st.sidebar.expander("Gemini connection check"):
    st.caption(
        "Sends a model-list request and a tiny structured generation request (up to 2 API calls from your session budget). May use paid quota. No user question is sent."
    )
    if st.button("Check connection", key="check_gemini"):
        from nl2sql.diagnostics import check_connection

        with st.spinner("Checking the selected Gemini model..."):
            st.session_state.connection_check = check_connection(
                config, st.session_state.budget
            )
    if "connection_check" in st.session_state:
        check = st.session_state.connection_check
        if check.status == "success":
            st.success(check.message)
        else:
            st.warning(check.message)
        st.json(check.to_dict())
try:
    get_api_key()
except MissingKey:
    st.info(
        "Add GEMINI_API_KEY to .streamlit/secrets.toml or your environment to generate SQL. Schema browsing and offline tests work without a key."
    )
ask, compare, experiments, errors = st.tabs(
    ["Ask", "Compare", "Experiments", "Error Analysis"], on_change="rerun"
)
if ask.open:
    with ask:
        samples = [
            "How many tracks belong to genre ID 7?",
            "Show revenue billed in London by year, sorted by year.",
            "List the four albums with the most catalog tracks, ties by album ID.",
        ]
        sample = st.selectbox("Sample development question", samples)
        with st.form("ask_form"):
            question = st.text_area("Business question", value=sample, max_chars=2000)
            submitted = st.form_submit_button("Generate and run", type="primary")
        if submitted and question.strip():
            with st.spinner("Retrieving context, generating and validating SQL…"):
                st.session_state.ask_result = Pipeline(
                    config, st.session_state.budget
                ).run(question.strip())
        if "ask_result" in st.session_state:
            show_result(st.session_state.ask_result)
        with st.expander("Browse the Chinook schema"):
            from nl2sql.schema_docs import documents

            for doc in documents():
                if doc.kind == "table":
                    st.write(doc.text)
if compare.open:
    with compare:
        st.caption(
            "Both implementations share settings and safety. Separate cache keys prevent shared generated outputs from counting as independent model runs."
        )
        with st.form("compare_form"):
            question = st.text_area(
                "Question for both implementations",
                value="How many tracks belong to genre ID 7?",
                max_chars=2000,
            )
            submitted = st.form_submit_button("Compare pipelines", type="primary")
        if submitted and question.strip():
            with st.spinner("Running scratch and LangChain pipelines…"):
                st.session_state.comparison = [
                    Pipeline(
                        replace(config, implementation=impl), st.session_state.budget
                    ).run(question.strip())
                    for impl in ("scratch", "langchain")
                ]
        if "comparison" in st.session_state:
            for col, impl, result in zip(
                st.columns(2), ("Scratch", "LangChain"), st.session_state.comparison
            ):
                with col:
                    st.subheader(impl)
                    show_result(result)
if experiments.open:
    with experiments:
        show_experiments()
if errors.open:
    with errors:
        show_error_analysis()
