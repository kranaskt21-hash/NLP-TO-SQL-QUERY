import importlib

from lc.chains import sql_database
from nl2sql.config import REVISIONS

print("Configured pinned revisions:", REVISIONS)
modules = [
    "streamlit",
    "pandas",
    "numpy",
    "plotly",
    "faiss",
    "google.genai",
    "langchain_google_genai",
    "sentence_transformers",
    "rank_bm25",
    "sqlglot",
    "langchain_community.vectorstores.faiss",
    "langchain_community.retrievers.bm25",
    "langchain_core",
    "sqlalchemy",
]
for m in modules:
    importlib.import_module(m)
    print("import OK:", m)

print("SQLDatabase tables", sql_database().get_usable_table_names())
