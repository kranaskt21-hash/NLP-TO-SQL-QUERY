from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from nl2sql.llm import Generation, get_api_key


class LangChainModel:
    def __init__(self, config):
        self.config = config

    def invoke(self, prompt):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(
            model=self.config.model,
            api_key=get_api_key(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_output_tokens,
            max_retries=0,
            thinking_level=self.config.thinking_level,
        )
        structured = model.with_structured_output(
            Generation, method="json_schema", include_raw=True
        )
        chain = (
            ChatPromptTemplate.from_messages([("human", "{input}")])
            | structured
            | RunnableLambda(lambda x: x)
        )
        output = chain.invoke({"input": prompt})
        if output.get("parsing_error"):
            raise ValueError("Structured output parse failure")
        raw = output["raw"]
        usage = getattr(raw, "usage_metadata", None)
        return output["parsed"], (
            {
                "input": usage.get("input_tokens"),
                "output": usage.get("output_tokens"),
                "total": usage.get("total_tokens"),
            }
            if usage
            else None
        )


def sql_database():
    # Introspection only; query execution always goes through nl2sql.db.execute.
    from langchain_community.utilities import SQLDatabase
    from sqlalchemy import create_engine

    def creator():
        import sqlite3

        from nl2sql.config import DB_PATH

        c = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
        c.execute("PRAGMA query_only=ON")
        c.enable_load_extension(False)
        return c

    return SQLDatabase(
        create_engine("sqlite://", creator=creator), sample_rows_in_table_info=0
    )
