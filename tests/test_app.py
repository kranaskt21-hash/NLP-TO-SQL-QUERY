from streamlit.testing.v1 import AppTest

from nl2sql.config import ROOT


def test_app_without_key():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == "NLP TO SQL QUERY"
    assert [t.label for t in app.tabs] == [
        "Ask",
        "Compare",
        "Experiments",
        "Error Analysis",
    ]
