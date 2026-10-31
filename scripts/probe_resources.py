"""Measure this machine, never a cloud guarantee. No API requests."""

import json
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

import psutil

from nl2sql.config import ROOT, Config


def main():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    start = time.perf_counter()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            "--server.headless=true",
            "--server.address=127.0.0.1",
            "--server.port=" + str(port),
            "--browser.gatherUsageStats=false",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "server_health": False,
    }
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for _ in range(50):
            if process.poll() is not None:
                break
            try:
                with opener.open(
                    f"http://127.0.0.1:{port}/_stcore/health", timeout=0.5
                ) as response:
                    report["server_health"] = response.status == 200
                    break
            except Exception:
                time.sleep(0.1)
        report["server_health_seconds"] = time.perf_counter() - start
        report["server_rss_mib"] = (
            sum(
                p.memory_info().rss
                for p in [psutil.Process(process.pid)]
                + psutil.Process(process.pid).children(recursive=True)
            )
            / 2**20
        )
        # A server health endpoint alone does not execute the Streamlit script.
        from streamlit.testing.v1 import AppTest

        tick = time.perf_counter()
        app = AppTest.from_file(ROOT / "app.py", default_timeout=30).run()
        report["ui_app_test_seconds"] = time.perf_counter() - tick
        report["ui_exceptions"] = len(app.exception)
        from nl2sql.embeddings import Embedder
        from nl2sql.retrieval import Retriever
        from nl2sql.schema_docs import documents

        tick = time.perf_counter()
        hits = Retriever(documents(), Embedder(), Config(), "schema").search(
            "customer invoice spending", 8
        )
        report["warm_download_dense_query_seconds"] = time.perf_counter() - tick
        report["in_process_ui_plus_embedding_rss_mib"] = (
            psutil.Process().memory_info().rss / 2**20
        )
        report["schema_hits"] = len(hits)
        report["note"] = (
            "Windows local measurements; model files already downloaded. RSS is not peak memory or cloud assurance."
        )
    finally:
        for child in psutil.Process(process.pid).children(recursive=True):
            child.terminate()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    (ROOT / "results/resource_probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
