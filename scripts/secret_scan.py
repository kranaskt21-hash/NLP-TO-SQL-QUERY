"""Inspect staged UTF-8 text without printing potential secret contents."""

import re
import subprocess

PATTERNS = [
    r"AIza[0-9A-Za-z_-]{35}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"github_pat_[A-Za-z0-9_]{30,}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
]


def main():
    names = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"], text=True
    ).splitlines()
    flagged = []
    for name in names:
        if name == ".streamlit/secrets.toml" or name.startswith(
            (".cache/", ".venv/", "results/local/")
        ):
            flagged.append(name)
            continue
        data = subprocess.check_output(["git", "show", ":" + name])
        if b"\x00" in data:
            continue
        content = data.decode("utf-8", errors="replace")
        if any(re.search(pattern, content) for pattern in PATTERNS):
            flagged.append(name)
    print(
        "Staged secret scan:",
        "PASS" if not flagged else "FAIL",
        "; files inspected:",
        len(names),
    )
    for name in flagged:
        print("Review file:", name)
    if flagged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
