import sqlite3
from collections import deque

from nl2sql.config import DB_PATH
from nl2sql.db import schema


def expand(selected):
    known = schema()
    graph = {t: [] for t in known}
    edges = {}
    con = sqlite3.connect(DB_PATH.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        for t in known:
            for f in con.execute(f'PRAGMA foreign_key_list("{t}")'):
                graph[t].append(f[2])
                graph[f[2]].append(t)
                edges[frozenset((t, f[2]))] = (t + "." + f[3], f[2] + "." + f[4])
    finally:
        con.close()
    selected = sorted(set(selected))
    result = set(selected)
    disconnected = [[t] for t in selected if t not in known]
    joins = set()
    for i, a in enumerate(selected):
        for b in selected[i + 1 :]:
            queue = deque([[a]])
            seen = {a}
            found = None
            while queue:
                path = queue.popleft()
                if path[-1] == b:
                    found = path
                    break
                for n in sorted(graph.get(path[-1], [])):
                    if n not in seen:
                        seen.add(n)
                        queue.append(path + [n])
            if found:
                result.update(found)
                for x, y in zip(found, found[1:]):
                    joins.update(edges[frozenset((x, y))])
            else:
                disconnected.append([a, b])
    return {
        "tables": sorted(result),
        "bridge_tables": sorted(result - set(selected)),
        "join_columns": sorted(joins),
        "disconnected": disconnected,
    }
