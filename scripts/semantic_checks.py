"""Independent Python oracles for representative boundary/aggregation/ranking cases.
Oracles operate on whole source tables and use Python joins, groups, filters and rankings.
"""

from collections import defaultdict
from statistics import median

from nl2sql.db import execute
from shared.interfaces import Execution
from shared.metrics import execution_equal


def check_semantics(groups, con):
    golds = {r["id"]: r for rs in groups.values() for r in rs}
    out = []

    def check(id, expected, description):
        actual = execute(golds[id]["sql"], evaluation=True, row_limit=100000)
        out.append(
            {
                "id": id,
                "check": description,
                "passed": execution_equal(
                    actual, Execution(actual.columns, expected), ordered=True
                ),
            }
        )

    invoices = list(
        con.execute(
            "SELECT InvoiceId,CustomerId,InvoiceDate,BillingCountry,Total FROM Invoice"
        )
    )
    tracks = list(
        con.execute("SELECT TrackId,GenreId,Milliseconds,UnitPrice,Bytes FROM Track")
    )
    customers = list(con.execute("SELECT CustomerId,Country,Company FROM Customer"))
    check(
        "E01",
        [(sum(c[1] == "Brazil" for c in customers),)],
        "Python count of Brazilian customers",
    )
    check("E04", [(sum(i[4] for i in invoices),)], "Python sum of invoice totals")
    check(
        "E05",
        [(min(t[3] for t in tracks), max(t[3] for t in tracks))],
        "Python price extrema",
    )
    check(
        "E06",
        [(c[0],) for c in sorted(customers) if c[2] is None],
        "Python null company filter",
    )
    check(
        "E08",
        [(sum(i[3] == "Canada" for i in invoices),)],
        "Python Canada invoice count",
    )
    check(
        "E09",
        [(t[0],) for t in sorted(tracks) if t[2] > 600000],
        "Strict ten-minute boundary",
    )
    check("E13", [(sum(t[4] for t in tracks),)], "Python sum of track bytes")
    check(
        "E14",
        [(i[0],) for i in sorted(invoices) if "2012-03-01" <= i[2] < "2012-04-01"],
        "Half-open March date interval",
    )
    check(
        "M02",
        [
            (g, sum(t[2] for t in ts) / len(ts))
            for g, ts in sorted(
                {
                    g: [t for t in tracks if t[1] == g] for g in {t[1] for t in tracks}
                }.items()
            )
        ],
        "Python grouped mean duration",
    )
    spend = defaultdict(float)
    for i in invoices:
        spend[i[1]] += i[4]
    check(
        "H01",
        [(c,) for c, s in sorted(spend.items()) if abs(s - max(spend.values())) < 1e-8],
        "All top spending ties",
    )
    expected = []
    for genre in sorted({t[1] for t in tracks}):
        for pos, t in enumerate(
            sorted([t for t in tracks if t[1] == genre], key=lambda t: (-t[2], t[0]))[
                :2
            ],
            1,
        ):
            expected.append((genre, t[0], pos))
    check("H02", expected, "Per-genre top two and explicit duration tie breaking")
    running = 0.0
    expected = []
    for i in sorted(invoices, key=lambda i: (i[2], i[0])):
        running += i[4]
        expected.append((i[0], i[2], running))
    check("H03", expected, "Python running sum with date/ID ordering")
    check("H13", [(median(i[4] for i in invoices),)], "Independent Python median")
    years = {i[2][:4] for i in invoices}
    check(
        "H14",
        [
            (c,)
            for c in sorted({i[1] for i in invoices})
            if {i[2][:4] for i in invoices if i[1] == c} == years
        ],
        "Python set coverage of observed invoice years",
    )
    check(
        "R02",
        [(con.execute("SELECT COUNT(*) FROM Album").fetchone()[0],)],
        "Catalog count invariant",
    )
    check(
        "R15",
        [
            (
                t[0],
                con.execute(
                    "SELECT Name FROM Track WHERE TrackId=?", (t[0],)
                ).fetchone()[0],
                t[2],
            )
            for t in sorted(tracks, key=lambda t: (-t[2], t[0]))[:3]
        ],
        "Python longest tracks and tie ordering",
    )

    # Remaining cases use an independent in-memory relational representation.
    def table(name):
        cursor = con.execute('SELECT * FROM "' + name + '"')
        cols = [c[0] for c in cursor.description]
        return [dict(zip(cols, row)) for row in cursor]

    T = {
        name: table(name)
        for name in (
            "Artist",
            "Album",
            "Track",
            "Customer",
            "Employee",
            "Invoice",
            "InvoiceLine",
            "Genre",
            "MediaType",
            "Playlist",
            "PlaylistTrack",
        )
    }
    idx = {
        name: {r[next(iter(r))]: r for r in data}
        for name, data in T.items()
        if name != "PlaylistTrack"
    }
    A, L, C, E, invoices_data, N, G, B, P, PT = [
        T[n]
        for n in (
            "Album",
            "Track",
            "Customer",
            "Employee",
            "Invoice",
            "InvoiceLine",
            "Genre",
            "MediaType",
            "Playlist",
            "PlaylistTrack",
        )
    ]

    def group(data, key):
        groups = defaultdict(list)
        for row in data:
            groups[row[key]].append(row)
        return groups

    ti = group(L, "AlbumId")
    tg = group(L, "GenreId")
    ai = group(A, "ArtistId")
    ic = group(invoices_data, "CustomerId")
    ni = group(N, "InvoiceId")
    nt = group(N, "TrackId")
    pp = group(PT, "PlaylistId")
    tp = group(PT, "TrackId")

    def rows(id, values):
        check(id, values, "Independent Python relational oracle for stated question")

    rows(
        "E02",
        [
            (r["PlaylistId"], r["Name"])
            for r in sorted(P, key=lambda r: (r["Name"], r["PlaylistId"]))
        ],
    )
    rows("E03", [(r["GenreId"], r["Name"]) for r in G if 5 <= r["GenreId"] <= 10])
    rows("E07", [(r["EmployeeId"],) for r in E if r["BirthDate"] < "1970-01-01"])
    rows(
        "E10", [(r["AlbumId"],) for r in A if r["Title"].lower().startswith("greatest")]
    )
    rows("E11", [(len({r["Composer"] for r in L if r["Composer"] is not None}),)])
    rows("E12", [(r["CustomerId"],) for r in C if r["City"] in ("Paris", "Berlin")])
    rows(
        "E15",
        [(r["MediaTypeId"], r["Name"]) for r in B if "video" in r["Name"].lower()],
    )
    rows("E16", [(sum(r["ReportsTo"] is None for r in E),)])
    rows("E17", [(r["CustomerId"], r["Email"]) for r in C[:10]])
    rows("E18", [(len(PT),)])
    rows(
        "E19",
        [(r["ArtistId"],) for r in T["Artist"] if r["Name"].lower().startswith("the ")],
    )
    rows("E20", [(r["InvoiceLineId"],) for r in N if r["Quantity"] >= 2])
    rows(
        "M01",
        [(c, idx["Customer"][c]["FirstName"], len(rs)) for c, rs in sorted(ic.items())],
    )
    rows("M03", [(a, len(rs)) for a, rs in sorted(ai.items()) if len(rs) >= 3])
    rows("M04", [(i, len(rs)) for i, rs in sorted(ni.items())])
    rows(
        "M05",
        [
            (r["TrackId"], r["Name"], idx["Album"][r["AlbumId"]]["Title"])
            for r in L
            if idx["Album"][r["AlbumId"]]["ArtistId"] == 12
        ],
    )
    months = defaultdict(float)
    for r in invoices_data:
        if r["InvoiceDate"][:4] == "2011":
            months[r["InvoiceDate"][:7]] += r["Total"]
    rows("M06", sorted(months.items()))
    rows(
        "M07",
        [
            (r["CustomerId"], r["FirstName"], r["LastName"])
            for r in C
            if (
                idx["Employee"][r["SupportRepId"]]["FirstName"],
                idx["Employee"][r["SupportRepId"]]["LastName"],
            )
            == ("Jane", "Peacock")
        ],
    )
    rows(
        "M08",
        [
            (c, len(rs))
            for c, rs in sorted(group(invoices_data, "BillingCountry").items())
            if len(rs) >= 10
        ],
    )
    rows(
        "M09",
        [
            (r["TrackId"], idx["Track"][r["TrackId"]]["Name"])
            for r in sorted(pp[5], key=lambda r: r["TrackId"])
        ],
    )
    rows(
        "M10",
        [
            (idx["MediaType"][m]["Name"], sum(r["Bytes"] for r in rs))
            for m, rs in sorted(group(L, "MediaTypeId").items())
        ],
    )
    rows(
        "M11",
        [
            (c,)
            for c in sorted(
                {
                    r["CustomerId"]
                    for r in invoices_data
                    if r["BillingCountry"]
                    != idx["Customer"][r["CustomerId"]]["Country"]
                }
            )
        ],
    )
    rows(
        "M12",
        [
            (
                r["EmployeeId"],
                r["LastName"],
                idx["Employee"][r["ReportsTo"]]["LastName"],
            )
            for r in E
            if r["ReportsTo"] is not None
        ],
    )
    rows(
        "M13",
        [
            (
                sum(
                    r["UnitPrice"] * r["Quantity"]
                    for r in N
                    if idx["Genre"][idx["Track"][r["TrackId"]]["GenreId"]]["Name"]
                    == "Jazz"
                ),
            )
        ],
    )
    rows(
        "M14",
        [
            (a,)
            for a, rs in sorted(ti.items())
            if sum(r["Milliseconds"] for r in rs) / len(rs) > 300000
        ],
    )
    rows(
        "M15",
        [
            (c, len({r["City"] for r in rs}))
            for c, rs in sorted(group(C, "Country").items())
        ],
    )
    rows("M16", [(t, len(rs)) for t, rs in sorted(tp.items()) if len(rs) > 3])
    rows(
        "M17",
        [
            (c,)
            for c, rs in sorted(ic.items())
            if min(r["InvoiceDate"] for r in rs)[:4] == "2009"
        ],
    )
    rows("M18", [(a, len({r["GenreId"] for r in rs})) for a, rs in sorted(ti.items())])
    rows(
        "M19",
        [(i,) for i, rs in sorted(ni.items()) if len({r["TrackId"] for r in rs}) >= 10],
    )
    artists = {idx["Album"][r["AlbumId"]]["ArtistId"] for r in L if r["GenreId"] == 2}
    rows("M20", [(a, idx["Artist"][a]["Name"]) for a in sorted(artists)])
    rows(
        "M21",
        [
            (c, round(sum(r["Total"] for r in rs) / len(rs) + 1e-12, 2))
            for c, rs in sorted(ic.items())
        ],
    )
    rows(
        "M22",
        [
            (r["InvoiceId"], idx["Customer"][r["CustomerId"]]["Email"])
            for r in invoices_data
            if r["Total"] == 0.99
        ],
    )
    sold_genres = {idx["Track"][r["TrackId"]]["GenreId"] for r in N}
    rows("M23", [(r["GenreId"],) for r in G if r["GenreId"] not in sold_genres])
    rows(
        "M24",
        [
            (p, sum(idx["Track"][r["TrackId"]]["Milliseconds"] for r in rs))
            for p, rs in sorted(pp.items())
        ],
    )
    rows(
        "M25",
        [
            (e, sum(len(ic.get(r["CustomerId"], [])) for r in rs))
            for e, rs in sorted(group(C, "SupportRepId").items())
        ],
    )
    customer_genres = defaultdict(set)
    country_units = defaultdict(lambda: defaultdict(int))
    artist_revenue = defaultdict(float)
    genre_revenue = defaultdict(float)
    for r in N:
        invoice = idx["Invoice"][r["InvoiceId"]]
        track = idx["Track"][r["TrackId"]]
        customer_genres[invoice["CustomerId"]].add(track["GenreId"])
        country_units[invoice["BillingCountry"]][r["TrackId"]] += r["Quantity"]
        artist_revenue[idx["Album"][track["AlbumId"]]["ArtistId"]] += (
            r["UnitPrice"] * r["Quantity"]
        )
        genre_revenue[track["GenreId"]] += r["UnitPrice"] * r["Quantity"]
    rows("H04", [(c,) for c in sorted(customer_genres) if {1, 2} <= customer_genres[c]])
    rows(
        "H05",
        sorted(
            [(a, round(v, 2)) for a, v in artist_revenue.items()],
            key=lambda r: (-r[1], r[0]),
        )[:5],
    )
    genre_avg = {
        g: sum(round(r["UnitPrice"] * 100) for r in rs) / len(rs)
        for g, rs in tg.items()
    }
    rows(
        "H06",
        [
            (r["TrackId"],)
            for r in L
            if round(r["UnitPrice"] * 100) > genre_avg[r["GenreId"]]
        ],
    )
    country_rev = {
        c: round(sum(r["Total"] for r in rs), 2)
        for c, rs in group(invoices_data, "BillingCountry").items()
    }
    # Floating point sums follow the same invoice order, and ties use exact stored sums.
    ranks = {
        v: i + 1 for i, v in enumerate(sorted(set(country_rev.values()), reverse=True))
    }
    rows(
        "H07",
        [
            (c, v, ranks[v])
            for c, v in sorted(country_rev.items(), key=lambda r: (ranks[r[1]], r[0]))
        ],
    )
    rows(
        "H08",
        [(a,) for a, rs in sorted(ti.items()) if all(r["TrackId"] in nt for r in rs)],
    )
    yr = defaultdict(float)
    for r in invoices_data:
        yr[r["InvoiceDate"][:4]] += r["Total"]
    observed = sorted(yr)
    rows(
        "H09",
        [
            (y, yr[y], None if j == 0 else yr[y] - yr[observed[j - 1]])
            for j, y in enumerate(observed)
        ],
    )
    rows(
        "H10",
        [
            (c,)
            for c, rs in sorted(ic.items())
            if any(r["Total"] > 2 * sum(x["Total"] for x in rs) / len(rs) for r in rs)
        ],
    )
    country_spend = defaultdict(dict)
    for c, s in spend.items():
        country_spend[idx["Customer"][c]["Country"]][c] = round(s, 2)
    rows(
        "H11",
        [
            (country, c, s)
            for country, cs in sorted(country_spend.items())
            for c, s in sorted(cs.items())
            if s == max(cs.values())
        ],
    )
    revenue_total = sum(genre_revenue.values())
    rows(
        "H12", [(g, 100 * r / revenue_total) for g, r in sorted(genre_revenue.items())]
    )
    rows(
        "H15",
        [
            (c, t, u, pos)
            for c, units in sorted(country_units.items())
            for pos, (t, u) in enumerate(
                sorted(units.items(), key=lambda r: (-r[1], r[0]))[:3], 1
            )
        ],
    )
    rows(
        "R01", [(r["Name"],) for r in sorted(T["Artist"], key=lambda r: r["Name"])[:5]]
    )
    rows(
        "R03",
        [
            (r["Name"],)
            for r in sorted(L, key=lambda r: r["Name"])
            if r["UnitPrice"] > 1
        ],
    )
    rows("R04", [(c,) for c in sorted({r["Country"] for r in C})])
    rows("R05", [(r["FirstName"], r["LastName"], r["Title"]) for r in E])
    rows(
        "R06", [(r["InvoiceId"], r["Total"]) for r in invoices_data if r["Total"] > 20]
    )
    rows(
        "R07",
        [
            (
                r["MediaTypeId"],
                r["Name"],
                sum(t["MediaTypeId"] == r["MediaTypeId"] for t in L),
            )
            for r in B
        ],
    )
    rows("R08", [(r["Title"], idx["Artist"][r["ArtistId"]]["Name"]) for r in A])
    rows("R09", [(c, len(rs)) for c, rs in sorted(group(C, "Country").items())])
    rows("R10", [(sum(r["Milliseconds"] for r in L) / len(L),)])
    rows("R11", sorted(country_rev.items()))
    rows("R12", [(r["Name"], idx["Genre"][1]["Name"]) for r in L if r["GenreId"] == 1])
    rows(
        "R13",
        [(r["InvoiceId"],) for r in invoices_data if r["InvoiceDate"][:4] == "2010"],
    )
    rows("R14", [(r["CustomerId"],) for r in C if r["CustomerId"] not in ic])
    rows("R16", [(r["Name"], len(pp.get(r["PlaylistId"], []))) for r in P])
    rows("R17", [(sum(r["Quantity"] for r in N),)])
    rows("R18", [(c, s) for c, s in sorted(spend.items()) if s > 40])
    rows("R19", [(r["Name"],) for r in L if "love" in r["Name"].lower()])
    rows(
        "R20",
        [
            (r["EmployeeId"], r["FirstName"], r["LastName"])
            for r in E
            if r["ReportsTo"] == 2
        ],
    )
    rows("R21", [(a, min(r["UnitPrice"] for r in rs)) for a, rs in sorted(ti.items())])
    rows("R22", [(r["TrackId"],) for r in L if r["TrackId"] not in tp])
    rows(
        "R23",
        [
            (
                y,
                sum(r["Total"] for r in invoices_data if r["InvoiceDate"][:4] == y)
                / sum(r["InvoiceDate"][:4] == y for r in invoices_data),
            )
            for y in observed
        ],
    )
    genre_units = defaultdict(int)
    for r in N:
        genre_units[idx["Track"][r["TrackId"]]["GenreId"]] += r["Quantity"]
    rows(
        "R24",
        [
            (g, idx["Genre"][g]["Name"], u)
            for g, u in sorted(genre_units.items(), key=lambda r: (-r[1], r[0]))[:5]
        ],
    )
    rows(
        "R25",
        [
            (r["EmployeeId"], sum(c["SupportRepId"] == r["EmployeeId"] for c in C))
            for r in E
        ],
    )
    rows("R26", [(a, len(rs)) for a, rs in sorted(ti.items()) if len(rs) > 15])
    rows(
        "R27",
        [
            (c, s)
            for c, s in sorted(spend.items())
            if s > sum(spend.values()) / len(spend)
        ],
    )
    rows(
        "R28",
        [
            (t, sum(r["UnitPrice"] * r["Quantity"] for r in rs))
            for t, rs in sorted(nt.items())
        ],
    )
    rows(
        "R29", [(c, max(r["InvoiceDate"] for r in rs)) for c, rs in sorted(ic.items())]
    )
    rows(
        "R30",
        [(r["ArtistId"], r["Name"]) for r in T["Artist"] if r["ArtistId"] not in ai],
    )
    rows("D01", [(len(tg.get(7, [])),)])
    london = defaultdict(float)
    for r in invoices_data:
        if r["BillingCity"] == "London":
            london[r["InvoiceDate"][:4]] += r["Total"]
    rows("D02", sorted(london.items()))
    rows(
        "D03",
        [
            (a, len(rs))
            for a, rs in sorted(ti.items(), key=lambda pair: (-len(pair[1]), pair[0]))[
                :4
            ]
        ],
    )
    assert (
        len(out)
        == len({r["id"] for r in out})
        == len({r["id"] for rs in groups.values() for r in rs})
    )
    return out
