# Gold label and overlap review

Status: **human review pending**. Automated checks and agent implementation checks are not a review by Kashish Rata.

For each stable ID in examples, development and evaluation:

- [ ] Restate requested entities, output columns, population and metric without looking at SQL.
- [ ] Check join cardinality; do line-level joins accidentally duplicate invoice totals?
- [ ] Check filters, NULL behavior, distinct versus non-distinct counts and units.
- [ ] Check half-open dates and observed-year versus all-year wording.
- [ ] Check grouped denominators and HAVING versus WHERE.
- [ ] Check all ties versus top-k, monetary rounding, and deterministic ordering.
- [ ] Compare the SQL with the independent Python oracle; inspect actual rows, especially empty outputs.
- [ ] Inspect expected table/column references, CTE lineage and expanded stars.
- [ ] Compare task semantics across splits; reject near-paraphrases even when lexical similarity is low.
- [ ] Record reviewer, date and corrections before changing human_review to approved.

`verify_data` runs lexical Jaccard/sequence overlap checks, exact question uniqueness tests and execution/reference checks. These do not prove absence of semantic overlap. The current five flagged pairs were inspected during implementation and deliberately retained for distinct operations:

| Pair | Distinction requiring human confirmation |
|---|---|
| R12 / M05 | Genre-filtered track/genre join versus artist-filtered track/album join |
| R12 / M09 | Genre-filtered track/genre join versus many-to-many playlist membership |
| R16 / M24 | Playlist track count with empty playlists versus duration sum over nonempty playlists |
| R18 / E12 | Aggregate spending threshold versus a customer city filter |
| R25 / M25 | Assigned-customer count including all employees versus invoice count for employees with assigned customers |

Similar SQL primitives are expected in this single-schema benchmark; copying a solved intent with only a changed constant is disallowed. Human semantic review remains pending for flagged and unflagged pairs. Tune only on D01–D03. Repeated selection on held-out results makes reported comparisons exploratory.
