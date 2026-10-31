# Interview preparation — NLP TO SQL QUERY

Author: Kashish Rata · https://github.com/kranaskt21-hash

## 30-second introduction

“I built a Python and Streamlit system that translates business questions into read-only SQLite queries over Chinook. It retrieves schema and solved examples separately, expands foreign-key paths and asks Gemini for structured SQL or abstention. Both scratch and LangChain implementations use the same validator, restricted executor and evaluation code. I built a 60-question evaluation set and controlled ablations so I can test whether retrieval choices actually help. Live model accuracy is not yet measured; offline semantic checks and mocked pipeline tests are separate evidence.”

## Two-minute walkthrough

“Start with the Ask tab: the question retrieves table/column documents and similar solved examples. Dense vectors are normalized and searched with exact FAISS inner product; sparse retrieval is BM25. RRF combines ranks instead of adding incompatible similarity scales. A cross-encoder is optional. The schema and example stores have independent budgets.

“Retrieval can select tables that need a bridge. I model foreign keys as a graph and use sorted breadth-first searches for deterministic shortest paths, then add complete selected/bridge table context and required join columns. This improves context coverage but does not prove that the model will choose the right join.

“Gemini returns SQL or an abstention. The question and retrieved documents are explicitly untrusted data. SQLGlot checks the AST, scoped aliases and catalog references before a read-only SQLite connection executes anything. SQLite adds query-only mode, an authorizer, disabled extensions, a deadline and bounded fetching. Eligible errors get at most two repairs; API authentication and quota errors are handled separately.

“Compare runs the same question through scratch SDK generation and LangChain retrieval/LCEL generation. Evaluation separates the 30-example store, three development questions and 60 held-out questions. Independent Python oracles verify all gold queries against the data, while human review is still pending. The experiment runner controls one factor at a time, records cache provenance and reports bootstrap intervals. I do not claim RAG or LangChain is more accurate until actual matched live runs establish that.”

## Questions and answers

1. **What is schema linking?** Mapping phrases such as ‘money spent’ to tables/columns such as Invoice.Total and Customer.CustomerId. Embeddings help semantic matching; BM25 helps exact identifiers. The model still has to select the right metric and join grain.

2. **Why two retrieval stores?** A long example list should not crowd out schema evidence. Independent candidate/final budgets make recall and prompt-size tradeoffs explicit. The evaluation split is never a source of examples.

3. **How does dynamic few-shot selection work?** Retrieve solved training questions by the new question and include their SQL as examples. The LC path uses a BaseExampleSelector implementation backed by real LC retrieval. Zero and random examples are useful controls; similar examples can also anchor the wrong logic.

4. **Why normalize vectors and use IndexFlatIP?** Inner product of unit vectors equals cosine similarity. With only 75 real schema documents and 30 examples, exact search is cheap and avoids approximation parameters. A larger corpus might justify HNSW or IVF after measuring recall, latency and update costs.

5. **Why RRF rather than adding BM25 and cosine scores?** Their numeric scales differ. RRF scores rank positions, making fusion simple without score calibration. It discards score magnitude and may dilute a strong signal; evaluate dense, sparse and hybrid conditions rather than assuming fusion wins.

6. **What does the cross-encoder add?** Joint question/document encoding can distinguish candidates that a bi-encoder missed. It costs CPU time/memory and cannot recover a relevant document excluded from the candidate pool. Reranker benefit is unmeasured here.

7. **Why FK graph expansion?** Customer and Track require Invoice and InvoiceLine to connect purchases. Sorted shortest-path search adds those bridges and join columns. A shortest FK path is a structural hypothesis, not a guarantee of business correctness; disconnected/noise selections are explicit.

8. **Why isn't a SELECT prefix enough for safety?** SQL can contain multiple statements, nested writes, virtual tables or dangerous functions. AST validation plus catalog qualification narrows syntax; mode=ro, query_only, authorizer, disabled extensions and bounded execution provide independent controls. This is still not a process sandbox for hostile multi-tenant workloads.

9. **How do aliases, CTEs and stars affect validation?** Aliases identify sources within scopes; CTE names are derived relations, not physical tables. SQLGlot qualification resolves those scopes and expands stars before reference extraction. Unknown/unsupported extraction is marked, not converted into invented labels.

10. **What exactly is execution accuracy?** Whether predicted and gold outputs match on this database, with column positions, shape, duplicate multiplicity, numeric tolerance, NULLs and meaningful ordering preserved. Two wrong queries can return identical or empty results; counterfactual databases and human checks would strengthen validation.

11. **Why not compare SQL strings alone?** Different joins/subqueries can be equivalent. Normalized exact match is useful as a secondary conservative signal but does not establish semantic correctness. Executing successfully also does not establish correctness.

12. **How did you prevent evaluation leakage?** The retriever reads only examples.json. Development questions tune prompts; evaluation gold is read only by scoring and error analysis. Tests check path/content exclusion. Lexical overlap detection and a pending human checklist address near-paraphrases, but neither automated checks nor an agent review proves zero semantic overlap.

13. **What did semantic checks catch?** Independent Python oracles exposed floating-point edge cases in within-genre price comparisons and tied monetary rankings. The gold queries now compare cents or rounded monetary aggregates. This is an actual observed development finding, not a model-accuracy result.

14. **How is self-correction bounded?** At most two corrections after the initial generation. Each receives previous SQL, a bounded sanitized error and permitted context. Every attempt re-enters shared validation. API calls for both pipelines, transport attempts and remote embeddings share the budget. Auth/quota errors are not repaired as SQL.

15. **What is the LangChain tradeoff?** LC components provide standard retriever/model/structured-output interfaces and LCEL composition. They add dependencies and version surface. The project shares coordination and safety to keep comparisons fair. Adapter line counts are not a valid standalone total-code comparison, and fewer lines do not imply better engineering.

16. **How are caches handled fairly?** Keys include the exact prompt, implementation/model/settings, schema/database hashes and encoding. Cross-pipeline generation caches are isolated. Hit labels retain original usage provenance; fresh latency excludes cached or mixed runs. Changed configurations can cost calls; reruns are not guaranteed free.

17. **What do confidence intervals mean with 60 questions?** Seeded bootstrap intervals summarize sampling variability under the observed set; paired resampling uses matching IDs. They do not remove hand-authored selection bias, label errors, correlation, repeated tuning or multiple-comparison effects. Distinct means alone are not evidence of a meaningful improvement.

18. **What is the synthetic stress test measuring?** Robustness to irrelevant schema documents at 0/25/50/100 fixed-seed distractors. It does not simulate real database growth, query planner behavior or new meaningful joins. Distractor selection is an error, and nonexistent table execution is rejected.

19. **How would you scale?** First expand evaluation diversity, then measure retrieval bottlenecks. Candidates include hierarchical/schema-aware retrieval, approximate indexes, persistent embedding services, reranking budgets, schema versioning and calibrated abstention. Spider/BIRD and real multi-schema workloads are future work, not demonstrated results.

20. **Can you deploy it on a free host?** It is packaged for Streamlit with CPU dependencies, committed public indexes, lazy model loading and reranking off by default. Local Windows UI-plus-embedding RSS was approximately 611 MiB. No Community Cloud deployment or fixed memory guarantee is claimed. Query embeddings still require a runtime model; BM25 can reduce memory use.

21. **How are summaries grounded?** A shared typed deterministic summary reports actual result counts, columns and a first-row preview, with explicit preview/result truncation. It avoids an extra LLM call. It does not claim natural-language explanatory insights beyond the returned data.

22. **What would you show an interviewer today?** The real schema/index manifests, gold verification artifact, safety tests, mocked pipeline parity tests, AppTest and local resource probe. Live execution accuracy, reranker gains, dollar cost and cloud deployment remain unmeasured until credentials/budgets are supplied.

## Evidence discipline

Implemented capabilities: guarded SQL execution, both retrieval/orchestration paths, deterministic summary, leakage guards, controlled experiments and UI.

Observed: all 93 gold queries execute and pass independent Python semantic oracles; the committed verification report records the actual offline test count and local startup checks. Consult those generated artifacts for exact outcomes.

Pending: human gold-label review, paid/live generation smoke, full ablations, external benchmark generalization and hosted deployment. Never turn a mocked success into a live API or accuracy claim.
