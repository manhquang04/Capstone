# Search log

Search date: 2026-09-30. Scope: 2020--2026, works that empirically re-evaluate, benchmark, audit, or systematize federated-learning gradient-inversion defenses. Search ranking was topical fit, not citation count. A search hit was not used as evidence until its paper text was read.

| Time/order | Service | Exact query / URL | Result and action |
|---|---|---|---|
| 1 | OpenAlex + arXiv (`literature`) | `federated learning gradient inversion defenses benchmark re-evaluation`, 2020--2026 | 25 candidates returned. Retained Huang, Balunovic, Yue, Du, Li, Wei, and TabLeak for full-text screening. |
| 2 | OpenAlex + arXiv (`literature`) | `"SoK: On Gradient Leakage in Federated Learning"`, 2020--2026 | Found Du et al., arXiv:2404.05403. Read full text. |
| 3 | OpenAlex + arXiv (`literature`) | `"breaching" benchmark federated learning gradient inversion`, 2020--2026 | No directly citable archival Breaching paper found. Checked project README separately. |
| 4 | Semantic-Scholar-targeted research search | `site:api.semanticscholar.org federated learning gradient leakage benchmark defense re-evaluation` | Free fallback returned `search_unavailable`; no inference made from the zero hits. |
| 5 | OpenAlex-targeted research search | `OpenAlex federated learning gradient inversion defense benchmark re-evaluation` | Returned Huang et al. and related items; read Huang full text. |
| 6 | Semantic-Scholar-targeted research search, retry | `"LeakProf" gradient leakage` | Free fallback circuit was open, `search_unavailable`; no usable content. The request was not rate-limited (no HTTP 429), so retries beyond the second unavailable attempt would be uninformative. |
| 7 | Semantic-Scholar-targeted developer search, retry | `"breaching" "gradient inversion" Geiping GitHub` | Free fallback circuit open, `search_unavailable`; used authoritative GitHub README through WebFetch instead. |
| 8 | Research search, retry | `"gradient leakage" "benchmark" federated learning defenses` | Free fallback circuit open, `search_unavailable`. |
| 9 | OpenAlex + arXiv (`literature`) | `"privacy defenses" "generative gradient leakage" auditing federated learning`, 2020--2026 | Found Li et al., arXiv:2203.15696 / CVPR 2022. Read full text. |
| 10 | OpenAlex + arXiv (`literature`) | `"Evaluating Gradient Inversion Attacks and Defenses" defenses Soteria PRECODE`, 2020--2026 | Confirmed related defenses and located follow-up candidates; no additional multi-defense audit was added without full-text evidence. |
| 11 | OpenAlex + arXiv (`literature`) | `gradient leakage benchmark LeakProf federated learning`, 2020--2026 | 16 candidates. No identifiable work called “LeakProf” was returned. This is an unresolved name, not evidence that it does not exist. |
| 12 | WebFetch | `https://raw.githubusercontent.com/JonasGeiping/breaching/main/README.md` | Read README: modular attack/use-case framework; explicit threat-model information; no defense suite except user-level DP/aggregation. |
| 13 | Semantic Scholar Graph API | `https://api.semanticscholar.org/graph/v1/paper/ARXIV:2404.05403?fields=title,venue,year,externalIds` | HTTP 429 rate limit. No immediate retry was made, per host pacing guidance. |

## Read corpus

- Balunovic et al., *Bayesian Framework for Gradient Leakage*, arXiv:2111.04706 / ICLR 2022: pp. 1--2, 3, 6--9 and targeted passages for DP/BN/knowledge.
- Yue et al., *Gradient Obfuscation Gives a False Sense of Security in Federated Learning*, arXiv:2206.04055 / USENIX Security 2023: pp. 1--2, 7--10 and targeted passages for Soteria/PRECODE/DP.
- Huang et al., *Evaluating Gradient Inversion Attacks and Defenses in Federated Learning*, arXiv:2112.00059 / NeurIPS 2021: pp. 1--3 and targeted passages for DP/BatchNorm.
- Du et al., *SoK: On Gradient Leakage in Federated Learning*, arXiv:2404.05403: pp. 1--2, 5, 20 and targeted DP/tabular passages. (The full-text record is arXiv v2, dated 2025; venue metadata was not verified from Semantic Scholar because of 429.)
- Li et al., *Auditing Privacy Defenses in Federated Learning via Generative Gradient Leakage*, arXiv:2203.15696 / CVPR 2022: pp. 1--2 and targeted defense/DP passages.
- Wei et al., *A Framework for Evaluating Gradient Leakage Attacks in Federated Learning*, arXiv:2004.10397: pp. 1--2.
- Vero et al., *TabLeak*, arXiv:2210.01785: pp. 6, 9, 13 and targeted defense passages; included because it bears directly on the planned tabular native setting, not because it is a cross-defense benchmark.
- Geiping et al., Breaching README, fetched 2026-09-30; framework documentation rather than an archival paper.

Excluded after screening: broad FL/security surveys, single-defense proposals, and attack-only papers without an empirical multiple-defense re-evaluation. The 2026 arXiv search results were excluded if their arXiv timestamp postdated this search date or if they did not match the multi-defense/audit question.
