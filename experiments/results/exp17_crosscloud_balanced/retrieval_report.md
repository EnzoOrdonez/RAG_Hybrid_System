# exp17 — provider-balanced retrieval (25 cross-cloud q)

Self-validation: baseline top-5 vs exp13 exp_off mean overlap@5 = **5.0/5** (5.0 = perfect replication).

**Provider coverage (all wanted providers present in top-5):** baseline 7/25 -> balanced **25/25**.
**Oracle NDCG@5 (bge-reranker, within-pool):** baseline None -> balanced None (trade-off).

| qid | wanted | base providers | bal providers | base_cov | bal_cov | ndcg_b | ndcg_bal | ovlp13 |
|---|---|---|---|---|---|---|---|---|
| q171 | aws+azure | azure,azure,azure,azure,azure | azure,azure,azure,aws,aws | False | True | None | None | 5 |
| q172 | aws+azure | azure,azure,azure,azure,aws | azure,azure,azure,aws,aws | True | True | None | None | 5 |
| q173 | aws+azure+gcp | azure,azure,azure,gcp,azure | azure,azure,gcp,gcp,azure | True | True | None | None | 5 |
| q174 | aws+azure+gcp | azure,azure,azure,azure,azure | azure,azure,aws,aws,gcp | False | True | None | None | 5 |
| q175 | aws+azure+gcp | gcp,gcp,aws,aws,aws | gcp,gcp,aws,aws,azure | False | True | None | None | 5 |
| q176 | aws+azure+gcp | aws,gcp,aws,aws,gcp | aws,gcp,aws,gcp,azure | False | True | None | None | 5 |
| q177 | aws+azure | azure,azure,azure,azure,azure | azure,azure,azure,aws,aws | False | True | None | None | 5 |
| q178 | aws+azure | aws,azure,aws,aws,azure | aws,azure,aws,aws,azure | True | True | None | None | 5 |
| q179 | aws+azure+gcp | gcp,azure,gcp,gcp,gcp | gcp,azure,gcp,azure,aws | False | True | None | None | 5 |
| q180 | aws+azure | aws,aws,aws,aws,aws | aws,aws,aws,azure,azure | False | True | None | None | 5 |
| q181 | aws+azure+gcp | aws,aws,aws,aws,aws | aws,aws,azure,azure,gcp | False | True | None | None | 5 |
| q182 | aws+azure+gcp | aws,aws,aws,gcp,aws | aws,aws,gcp,gcp,aws | True | True | None | None | 5 |
| q183 | aws+azure | azure,azure,azure,azure,azure | azure,azure,azure,azure,azure | True | True | None | None | 5 |
| q184 | aws+azure+gcp | gcp,gcp,gcp,aws,gcp | gcp,gcp,aws,aws,gcp | True | True | None | None | 5 |
| q185 | aws+azure | azure,azure,azure,azure,azure | azure,azure,azure,azure,azure | True | True | None | None | 5 |
| q187 | aws+azure+gcp | gcp,gcp,aws,gcp,gcp | gcp,gcp,aws,azure,azure | False | True | None | None | 5 |
| q189 | aws+azure+gcp | azure,gcp,gcp,azure,azure | azure,gcp,gcp,azure,aws | False | True | None | None | 5 |
| q190 | aws+azure+gcp | aws,aws,aws,azure,aws | aws,aws,azure,azure,gcp | False | True | None | None | 5 |
| q191 | aws+azure | azure,azure,azure,azure,azure | azure,azure,azure,aws,aws | False | True | None | None | 5 |
| q192 | aws+azure+gcp | gcp,gcp,gcp,gcp,gcp | gcp,gcp,azure,azure,aws | False | True | None | None | 5 |
| q193 | aws+azure+gcp | azure,aws,aws,aws,aws | azure,aws,aws,azure,gcp | False | True | None | None | 5 |
| q195 | aws+azure+gcp | gcp,azure,gcp,azure,gcp | gcp,azure,gcp,azure,aws | False | True | None | None | 5 |
| q196 | aws+azure+gcp | azure,azure,azure,azure,azure | azure,azure,aws,gcp,aws | False | True | None | None | 5 |
| q197 | aws+azure+gcp | azure,aws,aws,azure,aws | azure,aws,aws,azure,gcp | False | True | None | None | 5 |
| q198 | aws+azure+gcp | aws,aws,aws,azure,azure | aws,aws,azure,azure,gcp | False | True | None | None | 5 |