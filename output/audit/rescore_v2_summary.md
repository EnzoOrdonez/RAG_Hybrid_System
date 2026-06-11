# Acuerdo entre verificadores NLI — exp12_matrix (F3b, N5)


## Arm `base` — verificador C:\Users\enziz\Projects\hybrid-rag-system\data\models\nli-deberta-v3-base, strip_header=False
claim-count mismatches vs persistidos: 0

| config | pub v1 | pub base | prim v1 | prim base | contr% v1 | contr% base |
|---|---|---|---|---|---|---|
| lexico | granite4.1-8b | 0.170 | 0.123 | 0.234 | 0.183 | 34.2 | 25.9 |
| denso | granite4.1-8b | 0.193 | 0.151 | 0.255 | 0.206 | 33.6 | 19.1 |
| hibrido | granite4.1-8b | 0.202 | 0.133 | 0.293 | 0.205 | 32.5 | 27.3 |
| lexico | gemma4-e4b | 0.331 | 0.305 | 0.381 | 0.367 | 26.8 | 22.4 |
| denso | gemma4-e4b | 0.322 | 0.255 | 0.356 | 0.232 | 29.2 | 21.4 |
| hibrido | gemma4-e4b | 0.268 | 0.197 | 0.312 | 0.237 | 34.5 | 31.5 |
| lexico | mistral-7b-instruct | 0.222 | 0.178 | 0.245 | 0.178 | 31.7 | 23.8 |
| denso | mistral-7b-instruct | 0.258 | 0.189 | 0.275 | 0.197 | 28.8 | 17.0 |
| hibrido | mistral-7b-instruct | 0.256 | 0.157 | 0.282 | 0.167 | 29.2 | 26.0 |
| lexico | qwen3.5-9b | 0.306 | 0.223 | 0.246 | 0.192 | 37.0 | 33.5 |
| denso | qwen3.5-9b | 0.254 | 0.261 | 0.246 | 0.242 | 36.3 | 20.8 |
| hibrido | qwen3.5-9b | 0.278 | 0.192 | 0.293 | 0.200 | 39.0 | 31.7 |

Spearman 12 configs (publicada): rho=0.825 p=0.0010
Spearman 12 configs (primaria v2): rho=0.559 p=0.0586

Orden de escenarios por modelo (primaria):
  granite4.1-8b          v1: hibrido>denso>lexico     base: denso>hibrido>lexico
  gemma4-e4b             v1: lexico>denso>hibrido     base: lexico>hibrido>denso
  mistral-7b-instruct    v1: hibrido>denso>lexico     base: denso>lexico>hibrido
  qwen3.5-9b             v1: hibrido>lexico>denso     base: denso>hibrido>lexico

Muestra de 50 claims: matched=49  kappa(v1 vs base)=0.411
confusión (v1 -> arm): contradicted->contradicted: 12, contradicted->supported: 2, contradicted->unsupported: 6, supported->contradicted: 3, supported->supported: 3, supported->unsupported: 3, unsupported->contradicted: 2, unsupported->supported: 2, unsupported->unsupported: 16

q085 bajo base: {'total_claims': 28, 'supported': 0, 'contradicted': 28, 'unsupported': 0, 'faithfulness': 0.0}  (v1: 28/28 contradicted, faith 0.0)

## Arm `small_noheader` — verificador cross-encoder/nli-deberta-v3-small, strip_header=True
claim-count mismatches vs persistidos: 0

| config | pub v1 | pub small_noheader | prim v1 | prim small_noheader | contr% v1 | contr% small_noheader |
|---|---|---|---|---|---|---|
| lexico | granite4.1-8b | 0.170 | 0.161 | 0.234 | 0.233 | 34.2 | 36.9 |
| denso | granite4.1-8b | 0.193 | 0.194 | 0.255 | 0.264 | 33.6 | 34.0 |
| hibrido | granite4.1-8b | 0.202 | 0.208 | 0.293 | 0.302 | 32.5 | 34.5 |
| lexico | gemma4-e4b | 0.331 | 0.334 | 0.381 | 0.393 | 26.8 | 29.7 |
| denso | gemma4-e4b | 0.322 | 0.332 | 0.356 | 0.372 | 29.2 | 29.2 |
| hibrido | gemma4-e4b | 0.268 | 0.283 | 0.312 | 0.315 | 34.5 | 33.1 |
| lexico | mistral-7b-instruct | 0.222 | 0.211 | 0.245 | 0.233 | 31.7 | 31.6 |
| denso | mistral-7b-instruct | 0.258 | 0.242 | 0.275 | 0.254 | 28.8 | 30.5 |
| hibrido | mistral-7b-instruct | 0.256 | 0.254 | 0.282 | 0.284 | 29.2 | 30.4 |
| lexico | qwen3.5-9b | 0.306 | 0.315 | 0.246 | 0.275 | 37.0 | 36.9 |
| denso | qwen3.5-9b | 0.254 | 0.259 | 0.246 | 0.262 | 36.3 | 37.0 |
| hibrido | qwen3.5-9b | 0.278 | 0.277 | 0.293 | 0.304 | 39.0 | 39.8 |

Spearman 12 configs (publicada): rho=0.965 p=0.0000
Spearman 12 configs (primaria v2): rho=0.944 p=0.0000

Orden de escenarios por modelo (primaria):
  granite4.1-8b          v1: hibrido>denso>lexico     small_noheader: hibrido>denso>lexico
  gemma4-e4b             v1: lexico>denso>hibrido     small_noheader: lexico>denso>hibrido
  mistral-7b-instruct    v1: hibrido>denso>lexico     small_noheader: hibrido>denso>lexico
  qwen3.5-9b             v1: hibrido>lexico>denso     small_noheader: hibrido>lexico>denso

Muestra de 50 claims: matched=49  kappa(v1 vs small_noheader)=0.870
confusión (v1 -> arm): contradicted->contradicted: 18, contradicted->unsupported: 2, supported->contradicted: 1, supported->supported: 8, unsupported->contradicted: 1, unsupported->unsupported: 19

q085 bajo small_noheader: {'total_claims': 28, 'supported': 0, 'contradicted': 28, 'unsupported': 0, 'faithfulness': 0.0}  (v1: 28/28 contradicted, faith 0.0)