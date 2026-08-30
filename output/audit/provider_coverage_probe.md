# Probe descriptivo de cobertura por proveedor

Este probe es descriptivo y queda fuera de toda familia de contraste/BH. No reejecuta retrieval: usa las selecciones congeladas de exp17.

Queries multi-nube detectadas: **25**.

| Brazo persistido | Cobertura estricta | Fracción | Chunks fuera de los proveedores de la query | Tasa |
|---|---:|---:|---:|---:|
| baseline | 2/25 | 0.0800 | 0/125 | 0.0000 |
| balanced | 20/25 | 0.8000 | 0/125 | 0.0000 |

Cobertura estricta significa que el top-k contiene al menos un chunk de cada proveedor mencionado. La tasa de desajuste cuenta chunks cuyo `cloud_provider` persistido no pertenece al conjunto de proveedores de la query.

Fuente: `experiments/results/exp17_crosscloud_balanced/retrieval_ids.json`; queries: `data/evaluation/test_queries.json`.
