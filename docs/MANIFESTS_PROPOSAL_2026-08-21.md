# Propuesta de manifiestos — archivo histórico, parcialmente ejecutado

> **ESTADO AL 2026-08-30.** Se materializaron el lockfile versionado
> (`requirements-lock.txt`) y el CI (`.github/workflows/ci.yml`). El resto de esta
> propuesta conserva su formulación original y no debe leerse como una lista vigente de
> cambios autorizados; `requirements.txt` sigue siendo la fuente editable.

> **Sección de Claude Code — 2026-08-21 07:50 (hora local).**
> Ítem 4 del prompt P-DOC (`docs/PLAYBOOK_GATES_2026-08-06.md`), que pide **proponer antes de
> ejecutar**. Este documento no cambia `requirements.txt` ni `setup.py`, y **no genera lockfile**.
> Nada de lo de abajo se aplica sin OK explícito de Enzo.

## El problema, medido

`requirements.txt` y `setup.py` describen dos conjuntos distintos, y **ninguno de los dos instala
lo que el repo importa**. Imports de terceros encontrados en `src/`, `scripts/` y `tests/`:

```
bs4  datasets  datasketch  dotenv  faiss  huggingface_hub  matplotlib  numpy  ollama
openpyxl  pandas  plotly  pydantic  pytest  rank_bm25  requests  rich  safetensors
scipy  sentence_transformers  statsmodels  streamlit  tiktoken  torch  tqdm  transformers  yaml
```

Contra eso:

| Paquete importado | ¿en `requirements.txt`? | ¿en `setup.py`? |
|---|---|---|
| `streamlit`, `plotly` | **no** | **no** |
| `pytest` | **no** | **no** |
| `scipy`, `openpyxl`, `huggingface_hub`, `safetensors`, `ollama`, `datasets` | **no** (llegan como dependencias transitivas) | **no** |
| `faiss`, `rank_bm25`, `torch`, `statsmodels`, `scikit-learn`, `nltk`, `sentence-transformers` | sí | **no** (salvo `sentence-transformers` en el extra `ml`) |
| `matplotlib`, `datasketch`, `jupyterlab` | sí (obligatorias) | sí, pero como extras `dev`/`ml` |

Consecuencia práctica: el README manda `pip install -r requirements.txt` y luego
`streamlit run src/ui/app.py`, y ese comando **falla en un entorno limpio**. Y `pip install -e .`
deja fuera todo el stack de retrieval.

Aparte, hay tres imports que **no deben entrar en ningún manifiesto**: `anthropic`, `openai`,
`ragas` y `bert_score` aparecen en scripts auxiliares o comentados y no participan de ninguna cifra
citable; `InstructorEmbedding` ya está comentado como opcional en `requirements.txt`.

## Propuesta: `setup.py` canónico, `requirements.txt` como receta de entorno

**Canónico = `setup.py`.** Es el que declara `python_requires`, el que `pip install -e .` respeta y
el único que puede expresar extras. `requirements.txt` pasa a ser una receta plana equivalente a
`.[all]`, para quien prefiera el flujo del README sin editar nada.

```
install_requires   núcleo que necesita CUALQUIER flujo, incluido el pipeline:
                   pydantic pyyaml python-dotenv tiktoken requests beautifulsoup4 lxml
                   rich tqdm pandas numpy scipy

extras_require:
  retrieval        faiss-cpu rank-bm25 scikit-learn nltk sentence-transformers torch
                   transformers safetensors huggingface_hub markdownify gitpython datasketch
  eval             statsmodels openpyxl        (estadística + export de tablas)
  ui               streamlit plotly            (la demo del README)
  test             pytest
  dev              jupyterlab matplotlib seaborn
  all              retrieval + eval + ui + test
```

Con eso:
- `pip install -e .` da el núcleo, que es lo que un consumidor del paquete necesita.
- `pip install -e ".[all]"` reproduce el entorno completo, y el README pasa a recomendarlo.
- `requirements.txt` se regenera como el listado plano de `.[all]` y deja de divergir por omisión.

**Nota de alcance:** `python_requires=">=3.11"` **no cambia** — es soporte del paquete, distinto del
entorno reproducible 3.14 de `REPRODUCE.md §0`. Esa separación es decisión ya tomada (2026-08-04).

## Lockfile — recomendación, y por qué NO lo genero ahora

Un lockfile con hashes es lo correcto para una tesis, pero **congelaría el entorno actual**, y el
entorno actual acaba de demostrar que no es estable: el 2026-08-21 el mismo prompt, el mismo modelo
y la misma semilla produjeron respuestas distintas a las de exp18 (`tokens_in` idéntico,
`tokens_out` distinto — ledger entrada 25). Un lockfile generado hoy grabaría ese estado como si
fuera el que produjo la evidencia firmada, que es **falso**.

Recomendación: generar el lockfile **después** de cerrar exp19b y de decidir si se congela o se
actualiza el runtime de Ollama, y acompañarlo de la versión de Ollama (`0.22.1` hoy) y del digest
del modelo, que es lo que `pip freeze` no captura y es justo lo que se movió. Si Enzo lo autoriza
igualmente antes, se hace, pero con esa advertencia escrita al lado.

## Qué hace falta de Enzo

1. ¿`setup.py` canónico con extras, y `requirements.txt` como equivalente plano de `.[all]`?
2. ¿Se ajusta el README para recomendar `pip install -e ".[all]"`?
3. ¿Lockfile ahora o después de exp19b? (Recomiendo después, por lo de arriba.)
