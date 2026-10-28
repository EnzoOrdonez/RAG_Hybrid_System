# ADR-0019: Correcciones del ledger y la admisión humana I7

[DECLARADO] Estado: aceptado para preparación local; no autoriza publicación,
gasto ni ejecución de Fase B. Base: d8954cb más los cherry-picks ab267dd y d0ba95a
(HEAD de integración 5ae784f). Tarea F-020, revisión cruzada del 2026-10-27.

## Entorno y línea base

[VERIFICADO] El worktree i7-human-ledger-ready usa una junction .venv-app al
mismo destino que f020-docs: el entorno existente de interview-readiness.
No se instalaron ni copiaron paquetes. El enlace está ignorado por Git.

[VERIFICADO] Git Bash seleccionó .venv-app/Scripts/python.exe sin PYTHON
sobrescrito. Con HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, PYTHONHASHSEED=42 y
PYTHONUTF8=1, bash init.sh dio secretos=0, CPU=0 y docs=0: 1896 pasan,
0 fallan, 11 omitidas, 5 excluidas y 9 subtests pasan (211,22 s).

## F1: aislar el ledger heredado

[DECLARADO] reserve_cost e ip_reserve rechazan iteration=7 antes de consultar
nube o tocar el ledger de I5. La comprobación vive en lifecycle, por lo que
también protege una instancia de la clase heredada con instalación I7.
I7 debe usar su admisión/reserva propia; no se eleva ningún tope.

[VERIFICADO] El test parametrizado de fase B/humana y clase I7/heredada falló
en sus ocho casos antes del cambio. Comando de reproducción offline:

```text
.venv-app/Scripts/python.exe -m pytest tests/test_study_i7_ledger.py -o addopts= -q -p no:cacheprovider
```

[DECLARADO] Las demás correcciones se registrarán al verificarse, antes del
cierre local. El despliegue, GO real y aprobación ética real siguen pendientes.

[VERIFICADO] F1 verde: 37 tests de ledger y ciclo/IP heredados pasan. Al retirar
por separado la guarda de reserve_cost y la de ip_reserve, fallan cuatro casos
por mutante; los bytes originales se restauraron en ambos casos.
