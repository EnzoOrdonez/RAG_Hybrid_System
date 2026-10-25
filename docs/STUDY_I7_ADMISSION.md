# Admisión de publicación y contextos

Estándar: sección Prerrequisitos de la enmienda de servicio I4. El anexo
efectivo debe estar commiteado y publicado; observadores, enmienda y calendario
deben anclarse antes de medir. El congelamiento incluye doce contextos vivos
verificados sin generar contenido. La versión de I7 adopta la alternativa de
commit publicado, sin crear objetos GCS para acreditar publicación.

## Código

El sondeo aislado i7-contexts recorre la consulta congelada y se detiene en
build_context antes de llamar al LLM. Comprueba servicio frío antes y después
e inventario constante. Los doce conjuntos ordenados deben coincidir exactamente
con el manifiesto I5 que conserva las referencias a I3, anclado por SHA-256.
El sondeo solo funciona con propósito technical y contenedor sin red.

La admisión exige que el commit del anexo sea el tip publicado de la rama,
que sus archivos coincidan por SHA-256 y que src/scripts/config no difieran
del commit de la imagen. Exige coincidencia de software con el inventario del
sondeo. Una publicación ausente, un hash distinto, un contexto alterado o una
referencia incompleta bloquean el inicio. Las pruebas de publicación local son
dobles; no acreditan que un anexo real ya se haya publicado.

Después del paso contexts de B:

```powershell
python -m scripts.study_operator.i7_admission --root $run --draft
```

Tras revisar, commitear y publicar el anexo con permiso:

```powershell
python -m scripts.study_operator.i7_admission --root $run --anchor-published
```

Revalidar antes de medir con el mismo módulo y solo --root. Si el remoto
avanzó, revisar y generar un anclaje nuevo; no sobrescribir el anterior.

## Decisión y evidencia de Enzo

Enzo debe revisar el anexo efectivo de la imagen construida y autorizar su
publicación en la rama. El código comprueba el commit remoto y sus bytes;
no inventa aprobación ni publicación. No se requiere un sitio externo
adicional si se cumple esta alternativa del preregistro. El inventario privado
con identificadores permanece fuera del repositorio público; el borrador
publica únicamente identidades por hash y contexto técnico mínimo.
