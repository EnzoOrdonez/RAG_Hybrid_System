# Revisión literal de tareas, anterior a las respuestas

Autorización: §0.7 del encargo de 2026-10-02. Se examinó el texto de los
24 481 fragmentos del índice, mediante cadenas literales y lectura de candidatos.
No se ejecutó recuperación, generación ni clasificación de respuestas para elegir.
Corpus SHA-256: `a48865398c2ef9eac1b43b95ab1752c4db206d2577f6ee197ffc22bc341eb58c`.

La búsqueda cruzó servicios dentro de cada proveedor: la etiqueta `ECS` no impide
que un fragmento documente EKS, VPC o CloudFormation. Se descartaron menciones,
índices de enlaces y ejemplos que no contienen la información solicitada.
Las ocho consultas con premisa falsa permanecen excluidas.

| Tarea | Fragmentos oficiales revisados | Información directa |
|---|---|---|
| q001, conservada | `33a48262-d674-4750-92cf-a70c0a41ee56` | Cuotas de lanzamiento de pods EKS en Fargate: ráfaga, reposición y excepción de cuentas solo EKS. Alcance Fargate explícito; no demuestra todas las cuotas EKS. |
| q016, sustituye q010 | `055ccb10-9a52-44ad-875b-b5cea5d54ab3` | Límites de 1 000 tareas por servicio con discovery/Service Connect; cuotas iniciales frente a aplicadas. La tabla externa ausente no cuenta. |
| q068, sustituye q064 | `bbebad97-021b-47d4-acc7-30a40b208138` | Pasos VPC con subredes públicas/privadas, IP y distribución entre zonas. |
| q070, conservada | `711c7690-e491-4a92-9dcd-598b0d273428`, `199b5e13-83fd-4ca4-8474-fa9c78fdf208` | Crear servicio desde definición de tarea, cantidad deseada, red, rollback, balanceador y distribución por zona. |
| q180, sustituye q171 | `3acc0829-f11b-40f5-b07c-88530ac2d34a`, `112f91e2-606d-4ea6-8d91-9dc18ae66842` | CloudFormation: formatos JSON/YAML y stacks AWS; ARM: esquema JSON, tipos Microsoft, apiVersion y parámetros, con ejemplo Bicep. No se presume una comparación exhaustiva. |
| q172, conservada | `543f812f-6e8e-4222-a6de-212ac9216627`, `e5f16001-7611-4273-a747-1ee68da466b5`, `66eeca24-f620-42b3-9cbb-a72cc7fbf703` | Descripciones oficiales de VPC y Virtual Network: recursos, aislamiento, direccionamiento, subredes y seguridad. No atribuir a Azure propiedades zonales que estos textos no explican. |

q010 no obtuvo un fragmento directo de cuotas S3: los candidatos trataban cuotas
EC2 o límites de archivos de entorno ECS. q064 obtuvo enlaces a EKS, no pasos de
configuración productiva. q171 obtuvo ejemplos Lambda, sin descripción directa
de su oferta de cómputo serverless. Es una conclusión de la búsqueda documentada,
no una prueba matemática de ausencia de cualquier paráfrasis posible.

Los pares conservan dificultad media, tipo y proveedores. Las plantillas son
«limits and quotas», «set up … for a production workload» y «differences between
… AWS … Azure …». Para la última se normaliza únicamente «main» y la puntuación;
no se reduce el requisito a compartir la etiqueta comparative. Se descartó q178
porque las menciones Azure a VMSS no explicaban suficientemente su autoescalado.

Sello nuevo (se conserva el original y una copia verificada):
`d3e79ce98dfa21c5ba98a3e6ac27b91748b4251a11f44e6930bb9c8c1c80dfe5`.
T1: q001/q068/q180. T2: q016/q070/q172. CSV de asignaciones y etiquetas sin cambios.
SHA-256 de evidencia literal, enlazado por la configuración sellada:
`f184239a484bcce15ed4cca39dee14b598029c276fabb3372fa5ff672e98c9d4`.

Evidencia completa: paquete externo `autonomous-run-20261002T203006Z`, archivos
`corpus-*.json`, `corpus-*.log`, `study-config-reviewed/task_evidence.json` y
`task-reseal.log`. La cobertura mínima no demuestra intercambiabilidad cognitiva,
actualidad de las instrucciones del corpus ni una respuesta exhaustiva.

UX: cambian tres preguntas visibles; aparecen en su ranura habitual, con texto
literal del catálogo de 194 consultas, sin indicar sustitución ni condición.
No se modifica ningún instrumento. La selección precede a toda respuesta nueva.
