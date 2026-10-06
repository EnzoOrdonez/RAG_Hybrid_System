# Anexo de entorno: recuperación b4 y ensayo de infraestructura

Generado desde el inventario vivo del primer arranque, excluido de la aceptación y disponibilidad. La VM se detuvo antes de commitear este anexo.

```json
{
  "target": {
    "id": "3701269511524979546",
    "name": "cloudrag-i4-alternate-b4-final-20261005",
    "zone": "us-central1-b"
  },
  "project": "pure-loop-474323-a8",
  "region": "us-central1",
  "image": {
    "kind": "docker",
    "container_image_id": "sha256:b1ae8bb06c4b700afb5aa3cc76979d4fdf9498fe80218eb054cc0ae8efd23ba3",
    "image_id": "sha256:b1ae8bb06c4b700afb5aa3cc76979d4fdf9498fe80218eb054cc0ae8efd23ba3"
  },
  "source": {
    "commit": "93219c0e12c23cc536651e20e8e2df7247488499",
    "tree": "fd7c4f3978b310c3066964669bdf3dfa94281026"
  },
  "inventory_path": "C:\\CloudRAG\\autonomous-run-20261004T230147Z\\runtime-user25-bootstrap-inventory.json",
  "inventory_sha256": "6381dd8756d6baf34f1b6b3ae2ed05cda4f9ea15363790f5c375f54debebe9d9",
  "observed_host_sha256": {
    "scripts/study_operator/deployment.py": "4fa4707d814a9dde7484af13465e1a28bfb1a380364b30afa9438784b462964e",
    "scripts/study_operator/host_runtime.py": "9d0a17dc446d096c3c5fdde2ec119cf0b7074de8e72e3710b0bde2310414423e"
  },
  "pending_host_user_candidate": "4f43cfc1c386a24fa5e85088de73a9d0ff8ed375",
  "paired_measurement_started": false
}
```

Dos solicitudes anteriores en us-central1-c fueron rechazadas por capacidad antes del ensayo pareado. Esta recuperación existente en us-central1-b conserva la imagen final y el certificado de la IP regional. No es el simulacro de conmutación ni concede GO. El próximo ensayo compara solo USER ausente frente a USER=cloudrag, con UID/GID 10001 y sin modelos o generación; el cambio solo se instala si pasa. La prueba de los doce contextos, el congelamiento RAG y el anexo final siguen pendientes antes del estímulo, smoke y compuerta. No se combinan datos con otro entorno.
