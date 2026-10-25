"""Fixed I5 report from current hash-bound claims; no acceptance synthesis."""
import argparse
import json
from pathlib import Path

from scripts.study_operator.evidence import verify
from scripts.study_operator.run_control import require_limited

SECTIONS = ('Resumen ejecutivo', 'Cierre de la iteración 4', 'Auditoría', 'Defectos heredados',
    'Retención y costos', 'Congelamiento RAG', 'UEQ-S y plan de análisis', 'Estímulo',
    'Privacidad y borrado', 'Capacidad y entorno', 'TLS y continuidad', 'Smoke y compuerta',
    'Atributos de calidad', 'Plazos y tiempo', 'Relevos', 'Decisiones autónomas',
    'Matriz de riesgos residuales', 'Checklist humano', 'Cambios documentales pendientes',
    'Veredicto', 'Bloqueos', 'Falta')
CRITERIA = {
    'Disponibilidad': 'READY ≤15 min en ≥3 arranques consecutivos, sin emisión nueva de certificado.',
    'Continuidad': 'Conmutación real, identidad verificada y retorno; regla de entornos preregistrada.',
    'Confiabilidad': 'Compuerta sin fallos/inválidas y respaldo por generación/SHA-256 en cada sesión sintética.',
    'Recuperabilidad': 'Restauración real de una sesión sintética y exportación idéntica.',
    'Rendimiento': 'p95 lineal ≤60 s por condición exclusivamente en el agregado de120 intentos.',
    'Seguridad': 'Solo443 público, XSRF funcional, SA mínima, metadatos aislados y secretos limpios.',
    'Privacidad': 'Prueba nueva de canarios cliente y cuatro comprobaciones de borrado de§73.',
    'Operabilidad': 'Runbook literal completo y errores con acción siguiente.',
    'Costo': 'Estimado/márgenes separados, retención ≤USD0,45/día y proyección30d bajo90.',
    'UEQ-S': 'Ítems oficiales/polaridad/7puntos; captura, puntuación, exportación y retiro probados.',
    'Análisis': 'Regla pareada preregistrada, bootstrap/d_z/BH2UEQ y tests sintéticos normales/no normales.',
}
STATUSES = {'OBSERVADO', 'FALLÓ', 'NO_MEDIDO', 'BLOQUEADO-HUMANO', 'VERIFICACIÓN_FUNCIONAL'}


def render(root, plan):
    root = Path(root)
    verified = verify(root)
    excluded = set(verified['retracted_claims'])
    claims = {row['id']: row for row in json.loads((root/'claims.json').read_bytes()) if row['id'] not in excluded}
    if set(plan['sections']) != set(SECTIONS) or set(plan['attributes']) != set(CRITERIA):
        raise ValueError('All22 sections and11 attributes required, without substitutes')
    if len(plan['sections']['Resumen ejecutivo']) != 7:
        raise ValueError('Executive summary requires seven claim references')
    def assertion(identifier):
        if identifier not in claims:
            raise ValueError('Unknown or retracted report claim')
        row = claims[identifier]
        statement = row['statement'].replace('\n', ' ').strip()
        return row['certainty']+': '+statement+' ['+identifier+'](CLAIMS_LEDGER.md)'
    text = ('# Iteración5 · reporte de evidencia\n\n'
            'Dictamen de aptitud reservado a la auditoría independiente y la aprobación ética. '
            'Este documento presenta únicamente afirmaciones vigentes del ledger. '
            'La integridad final y el SHA-256 del manifiesto se verifican en el recibo externo de sello; '
            'el manifiesto no se introduce dentro de su propio inventario.\n\n')
    for section in SECTIONS:
        values = plan['sections'][section]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ValueError('Distinct claim references required in each section')
        text += '## '+section+'\n\n'
        text += ('\n\n'.join(assertion(value) for value in values) if values else
                 'Sin afirmación respaldada asignada en este reporte; ver Bloqueos y Falta.')+'\n\n'
        if section == 'Atributos de calidad':
            text += '| Atributo | Criterio | Estado reportado | Evidencia |\n|---|---|---|---|\n'
            for attribute, criterion in CRITERIA.items():
                row = plan['attributes'][attribute]
                if (set(row) != {'status', 'claims'} or row['status'] not in STATUSES
                        or not isinstance(row['claims'], list)
                        or not row['claims'] and row['status'] != 'NO_MEDIDO'):
                    raise ValueError('Measured claims or explicit unmeasured attribute required; no GO synthesis')
                for identifier in row['claims']:
                    assertion(identifier)
                    if row['status'] in {'OBSERVADO', 'VERIFICACIÓN_FUNCIONAL'} and claims[identifier]['certainty'] != 'VERIFICADO':
                        raise ValueError('Inherited, projected or assumed claims cannot establish observed attributes')
                text += '| '+attribute+' | '+criterion+' | '+row['status']+' | '+', '.join(row['claims'])+' |\n'
            text += '\n'
    return text


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ('package', 'plan', 'output'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args(argv)
    require_limited()
    root, output = Path(args.package).resolve(), Path(args.output).resolve()
    if (root/'MANIFEST_SHA256.jsonl').exists() or output.parent != root or output.exists():
        raise ValueError('Unsealed package and new report destination required')
    text = render(root, json.loads(Path(args.plan).read_bytes()))
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    print(json.dumps(dict(status='REPORT22_FROM_VERIFIED_LEDGER', sections=22,
        attributes=11, participant_aptitude_not_decided=True, acceptance_not_synthesized=True)))


if __name__ == '__main__':
    main()
