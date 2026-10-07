"""US L4 zones observed in the iteration5 API catalog; no stock inference."""
from scripts.study_operator.policy import OperatorError

US_L4_ZONES = frozenset(('us-central1-a', 'us-central1-b', 'us-central1-c',
    'us-west1-a', 'us-west1-b', 'us-west1-c', 'us-east1-b', 'us-east1-c', 'us-east1-d',
    'us-east4-a', 'us-east4-c', 'us-west4-a', 'us-west4-c'))


def region(zone):
    if zone not in US_L4_ZONES:
        raise OperatorError('Zona fuera del catálogo L4 de EE.UU. verificado. Revisa el inventario de capacidad; no crees recursos fuera del ámbito.')
    return zone.rsplit('-', 1)[0]
