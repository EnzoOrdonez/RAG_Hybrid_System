"""Literal Spanish UEQ-S, separate from SUS and the frozen RAG procedure.

Items/order/polarity: official UEQS_Items.pdf, Spanish version, page 1.
Instruction is application UX text; the handbook supplies no standard short instruction.
"""
import copy
import json
from pathlib import Path

SOURCE_URL = 'https://www.ueq-online.org/Material/UEQS_Items.pdf'
HANDBOOK_URL = 'https://www.ueq-online.org/Material/Handbook.pdf'
ITEMS = (
    ('obstructivo', 'impulsor de apoyo'),
    ('complicado', 'facil'),
    ('ineficiente', 'eficiente'),
    ('confuso', 'claro'),
    ('aburrido', 'emocionante'),
    ('no interesante', 'interesante'),
    ('convencional', 'original'),
    ('convencional', 'novedoso'),
)
INSTRUCTION = ('Para cada par de palabras, marca una de las siete posiciones según tu experiencia con este sistema. '
               'No hay respuestas correctas o incorrectas. Esta parte toma aproximadamente un minuto.')
SOURCE_FILE = Path(__file__).resolve().parents[3] / 'config/UEQS_ES_official.json'


def validate(values):
    if not isinstance(values, (list, tuple)) or len(values) != 8 or any(
            type(value) is not int or not 1 <= value <= 7 for value in values):
        raise ValueError('Complete all eight UEQ-S items with integer positions 1..7')
    return list(values)


def score(values):
    transformed = [value - 4 for value in validate(values)]
    return dict(item_scores=transformed, pragmatic=sum(transformed[:4]) / 4,
                hedonic=sum(transformed[4:]) / 4, overall=sum(transformed) / 8)


def instrument():
    value = json.loads(SOURCE_FILE.read_text(encoding='utf-8'))
    if (value.get('schema_version') != 1 or value.get('source_url') != SOURCE_URL
            or value.get('items') != [list(pair) for pair in ITEMS]
            or value.get('instruction') != INSTRUCTION
            or value.get('positions') != list(range(1, 8))
            or value.get('negative_on_left') is not True):
        raise ValueError('Literal UEQ-S text, order or polarity changed')
    return copy.deepcopy(value)
