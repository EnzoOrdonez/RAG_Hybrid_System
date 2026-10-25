import pytest

from src.ui.components.study_ueq import ITEMS, score, validate


def test_literal_polarity_order_and_official_spellings():
    assert ITEMS == (
        ('obstructivo', 'impulsor de apoyo'), ('complicado', 'facil'),
        ('ineficiente', 'eficiente'), ('confuso', 'claro'),
        ('aburrido', 'emocionante'), ('no interesante', 'interesante'),
        ('convencional', 'original'), ('convencional', 'novedoso'))


@pytest.mark.parametrize('position,expected', [(1, -3), (4, 0), (7, 3)])
def test_known_endpoints_and_midpoint(position, expected):
    result = score([position] * 8)
    assert result == dict(item_scores=[expected] * 8, pragmatic=expected, hedonic=expected, overall=expected)


def test_pragmatic_and_hedonic_are_separate_and_global_is_eight_item_mean():
    result = score([1, 2, 3, 4, 4, 5, 6, 7])
    assert result['pragmatic'] == -1.5 and result['hedonic'] == 1.5 and result['overall'] == 0


@pytest.mark.parametrize('values', [[4]*7, [4]*9, [None]*8, [True]*8, [4.0]*8, [0]*8, [8]*8, '44444444'])
def test_incomplete_or_noninteger_instruments_never_imputed(values):
    with pytest.raises(ValueError):
        validate(values)
