"""Limite del "checkout" para exigir que la evidencia de I7 viva fuera de el (F-014).

Antes el codigo tomaba ``ROOT.parent.parent`` como el checkout. Eso solo es cierto cuando el
repo vive en ``<checkout>/.worktrees/<nombre>``; en el checkout principal resolvia a la carpeta
del usuario y trataba todo ``AppData/Temp`` como interior. La regla es:

* ``ROOT`` dentro de ``<checkout>/.worktrees/<nombre>``: el limite es ``<checkout>`` (la
  evidencia tampoco puede escribirse en el checkout padre ni en sus otros worktrees).
* En cualquier otro caso: el limite es ``ROOT`` mismo.

Es una regla sobre rutas, sin llamar a git, para que sea determinista y se pueda probar.
"""

from pathlib import Path

WORKTREES_DIR = ".worktrees"


def checkout_boundary(root) -> Path:
    """Devuelve la raiz que no debe contener evidencia externa."""
    root = Path(root).resolve()
    parent = root.parent
    if parent.name.lower() == WORKTREES_DIR:
        return parent.parent
    return root


def is_inside_checkout(path, root) -> bool:
    """True si ``path`` queda dentro del checkout al que pertenece ``root``."""
    return Path(path).resolve().is_relative_to(checkout_boundary(root))
