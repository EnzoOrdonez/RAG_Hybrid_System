"""F-014: el limite del checkout depende de donde vive el repo, no de ROOT.parent.parent."""

from pathlib import Path
import tempfile

from src.ui.components.study_protocol import ROOT
from src.utils.checkout_boundary import checkout_boundary, is_inside_checkout


def test_worktree_layout_protects_the_parent_checkout(tmp_path):
    checkout = tmp_path / "repo"
    root = checkout / ".worktrees" / "wt"
    root.mkdir(parents=True)
    assert checkout_boundary(root) == checkout.resolve()
    assert is_inside_checkout(root / "out", root)
    assert is_inside_checkout(checkout / "data", root)  # el checkout padre
    assert is_inside_checkout(checkout / ".worktrees" / "otro" / "x", root)  # otro worktree
    assert not is_inside_checkout(tmp_path / "evidence", root)


def test_direct_checkout_boundary_is_the_root_itself(tmp_path):
    root = tmp_path / "home" / "Projects" / "repo"
    root.mkdir(parents=True)
    assert checkout_boundary(root) == root.resolve()
    assert is_inside_checkout(root / "out", root)
    # Regresion: antes el limite era dos niveles arriba y esto contaba como interior.
    assert not is_inside_checkout(tmp_path / "home" / "AppData" / "Temp" / "x", root)
    assert not is_inside_checkout(tmp_path / "home" / "Projects" / "otro", root)


def test_only_a_directory_named_worktrees_widens_the_boundary(tmp_path):
    root = tmp_path / "a" / "work" / "wt"
    root.mkdir(parents=True)
    assert checkout_boundary(root) == root.resolve()


def test_the_real_repo_allows_the_system_temp_directory():
    # Pasa tanto en el checkout principal como dentro de .worktrees/<x>.
    assert not is_inside_checkout(Path(tempfile.gettempdir()) / "evidence", ROOT)
    assert is_inside_checkout(ROOT / "data" / "x", ROOT)
