from types import SimpleNamespace
import numpy as np
import trimesh
import hpsdecode

from app.geometry.io import load_mesh


def test_hps_dcm_becomes_mesh(tmp_path, monkeypatch):
    source = trimesh.creation.box()
    monkeypatch.setattr(hpsdecode, "load_hps", lambda _: (None, SimpleNamespace(
        vertices=np.asarray(source.vertices), faces=np.asarray(source.faces)
    )))
    path = tmp_path / "scan.dcm"
    path.write_bytes(b"HPS")
    mesh = load_mesh(path)
    assert mesh.is_watertight and len(mesh.faces) == 12
