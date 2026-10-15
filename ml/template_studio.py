"""Build reviewable crown candidates from the 32 registered full-tooth STLs.

The meshes have no labelled CEJ. A slice from this tool is a proposed cervical
boundary, never an anatomically verified CEJ. Do not import candidates into the
design template directory until a designer has reviewed them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


FDIS = tuple(q * 10 + n for q in range(1, 5) for n in range(1, 9))


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if length < 1e-9:
        raise ValueError("Cannot form a local axis from coincident teeth")
    return v / length


def frame_for(fdi, centers):
    center = centers[fdi]
    quadrant, number = divmod(fdi, 10)
    neighbor = quadrant * 10 + (number - 1 if number > 1 else 2)
    mesial = unit(centers[neighbor][:2] - center[:2])
    if number == 1:
        mesial = -mesial
    occlusal = np.array([0., 0., -1. if quadrant < 3 else 1.])
    jaw = [v for k, v in centers.items() if k // 10 in ((1, 2) if quadrant < 3 else (3, 4))]
    arch_center = np.mean(jaw, axis=0)
    outward = center[:2] - arch_center[:2]
    buccal = outward - mesial * np.dot(outward, mesial)
    buccal = unit(np.r_[buccal, 0.])
    mesial = np.r_[mesial, 0.]
    return {"origin_world_mm": center.tolist(), "mesial": mesial.tolist(),
            "distal": (-mesial).tolist(), "buccal": buccal.tolist(),
            "lingual": (-buccal).tolist(), "occlusal": occlusal.tolist(),
            "method": "arch_neighbor_and_jaw_centroid", "verified": False}


def cut_candidate(mesh, fdi, cut_z):
    normal = [0, 0, -1 if fdi // 10 < 3 else 1]
    section = mesh.section(plane_origin=[0, 0, cut_z], plane_normal=normal)
    if section is None:
        raise ValueError("No section at proposed cervical height")
    loops = [np.asarray(loop) for loop in section.discrete if len(loop) >= 3]
    if not loops:
        raise ValueError("No cervical section loop")
    loops.sort(key=lambda x: np.linalg.norm(np.diff(x, axis=0), axis=1).sum(), reverse=True)
    crown = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=normal, plane_origin=[0, 0, cut_z], cap=True)
    if crown is None or not crown.is_watertight or len(crown.split()) != 1:
        raise ValueError("Candidate crown is not one watertight component")
    return crown, loops


def regions_and_landmarks(crown, frame):
    vectors = {k: np.asarray(frame[k]) for k in ("mesial", "distal", "buccal", "lingual", "occlusal")}
    vertices = np.asarray(crown.vertices)
    local_height = (vertices - frame["origin_world_mm"]) @ vectors["occlusal"]
    height_gate = np.quantile(local_height, .72)
    centers = crown.triangles_center
    normals = crown.face_normals
    scores = np.column_stack([normals @ vectors[k] for k in vectors])
    labels = np.argmax(scores, axis=1)
    face_height = (centers - frame["origin_world_mm"]) @ vectors["occlusal"]
    labels[(face_height < height_gate) & (labels == 4)] = np.argmax(scores[(face_height < height_gate) & (labels == 4), :4], axis=1)
    names = list(vectors)
    regions = {name: np.flatnonzero(labels == i).tolist() for i, name in enumerate(names)}
    landmarks = {name + "_extreme_world_mm": vertices[np.argmax((vertices - frame["origin_world_mm"]) @ direction)].tolist()
                 for name, direction in vectors.items()}
    landmarks["method"] = "directional_mesh_extrema_not_anatomic_cusps"
    landmarks["verified"] = False
    return regions, landmarks


def build(source: Path, output: Path, *, only_fdi=None, cut_z=None):
    paths = {fdi: source / f"FDI_{fdi}.stl" for fdi in FDIS}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing full-tooth meshes: " + ", ".join(missing))
    meshes = {fdi: trimesh.load(path, force="mesh") for fdi, path in paths.items()}
    centers = {fdi: mesh.bounds.mean(axis=0) for fdi, mesh in meshes.items()}
    results = []
    for fdi in ([only_fdi] if only_fdi else FDIS):
        mesh = meshes[fdi]
        lo, hi = mesh.bounds[:, 2]
        # No labelled CEJ exists in this library. Mid-height is only a review
        # candidate; explicit cut_z lets a designer replace it after inspection.
        z = float(cut_z if cut_z is not None else (lo + hi) / 2)
        directory = output / f"FDI_{fdi}"
        directory.mkdir(parents=True, exist_ok=True)
        record = {"fdi": fdi, "source_sha256": hashlib.sha256(paths[fdi].read_bytes()).hexdigest(),
                  "cut_z_world_mm": z, "cut_method": "review_candidate_mid_axial_extent" if cut_z is None else "manual_z_override",
                  "status": "REVIEW_REQUIRED", "cej_verified": False}
        try:
            crown, loops = cut_candidate(mesh, fdi, z)
            frame = frame_for(fdi, centers)
            regions, landmarks = regions_and_landmarks(crown, frame)
            crown.export(directory / "crown_template.stl")
            record.update({"faces": len(crown.faces), "watertight": crown.is_watertight,
                           "crown_height_mm": float(crown.extents[2]), "boundary_loops": len(loops)})
            (directory / "cervical_boundary.json").write_text(json.dumps({
                "status": "PROPOSED_NOT_CEJ", "cut_z_world_mm": z,
                "loops_world_mm": [loop[::max(1, len(loop) // 256)].tolist() for loop in loops],
            }, indent=2), encoding="utf-8")
            (directory / "local_frame.json").write_text(json.dumps(frame, indent=2), encoding="utf-8")
            (directory / "semantic_regions.json").write_text(json.dumps({
                "method": "face_normal_directional_candidate", "verified": False,
                "face_indices": regions}, indent=2), encoding="utf-8")
            (directory / "landmarks.json").write_text(json.dumps(landmarks, indent=2), encoding="utf-8")
        except Exception as exc:
            record.update(status="FAILED", error=str(exc))
        (directory / "review.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        results.append(record)
    (output / "manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fdi", type=int, choices=FDIS)
    parser.add_argument("--cut-z", type=float, help="Designer reviewed world Z cut for --fdi")
    args = parser.parse_args()
    if args.cut_z is not None and args.fdi is None:
        parser.error("--cut-z requires --fdi")
    rows = build(args.source, args.out, only_fdi=args.fdi, cut_z=args.cut_z)
    print(json.dumps({"total": len(rows), "candidates": sum(r["status"] == "REVIEW_REQUIRED" for r in rows),
                      "failed": [r for r in rows if r["status"] == "FAILED"]}, ensure_ascii=False))
