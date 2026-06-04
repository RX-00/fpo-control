#!/usr/bin/env python3
"""Generate Viser assets for Isaac Lab robot visualizations.

This is a focused replacement for the extractor referenced by
``play_with_viser.py``.  The current implementation supports the
``Tracking-Flat-G1-v0`` whole-body tracking task by reading the task's URDF,
baking fixed child visuals into their nearest moving parent link, and writing
the metadata format consumed by :class:`isaaclab_fpo.viser.ViserIsaacLab`.

Example:
    python isaaclab_fpo/scripts/isaac_asset_extractor.py --task Tracking-Flat-G1-v0
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent
EXPERIMENTS_ROOT = PACKAGE_DIR.parent
WHOLE_BODY_ASSET_DIR = (
    EXPERIMENTS_ROOT
    / "thirdparty"
    / "whole_body_tracking"
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
)
TRACKING_G1_URDF = WHOLE_BODY_ASSET_DIR / "unitree_description" / "urdf" / "g1" / "main.urdf"
DEFAULT_ASSET_ROOT = PACKAGE_DIR / "viser_assets"


@dataclass(frozen=True)
class Joint:
    """Minimal URDF joint representation."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray


@dataclass(frozen=True)
class Visual:
    """A visual mesh attached to a URDF link."""

    link: str
    mesh_path: Path
    origin: np.ndarray
    scale: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract Viser assets for Isaac Lab playback.")
    parser.add_argument(
        "--task",
        required=True,
        help="Task name. Currently supported: Tracking-Flat-G1-v0.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output asset directory. Defaults to isaaclab_fpo/viser_assets/tracking_flat_g1_v0.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Override the URDF path. Useful for testing another compatible G1 URDF.",
    )
    return parser.parse_args()


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return the URDF roll-pitch-yaw rotation matrix."""
    roll, pitch, yaw = rpy

    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )
    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def parse_vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in value.split()], dtype=np.float64)


def parse_origin(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if element is None:
        return transform

    xyz = parse_vector(element.attrib.get("xyz"), (0.0, 0.0, 0.0))
    rpy = parse_vector(element.attrib.get("rpy"), (0.0, 0.0, 0.0))
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def resolve_mesh_path(filename: str, urdf_path: Path) -> Path:
    if filename.startswith("package://unitree_description/"):
        relative_path = filename.removeprefix("package://unitree_description/")
        return WHOLE_BODY_ASSET_DIR / "unitree_description" / relative_path
    if filename.startswith("file://"):
        return Path(filename.removeprefix("file://"))

    path = Path(filename)
    if path.is_absolute():
        return path
    return urdf_path.parent / path


def parse_urdf(urdf_path: Path) -> tuple[set[str], dict[str, Joint], dict[str, list[Visual]]]:
    root = ET.parse(urdf_path).getroot()

    links = {link.attrib["name"] for link in root.findall("link")}
    child_to_joint: dict[str, Joint] = {}
    visuals_by_link: dict[str, list[Visual]] = {link_name: [] for link_name in links}

    for joint_element in root.findall("joint"):
        parent = joint_element.find("parent")
        child = joint_element.find("child")
        if parent is None or child is None:
            continue

        joint = Joint(
            name=joint_element.attrib["name"],
            joint_type=joint_element.attrib.get("type", "fixed"),
            parent=parent.attrib["link"],
            child=child.attrib["link"],
            origin=parse_origin(joint_element.find("origin")),
        )
        child_to_joint[joint.child] = joint

    for link_element in root.findall("link"):
        link_name = link_element.attrib["name"]
        for visual_element in link_element.findall("visual"):
            geometry = visual_element.find("geometry")
            mesh = geometry.find("mesh") if geometry is not None else None
            if mesh is None or "filename" not in mesh.attrib:
                continue

            visuals_by_link.setdefault(link_name, []).append(
                Visual(
                    link=link_name,
                    mesh_path=resolve_mesh_path(mesh.attrib["filename"], urdf_path),
                    origin=parse_origin(visual_element.find("origin")),
                    scale=parse_vector(mesh.attrib.get("scale"), (1.0, 1.0, 1.0)),
                )
            )

    return links, child_to_joint, visuals_by_link


def find_root_link(links: set[str], child_to_joint: dict[str, Joint]) -> str:
    children = set(child_to_joint.keys())
    roots = sorted(links - children)
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one URDF root link, found {roots}")
    return roots[0]


def compute_parent_to_children(child_to_joint: dict[str, Joint]) -> dict[str, list[Joint]]:
    parent_to_children: dict[str, list[Joint]] = {}
    for joint in child_to_joint.values():
        parent_to_children.setdefault(joint.parent, []).append(joint)
    return parent_to_children


def collect_moving_bodies(root_link: str, child_to_joint: dict[str, Joint]) -> set[str]:
    moving = {root_link}
    for joint in child_to_joint.values():
        if joint.joint_type != "fixed":
            moving.add(joint.child)
    return moving


def assign_visual_owners(
    root_link: str,
    moving_bodies: set[str],
    parent_to_children: dict[str, list[Joint]],
) -> dict[str, tuple[str, np.ndarray]]:
    """Map every link to the moving body that should carry its visuals.

    The returned transform maps from the owner body frame to the link frame.
    Fixed descendants keep accumulating transforms.  Non-fixed children become
    their own moving owners and reset to identity.
    """
    assignments: dict[str, tuple[str, np.ndarray]] = {}

    def visit(link: str, owner: str, owner_to_link: np.ndarray) -> None:
        assignments[link] = (owner, owner_to_link)
        for joint in parent_to_children.get(link, []):
            if joint.joint_type == "fixed":
                visit(joint.child, owner, owner_to_link @ joint.origin)
            else:
                if joint.child not in moving_bodies:
                    raise RuntimeError(f"Non-fixed joint child {joint.child} is not marked as moving")
                visit(joint.child, joint.child, np.eye(4, dtype=np.float64))

    visit(root_link, root_link, np.eye(4, dtype=np.float64))
    return assignments


def load_visual_mesh(visual: Visual, transform: np.ndarray) -> trimesh.Trimesh:
    if not visual.mesh_path.exists():
        raise FileNotFoundError(f"Missing visual mesh: {visual.mesh_path}")

    mesh = trimesh.load_mesh(str(visual.mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh for {visual.mesh_path}, got {type(mesh)}")

    mesh = mesh.copy()
    if not np.allclose(visual.scale, np.ones(3)):
        scale_transform = np.eye(4, dtype=np.float64)
        scale_transform[0, 0] = visual.scale[0]
        scale_transform[1, 1] = visual.scale[1]
        scale_transform[2, 2] = visual.scale[2]
        mesh.apply_transform(scale_transform)
    mesh.apply_transform(transform @ visual.origin)
    return mesh


def build_body_meshes(
    moving_bodies: set[str],
    assignments: dict[str, tuple[str, np.ndarray]],
    visuals_by_link: dict[str, list[Visual]],
) -> dict[str, trimesh.Trimesh]:
    pieces_by_body: dict[str, list[trimesh.Trimesh]] = {body: [] for body in moving_bodies}

    for link, visuals in visuals_by_link.items():
        if not visuals:
            continue
        if link not in assignments:
            raise RuntimeError(f"Link {link} is unreachable from URDF root")

        owner, owner_to_link = assignments[link]
        if owner not in pieces_by_body:
            continue

        for visual in visuals:
            pieces_by_body[owner].append(load_visual_mesh(visual, owner_to_link))

    body_meshes: dict[str, trimesh.Trimesh] = {}
    for body_name, pieces in pieces_by_body.items():
        if not pieces:
            continue
        body_mesh = trimesh.util.concatenate(pieces)
        body_mesh.merge_vertices()
        body_meshes[body_name] = body_mesh
    return body_meshes


def write_yaml(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_asset_bundle(
    task: str,
    urdf_path: Path,
    output_dir: Path,
    root_link: str,
    moving_bodies: set[str],
    body_meshes: dict[str, trimesh.Trimesh],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ordered_bodies = [root_link]
    ordered_bodies.extend(sorted(body for body in moving_bodies if body != root_link))

    mappings: dict[str, str] = {}
    hierarchy: dict[str, dict] = {
        "/World": {
            "type": "Scope",
            "parent": None,
            "children": ["/World/envs"],
            "mesh_file": None,
            "transform": None,
            "is_instance": False,
            "prototype_path": None,
        },
        "/World/envs": {
            "type": "Scope",
            "parent": "/World",
            "children": ["/World/envs/env_0"],
            "mesh_file": None,
            "transform": None,
            "is_instance": False,
            "prototype_path": None,
        },
        "/World/envs/env_0": {
            "type": "Xform",
            "parent": "/World/envs",
            "children": ["/World/envs/env_0/Robot"],
            "mesh_file": None,
            "transform": np.eye(4).tolist(),
            "is_instance": False,
            "prototype_path": None,
        },
        "/World/envs/env_0/Robot": {
            "type": "Xform",
            "parent": "/World/envs/env_0",
            "children": [],
            "mesh_file": None,
            "transform": np.eye(4).tolist(),
            "is_instance": False,
            "prototype_path": None,
        },
    }

    total_vertices = 0
    total_faces = 0
    total_size = 0
    mesh_info: dict[str, dict] = {}

    for body_name in ordered_bodies:
        mesh = body_meshes.get(body_name)
        if mesh is None:
            continue

        mesh_filename = f"{body_name}.glb"
        mesh_path = mesh_dir / mesh_filename
        mesh.export(mesh_path)

        relative_mesh = f"meshes/{mesh_filename}"
        prim_path = f"/World/envs/env_0/Robot/{body_name}/visuals"
        body_path = f"/World/envs/env_0/Robot/{body_name}"

        mappings[prim_path] = relative_mesh
        hierarchy["/World/envs/env_0/Robot"]["children"].append(body_path)
        hierarchy[body_path] = {
            "type": "Xform",
            "parent": "/World/envs/env_0/Robot",
            "children": [prim_path],
            "mesh_file": None,
            "transform": np.eye(4).tolist(),
            "is_instance": False,
            "prototype_path": None,
        }
        hierarchy[prim_path] = {
            "type": "Mesh",
            "parent": body_path,
            "children": [],
            "mesh_file": relative_mesh,
            "transform": np.eye(4).tolist(),
            "is_instance": False,
            "prototype_path": None,
        }

        size_bytes = mesh_path.stat().st_size
        total_vertices += int(len(mesh.vertices))
        total_faces += int(len(mesh.faces))
        total_size += size_bytes
        mesh_info[body_name] = {
            "filename": relative_mesh,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "size_bytes": size_bytes,
        }

    metadata = {
        "task": task,
        "generated": generated,
        "extractor": "isaac_asset_extractor.py",
        "source": "whole_body_tracking_g1_urdf",
        "urdf": str(urdf_path),
    }

    write_yaml(
        output_dir / "prim_to_mesh.yaml",
        {
            "metadata": {
                **metadata,
                "description": "Mapping from generated robot body prim paths to GLB mesh files",
            },
            "mappings": mappings,
        },
    )
    write_yaml(
        output_dir / "scene_hierarchy.yaml",
        {
            "metadata": {
                **metadata,
                "description": "Minimal scene hierarchy for Viser playback",
                "transform_type": "body_local",
            },
            "hierarchy": hierarchy,
        },
    )
    write_yaml(
        output_dir / "extraction_info.yaml",
        {
            "metadata": metadata,
            "summary": {
                "moving_bodies": len(moving_bodies),
                "mesh_mappings": len(mappings),
                "total_vertices": total_vertices,
                "total_faces": total_faces,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 3),
            },
            "meshes": mesh_info,
            "notes": [
                "Visual meshes are exported in each Isaac moving body frame.",
                "Fixed child visuals, such as head_link and rubber hands, are baked into their nearest moving parent body.",
            ],
        },
    )


def main() -> None:
    args = parse_args()
    task_clean = args.task.lower().replace(":", "_").replace("-", "_")
    if task_clean != "tracking_flat_g1_v0":
        raise SystemExit(
            f"Unsupported task {args.task!r}. This extractor currently supports Tracking-Flat-G1-v0."
        )

    urdf_path = (args.urdf or TRACKING_G1_URDF).resolve()
    output_dir = (args.output_dir or (DEFAULT_ASSET_ROOT / task_clean)).resolve()

    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    links, child_to_joint, visuals_by_link = parse_urdf(urdf_path)
    root_link = find_root_link(links, child_to_joint)
    parent_to_children = compute_parent_to_children(child_to_joint)
    moving_bodies = collect_moving_bodies(root_link, child_to_joint)
    assignments = assign_visual_owners(root_link, moving_bodies, parent_to_children)
    body_meshes = build_body_meshes(moving_bodies, assignments, visuals_by_link)

    missing_meshes = sorted(moving_bodies - set(body_meshes))
    if missing_meshes:
        print(f"[WARNING] Moving bodies without visual meshes: {missing_meshes}")

    write_asset_bundle(args.task, urdf_path, output_dir, root_link, moving_bodies, body_meshes)

    print(f"[INFO] Wrote Viser assets to: {output_dir}")
    print(f"[INFO] Moving bodies: {len(moving_bodies)}")
    print(f"[INFO] Mesh mappings: {len(body_meshes)}")


if __name__ == "__main__":
    main()
