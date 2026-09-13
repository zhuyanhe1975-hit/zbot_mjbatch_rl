"""Extract the real ZBot 6-DoF module meshes from the pinned upstream USD asset."""

from __future__ import annotations

import tempfile
import urllib.request
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

REVISION = "aa340d41b803850a5b21e6ec6b9f76012f9580c0"
SOURCE = (
  "https://raw.githubusercontent.com/zhuyanhe1975-hit/zbot_rl_student/"
  f"{REVISION}/assets/zbot_usd/zbot/zbot_6s_new.usda"
)
BODIES = ("base", "b3", "a3", "b2", "a2", "b1", "foot_0", "b4", "a5", "b5", "a6", "foot_1")
OUTPUT = Path(__file__).parents[1] / "assets" / "meshes"


def triangulate(counts: np.ndarray, indices: np.ndarray):
  offset = 0
  for count in counts:
    face = indices[offset : offset + count]
    for i in range(1, count - 1):
      yield int(face[0]), int(face[i]), int(face[i + 1])
    offset += count


def export_mesh(stage: Usd.Stage, body: str) -> None:
  prim = stage.GetPrimAtPath(f"/zbot/{body}/visuals")
  mesh = UsdGeom.Mesh(prim)
  points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
  transform = np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation(), dtype=float)
  points = np.c_[points, np.ones(len(points))] @ transform
  counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
  indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)

  path = OUTPUT / f"{body}.obj"
  with path.open("w") as output:
    output.write(f"# Extracted from {SOURCE}\n")
    for x, y, z in points[:, :3]:
      output.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
    for a, b, c in triangulate(counts, indices):
      output.write(f"f {a + 1} {b + 1} {c + 1}\n")


def main() -> None:
  OUTPUT.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(suffix=".usda") as source_file:
    with urllib.request.urlopen(SOURCE) as response:
      source_file.write(response.read())
      source_file.flush()
    stage = Usd.Stage.Open(source_file.name)
    if stage is None:
      raise RuntimeError("could not open upstream ZBot USD")
    for body in BODIES:
      export_mesh(stage, body)
      print(f"exported {body}.obj")


if __name__ == "__main__":
  main()
