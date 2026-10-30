"""
Mini Digital Twin — two scans of the same scene → aligned, merged, meshed, compared.

Pipeline: load → preprocess → global registration (FPFH + RANSAC)
          → ICP refinement → merge → Poisson mesh → change map

Why two registration steps? ICP only finds the *nearest* good alignment.
Started from a bad pose it locks into a wrong one. The global step finds a
rough pose from shape features first; ICP then makes it precise.

Run:
  python src/mini_digital_twin.py
"""

import copy
import numpy as np
import open3d as o3d

from common import DATA_DIR

# =========================
# EDIT PATHS HERE
# =========================
A_PATH = DATA_DIR / "elephant.ply"   # first scan ("before")
B_PATH = None                        # second scan ("after"), e.g. DATA_DIR / "my_scan_2.ply"
# B_PATH = None → demo mode: B is a moved and slightly changed copy of A,
# so you can see registration and the change map work with one file.

# =========================
# PARAMETERS (safe defaults)
# =========================
VOXEL = 0.03
NB_NEIGHBORS = 20
STD_RATIO = 2.0

NORMAL_RADIUS = 0.10      # rule of thumb: 2–4 * VOXEL
NORMAL_MAX_NN = 30

ICP_MAX_CORR = 0.08       # increase (0.10–0.15) if ICP fails
POISSON_DEPTH = 9
DENSITY_Q = 0.02          # remove bottom 2% density vertices
CHANGE_THRESH = 0.03      # meters, e.g. 0.02–0.05

FPFH_RADIUS = VOXEL * 5   # feature radius for global registration
RANSAC_DIST = VOXEL * 1.5


def load_pcd(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {path}")
    return pcd


def preprocess(
    pcd: o3d.geometry.PointCloud,
    voxel: float,
    nb_neighbors: int,
    std_ratio: float,
    normal_radius: float,
    normal_max_nn: int,
) -> o3d.geometry.PointCloud:
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    # makes normals more consistent (optional but helps Poisson)
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


def make_demo_scan(a: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Simulated second scan: A rotated + shifted, with one region pushed outward."""
    b = copy.deepcopy(a)
    pts = np.asarray(b.points)
    center = pts.mean(axis=0)
    extent = pts.max(axis=0) - pts.min(axis=0)

    # local change: bump the points near one corner by 10% of the object size
    corner = pts.min(axis=0) + extent * 0.25
    near = np.linalg.norm(pts - corner, axis=1) < extent.max() * 0.15
    pts[near] += extent.max() * 0.10 * np.array([0.0, 0.0, 1.0])
    b.points = o3d.utility.Vector3dVector(pts)

    # different scanner pose
    R = b.get_rotation_matrix_from_xyz((0.3, -0.5, 0.8))
    b.rotate(R, center=center)
    b.translate(extent * np.array([0.4, -0.2, 0.1]))
    return b


def fpfh(pcd: o3d.geometry.PointCloud):
    """Fast Point Feature Histograms: a local shape descriptor per point."""
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=100)
    )


def global_register(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
) -> o3d.pipelines.registration.RegistrationResult:
    """Rough alignment from feature matches, robust to any starting pose."""
    reg = o3d.pipelines.registration
    return reg.registration_ransac_based_on_feature_matching(
        source, target, fpfh(source), fpfh(target),
        mutual_filter=True,
        max_correspondence_distance=RANSAC_DIST,
        estimation_method=reg.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            reg.CorrespondenceCheckerBasedOnDistance(RANSAC_DIST),
        ],
        criteria=reg.RANSACConvergenceCriteria(100000, 0.999),
    )


def icp_register(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_corr: float,
    init: np.ndarray,
) -> o3d.pipelines.registration.RegistrationResult:
    """Precise alignment, starting from the global result."""
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_corr, init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result


def poisson_mesh(
    pcd: o3d.geometry.PointCloud,
    depth: int,
    density_quantile: float,
) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)

    # simple cleanup: remove low-density vertices (floating junk)
    cutoff = np.quantile(densities, density_quantile)
    vertices_to_remove = densities < cutoff
    mesh.remove_vertices_by_mask(vertices_to_remove)

    mesh.compute_vertex_normals()
    return mesh


def change_map(
    a: o3d.geometry.PointCloud,
    b_aligned: o3d.geometry.PointCloud,
    threshold: float,
) -> o3d.geometry.PointCloud:
    d = np.asarray(a.compute_point_cloud_distance(b_aligned))
    colored = o3d.geometry.PointCloud(a)

    colors = np.zeros((len(d), 3), dtype=np.float64)
    changed = d > threshold

    # unchanged = gray, changed = red
    colors[~changed] = [0.6, 0.6, 0.6]
    colors[changed] = [1.0, 0.2, 0.2]

    colored.colors = o3d.utility.Vector3dVector(colors)
    return colored


def main():
    # 1) Load
    A_raw = load_pcd(str(A_PATH))
    B_raw = load_pcd(str(B_PATH)) if B_PATH else make_demo_scan(A_raw)

    # 2) Preprocess
    A = preprocess(A_raw, VOXEL, NB_NEIGHBORS, STD_RATIO, NORMAL_RADIUS, NORMAL_MAX_NN)
    B = preprocess(B_raw, VOXEL, NB_NEIGHBORS, STD_RATIO, NORMAL_RADIUS, NORMAL_MAX_NN)

    A.paint_uniform_color([0.9, 0.6, 0.1])   # A = orange
    B.paint_uniform_color([0.1, 0.5, 0.9])   # B = blue
    print(f"A points: {len(A.points)} | B points: {len(B.points)}")
    print("VIEW: A + B (before ICP)")
    o3d.visualization.draw_geometries([A, B])

    # 3a) Global registration: rough pose from shape features
    rough = global_register(B, A)
    print(f"Global fitness={rough.fitness:.4f}, rmse={rough.inlier_rmse:.4f}")

    # 3b) ICP: refine B -> A, starting from the rough pose
    reg = icp_register(B, A, ICP_MAX_CORR, init=rough.transformation)
    print(f"ICP fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.4f}")
    # Experiment: replace init=rough.transformation with np.eye(4) and compare

    B_aligned = copy.deepcopy(B)
    B_aligned.transform(reg.transformation)

    print("VIEW: A + B_aligned (after ICP)")
    o3d.visualization.draw_geometries([A, B_aligned])

    # 4) Merge
    merged = A + B_aligned
    merged = merged.voxel_down_sample(VOXEL)  # keep it lighter

    print("VIEW: merged cloud")
    o3d.visualization.draw_geometries([merged])

    # 5) Mesh reconstruction
    mesh = poisson_mesh(merged, depth=POISSON_DEPTH, density_quantile=DENSITY_Q)
    print("VIEW: mesh (Poisson)")
    o3d.visualization.draw_geometries([mesh])

    # 6) Change map
    cmap = change_map(A, B_aligned, threshold=CHANGE_THRESH)
    print("VIEW: change map (A colored by distance to B_aligned)")
    o3d.visualization.draw_geometries([cmap])


if __name__ == "__main__":
    main()
