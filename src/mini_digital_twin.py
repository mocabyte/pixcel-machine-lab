import copy
import numpy as np
import open3d as o3d

# =========================
# EDIT PATHS HERE
# =========================
A_PATH = "obj/classroom.ply"   # e.g. "obj/classroom.ply"
B_PATH = "obj/elephant.ply"   # e.g. "obj/classroom_2.ply"
# If you only have ONE file for quick testing:
# A_PATH = "obj/classroom.ply"
# B_PATH = "obj/classroom.ply"

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


def icp_register(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_corr: float,
) -> o3d.pipelines.registration.RegistrationResult:
    init = np.eye(4)
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
    A_raw = load_pcd(A_PATH)
    B_raw = load_pcd(B_PATH)

    # 2) Preprocess
    A = preprocess(A_raw, VOXEL, NB_NEIGHBORS, STD_RATIO, NORMAL_RADIUS, NORMAL_MAX_NN)
    B = preprocess(B_raw, VOXEL, NB_NEIGHBORS, STD_RATIO, NORMAL_RADIUS, NORMAL_MAX_NN)

    print(f"A points: {len(A.points)} | B points: {len(B.points)}")
    print("VIEW: A + B (before ICP)")
    o3d.visualization.draw_geometries([A, B])

    # 3) ICP: align B -> A
    reg = icp_register(B, A, ICP_MAX_CORR)
    print(f"ICP fitness={reg.fitness:.4f}, rmse={reg.inlier_rmse:.4f}")

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
