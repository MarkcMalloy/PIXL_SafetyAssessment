import numpy as np
import open3d as o3d
import cv2
from pathlib import Path
from .preprocessing import normalize_uint8


def depth_to_mesh(
    depth_path: str,
    output_dir: str,
    scale: float = 1.0,
    mask_path: str = None,
    poisson_depth: int = 9
):
    """
    Convert a depth map (from photometric stereo) into a 3D mesh and save as .ply and .stl.

    Args:
        depth_path: Path to depth.npy or depth.png
        output_dir: Folder to save mesh files
        scale: Scale factor to adjust Z exaggeration
        mask_path: Optional mask to zero out background
        poisson_depth: Octree depth for Poisson surface reconstruction
                       (use 7 for 'basic'/'tilted', 6 for 'point')
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Load depth map ---
    if depth_path.endswith(".npy"):
        z = np.load(depth_path)
    else:
        z = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        z = z.astype(np.float32)
        if z.max() > 1:
            z /= 255.0

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        z[mask == 0] = np.nan

    H, W = z.shape
    print(f"Loaded depth map: {W}x{H}")

    # --- Create grid of (x, y, z) points ---
    y, x = np.mgrid[0:H, 0:W]
    x = (x - W / 2) * scale
    y = (y - H / 2) * scale
    pts = np.stack((x, -y, z * scale), axis=-1).reshape(-1, 3)
    valid = np.isfinite(pts[:, 2])
    pts = pts[valid]

    # --- Create point cloud ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Estimate normals before Poisson reconstruction
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5 * scale, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(50)

    o3d.io.write_point_cloud(str(Path(output_dir) / "surface_pointcloud.ply"), pcd)
    print(f"Saved point cloud to {output_dir}/surface_pointcloud.ply")

    # --- Poisson reconstruction ---
    try:
        print(f"Running Poisson reconstruction (depth={poisson_depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
        if len(mesh.vertices) == 0:
            raise RuntimeError("Empty mesh returned by Poisson reconstruction.")
        mesh.compute_vertex_normals()
    except Exception as e:
        print(f"[WARNING] Poisson reconstruction failed at depth={poisson_depth}: {e}")
        print("[INFO] Falling back to Ball Pivoting reconstruction...")
        avg_dist = np.mean(pcd.compute_nearest_neighbor_distance())
        radii = o3d.utility.DoubleVector([avg_dist, 2 * avg_dist])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
        mesh.compute_vertex_normals()

    # --- Save mesh files ---
    ply_path = str(Path(output_dir) / "surface_mesh.ply")
    stl_path = str(Path(output_dir) / "surface_mesh.stl")
    o3d.io.write_triangle_mesh(ply_path, mesh)
    o3d.io.write_triangle_mesh(stl_path, mesh)
    print(f"Saved mesh as:\n - {ply_path}\n - {stl_path}")

    return mesh
