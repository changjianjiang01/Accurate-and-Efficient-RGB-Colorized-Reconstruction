import numpy as np
import cv2
import open3d as o3d
import yaml
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_pose_line(line):
    """Parse a line in pose file: timestamp tx ty tz qx qy qz qw"""
    parts = line.strip().split()
    timestamp = float(parts[0])
    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
    qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
    
    # Build rotation matrix
    rotation = R.from_quat([qx, qy, qz, qw])
    R_matrix = rotation.as_matrix()
    
    # Build transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_matrix
    T[:3, 3] = [tx, ty, tz]
    
    return timestamp, T


def load_poses(pose_file):
    """Load pose file"""
    poses = {}
    with open(pose_file, 'r') as f:
        for line in f:
            if line.strip():
                timestamp, T = parse_pose_line(line)
                poses[timestamp] = T
    return poses


def load_point_cloud(pcd_path):
    """Load point cloud file"""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    return np.asarray(pcd.points)


def load_image(image_path):
    """Load image file"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_camera_intrinsic(projection_config, image_shape):
    """
    Get camera intrinsics from configuration file
    
    Args:
        projection_config: Projection section in configuration file
        image_shape: Image dimensions (height, width, channels)
    
    Returns:
        K: Camera intrinsic matrix (3, 3)
        dist_coeffs: Distortion coefficients (4,)
    """
    # Get configuration parameters
    fx = projection_config.get('cam_fx', 1293.56944)
    fy = projection_config.get('cam_fy', 1293.3155)
    cx = projection_config.get('cam_cx', 626.91359)
    cy = projection_config.get('cam_cy', 522.799224)
    scale = projection_config.get('scale', 0.5)
    
    # Adjust intrinsics according to scale (if image is scaled)
    fx = fx * scale
    fy = fy * scale
    cx = cx * scale
    cy = cy * scale
    
    # Build intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Get distortion coefficients
    dist_coeffs = np.array([
        projection_config.get('cam_d0', -0.076160),
        projection_config.get('cam_d1', 0.123001),
        projection_config.get('cam_d2', -0.00113),
        projection_config.get('cam_d3', 0.000251)
    ])
    
    return K, dist_coeffs


def project_points_to_image(points_lidar, T_lidar, T_camera, Rcl, Pcl, K, dist_coeffs=None):
    """
    Project LiDAR point cloud to image plane
    
    Args:
        points_lidar: Point cloud in LiDAR body frame (N, 3)
        T_lidar: Transformation matrix from LiDAR body to world (4, 4)
        T_camera: Transformation matrix from camera body to world (4, 4)
        Rcl: Rotation matrix from camera body to LiDAR body (3, 3)
        Pcl: Translation vector from camera body to LiDAR body (3,)
        K: Camera intrinsic matrix (3, 3)
        dist_coeffs: Distortion coefficients (4,), optional
    
    Returns:
        points_camera: Point cloud in camera body frame (N, 3)
        pixel_coords: Pixel coordinates (N, 2)
        valid_mask: Valid point mask (N,)
    """
    # Rcl and Pcl are extrinsics from LiDAR body to camera body
    # Directly use extrinsics to transform point cloud from LiDAR body frame to camera body frame
    points_camera = (Rcl @ points_lidar.T).T + Pcl
    
    # Filter out points behind camera (z > 0 means in front of camera)
    valid_mask = points_camera[:, 2] > 0
    
    # Project to image plane
    points_camera_valid = points_camera[valid_mask]
    if len(points_camera_valid) == 0:
        pixel_coords_full = np.zeros((points_lidar.shape[0], 2))
        return points_camera, pixel_coords_full, valid_mask
    
    # Project to pixel coordinates (with distortion)
    points_camera_homo = points_camera_valid / points_camera_valid[:, 2:3]
    pixel_coords_distorted = (K @ points_camera_homo.T).T[:, :2]
    
    # If distortion coefficients exist, perform distortion correction
    if dist_coeffs is not None and np.any(np.abs(dist_coeffs) > 1e-6):
        # Use OpenCV's undistortPoints for distortion correction
        # Input: distorted pixel coordinates
        # Output: undistorted pixel coordinates (P=K)
        pixel_coords_distorted_reshaped = pixel_coords_distorted.reshape(-1, 1, 2).astype(np.float32)
        pixel_coords = cv2.undistortPoints(
            pixel_coords_distorted_reshaped, K, dist_coeffs, P=K
        ).reshape(-1, 2)
    else:
        # No distortion: directly use projection result
        pixel_coords = pixel_coords_distorted
    
    # Create complete pixel_coords
    pixel_coords_full = np.zeros((points_lidar.shape[0], 2))
    pixel_coords_full[valid_mask] = pixel_coords
    
    return points_camera, pixel_coords_full, valid_mask


def colorize_point_cloud(points_lidar, pixel_coords, valid_mask, image):
    """Get colors from image and colorize point cloud"""
    height, width = image.shape[:2]
    
    # Ensure pixel coordinates are within image bounds
    pixel_coords_int = pixel_coords.astype(int)
    in_bounds = (pixel_coords_int[:, 0] >= 0) & (pixel_coords_int[:, 0] < width) & \
                (pixel_coords_int[:, 1] >= 0) & (pixel_coords_int[:, 1] < height)
    
    valid_mask = valid_mask & in_bounds
    
    # Get colors
    colors = np.zeros((points_lidar.shape[0], 3))
    for i in range(points_lidar.shape[0]):
        if valid_mask[i]:
            u, v = pixel_coords_int[i]
            colors[i] = image[v, u] / 255.0  # Normalize to [0, 1]
    
    return colors, valid_mask


def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = load_config(config_path)
    
    projection_config = config['projection']
    Rcl = np.array(projection_config['Rcl']).reshape(3, 3)
    Pcl = np.array(projection_config['Pcl'])
    
    # Read data paths from config file
    base_dir = Path(__file__).parent.parent
    common_config = config['common']
    data_dir = base_dir / common_config['data_dir']
    image_dir = data_dir / common_config['image_dir']
    pcd_dir = data_dir / common_config['pcd_dir']
    image_pose_file = data_dir / common_config['image_pose_file']
    lidar_pose_file = data_dir / common_config['lidar_pose_file']
    
    # Load poses
    print("Loading pose files...")
    image_poses = load_poses(image_pose_file)
    lidar_poses = load_poses(lidar_pose_file)
    
    # Get matching timestamps
    common_timestamps = sorted(set(image_poses.keys()) & set(lidar_poses.keys()))
    print(f"Found {len(common_timestamps)} matching timestamps")
    
    if len(common_timestamps) == 0:
        print("Error: No matching timestamps found")
        return
    
    # Get camera intrinsics from config (use first image to get dimensions)
    first_image_path = image_dir / f"{common_timestamps[0]}.png"
    first_image = load_image(first_image_path)
    K, dist_coeffs = get_camera_intrinsic(projection_config, first_image.shape)
    print(f"Camera intrinsics:\n{K}")
    print(f"Distortion coefficients: {dist_coeffs}")
    
    # Store all successfully colored points
    all_colored_points = []
    all_colored_colors = []
    all_uncolored_points = []
    all_behind_points = []
    # Store all point clouds (for map.pcd, all set to blue)
    all_map_points = []
    
    total_points = 0
    total_colored = 0
    
    # Process point clouds for all timestamps
    print(f"\nStarting to process {len(common_timestamps)} point clouds...")
    for idx, timestamp in enumerate(common_timestamps):
        if (idx + 1) % 100 == 0:
            print(f"Processing progress: {idx + 1}/{len(common_timestamps)}")
        
        # Load point cloud and image
        pcd_path = pcd_dir / f"{timestamp}.pcd"
        image_path = image_dir / f"{timestamp}.png"
        
        if not pcd_path.exists() or not image_path.exists():
            continue
        
        try:
            points_lidar = load_point_cloud(pcd_path)
            image = load_image(image_path)
            
            # Get poses
            T_lidar = lidar_poses[timestamp]
            T_camera = image_poses[timestamp]
            
            # Project point cloud to image
            points_camera, pixel_coords, valid_mask = project_points_to_image(
                points_lidar, T_lidar, T_camera, Rcl, Pcl, K, dist_coeffs
            )
            
            # Colorize point cloud
            colors, final_valid_mask = colorize_point_cloud(
                points_lidar, pixel_coords, valid_mask, image
            )
            
            total_points += len(points_lidar)
            total_colored += np.sum(final_valid_mask)
            
            # Analyze unprojected points
            not_projected_mask = ~valid_mask
            not_colored_mask = valid_mask & ~final_valid_mask
            
            # Collect successfully colored points
            valid_points = points_lidar[final_valid_mask]
            valid_colors = colors[final_valid_mask]
            valid_points_homo = np.hstack([valid_points, np.ones((valid_points.shape[0], 1))])
            points_world_colored = (T_lidar @ valid_points_homo.T).T[:, :3]
            
            if len(points_world_colored) > 0:
                all_colored_points.append(points_world_colored)
                all_colored_colors.append(valid_colors)
            
            # Collect points outside image
            uncolored_points = points_lidar[not_colored_mask]
            if len(uncolored_points) > 0:
                uncolored_points_homo = np.hstack([uncolored_points, np.ones((uncolored_points.shape[0], 1))])
                points_world_uncolored = (T_lidar @ uncolored_points_homo.T).T[:, :3]
                all_uncolored_points.append(points_world_uncolored)
            
            # Collect points behind camera
            behind_points = points_lidar[not_projected_mask]
            if len(behind_points) > 0:
                behind_points_homo = np.hstack([behind_points, np.ones((behind_points.shape[0], 1))])
                points_world_behind = (T_lidar @ behind_points_homo.T).T[:, :3]
                all_behind_points.append(points_world_behind)
            
            # Collect all point clouds to map (all points, for map.pcd)
            all_points_homo = np.hstack([points_lidar, np.ones((points_lidar.shape[0], 1))])
            points_world_all = (T_lidar @ all_points_homo.T).T[:, :3]
            all_map_points.append(points_world_all)
                
        except Exception as e:
            print(f"Error processing {timestamp}: {e}")
            continue
    
    print(f"\nProcessing completed!")
    print(f"Total points: {total_points}")
    print(f"Successfully colored points: {total_colored}")
    
    # Merge all point clouds
    print("\nMerging point clouds...")
    if len(all_colored_points) > 0:
        all_colored_points_merged = np.vstack(all_colored_points)
        all_colored_colors_merged = np.vstack(all_colored_colors)
    else:
        all_colored_points_merged = np.empty((0, 3))
        all_colored_colors_merged = np.empty((0, 3))
    
    if len(all_uncolored_points) > 0:
        all_uncolored_points_merged = np.vstack(all_uncolored_points)
        gray_colors = np.ones((len(all_uncolored_points_merged), 3)) * 0.5
    else:
        all_uncolored_points_merged = np.empty((0, 3))
        gray_colors = np.empty((0, 3))
    
    if len(all_behind_points) > 0:
        all_behind_points_merged = np.vstack(all_behind_points)
        red_colors = np.zeros((len(all_behind_points_merged), 3))
        red_colors[:, 0] = 1.0
    else:
        all_behind_points_merged = np.empty((0, 3))
        red_colors = np.empty((0, 3))
    
    # Merge all point clouds for map.pcd
    if len(all_map_points) > 0:
        all_map_points_merged = np.vstack(all_map_points)
        # Set all points to blue
        blue_colors = np.zeros((len(all_map_points_merged), 3))
        blue_colors[:, 2] = 1.0  # Set B channel to 1.0 (blue)
    else:
        all_map_points_merged = np.empty((0, 3))
        blue_colors = np.empty((0, 3))
    
    # Create Open3D point cloud objects - colored point cloud
    print("Creating Open3D point clouds...")
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(all_colored_points_merged)
    pcd_colored.colors = o3d.utility.Vector3dVector(all_colored_colors_merged)
    
    # Create uncolored point cloud (displayed in gray)
    pcd_uncolored = o3d.geometry.PointCloud()
    if len(all_uncolored_points_merged) > 0:
        pcd_uncolored.points = o3d.utility.Vector3dVector(all_uncolored_points_merged)
        pcd_uncolored.colors = o3d.utility.Vector3dVector(gray_colors)
    
    # Create point cloud behind camera (displayed in red)
    pcd_behind = o3d.geometry.PointCloud()
    if len(all_behind_points_merged) > 0:
        pcd_behind.points = o3d.utility.Vector3dVector(all_behind_points_merged)
        pcd_behind.colors = o3d.utility.Vector3dVector(red_colors)
    
    # Save point clouds
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Save all colored point clouds as colormap.pcd (only successfully colored points)
    colormap_path = output_dir / "colormap.pcd"
    o3d.io.write_point_cloud(str(colormap_path), pcd_colored)
    print(f"\nSaved colored point cloud to: {colormap_path}")
    print(f"  Number of points: {len(all_colored_points_merged)}")
    
    # Save all point clouds as map.pcd (all points set to blue, no projection coloring)
    pcd_map = o3d.geometry.PointCloud()
    pcd_map.points = o3d.utility.Vector3dVector(all_map_points_merged)
    pcd_map.colors = o3d.utility.Vector3dVector(blue_colors)
    
    map_path = output_dir / "map.pcd"
    o3d.io.write_point_cloud(str(map_path), pcd_map)
    print(f"Saved full point cloud to: {map_path}")
    print(f"  Total points: {len(all_map_points_merged)} (all blue)")
    
    # Visualize all point clouds
    print("\nVisualizing point clouds...")
    geometries = [pcd_colored]
    if len(all_uncolored_points_merged) > 0:
        geometries.append(pcd_uncolored)
    if len(all_behind_points_merged) > 0:
        geometries.append(pcd_behind)
    
    o3d.visualization.draw_geometries(geometries,
                                      window_name="Point Cloud Visualization (Colored=Colored, Gray=Outside Image, Red=Behind Camera)",
                                      width=1920,
                                      height=1080)
    
    print("Completed!")


if __name__ == "__main__":
    main()
