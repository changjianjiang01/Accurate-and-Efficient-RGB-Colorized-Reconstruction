import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
import colorsys
from tqdm import tqdm


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def voxelize_point_cloud(points, voxel_size):
    """
    Voxelize point cloud
    
    Args:
        points: Point cloud coordinates (N, 3)
        voxel_size: Voxel size
    
    Returns:
        voxel_dict: Dictionary, key is voxel index (x, y, z), value is list of point indices in that voxel
    """
    # Calculate voxel index for each point
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Use dictionary to store points in each voxel
    voxel_dict = defaultdict(list)
    for i, idx in enumerate(voxel_indices):
        voxel_key = tuple(idx)
        voxel_dict[voxel_key].append(i)
    
    return voxel_dict


def calculate_planarity_and_curvature(points_in_voxel):
    """
    Calculate planarity and curvature of points in voxel
    
    Args:
        points_in_voxel: Point cloud coordinates in voxel (M, 3)
    
    Returns:
        planarity: Planarity value (lambda2 - lambda3) / lambda1
        curvature: Curvature value lambda3 / (lambda1 + lambda2 + lambda3)
        eigenvalues: Eigenvalues [lambda1, lambda2, lambda3], lambda1 >= lambda2 >= lambda3
    """
    if len(points_in_voxel) < 3:
        # Too few points, cannot calculate
        return None, None, None
    
    # Calculate covariance matrix
    points_centered = points_in_voxel - points_in_voxel.mean(axis=0)
    cov_matrix = np.cov(points_centered.T)
    
    # Calculate eigenvalues (sorted in descending order)
    eigenvalues = np.linalg.eigvals(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
    
    # lambda1 >= lambda2 >= lambda3
    lambda1, lambda2, lambda3 = eigenvalues[0], eigenvalues[1], eigenvalues[2]
    
    # Calculate planarity: (lambda2 - lambda3) / lambda1
    if lambda1 > 1e-10:  # Avoid division by zero
        planarity = (lambda2 - lambda3) / lambda1
    else:
        planarity = 0.0
    
    # Calculate curvature: lambda3 / (lambda1 + lambda2 + lambda3)
    lambda_sum = lambda1 + lambda2 + lambda3
    if lambda_sum > 1e-10:  # Avoid division by zero
        curvature = lambda3 / lambda_sum
    else:
        curvature = 0.0
    
    return planarity, curvature, eigenvalues


def calculate_plane_thickness(points_in_voxel):
    """
    Calculate plane thickness of points in voxel
    
    Args:
        points_in_voxel: Point cloud coordinates in voxel (M, 3)
    
    Returns:
        thickness: Plane thickness (distribution range of point cloud along normal vector direction)
    """
    if len(points_in_voxel) < 3:
        # Too few points, cannot calculate plane
        return None
    
    # Calculate covariance matrix
    points_centered = points_in_voxel - points_in_voxel.mean(axis=0)
    cov_matrix = np.cov(points_centered.T)
    
    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Direction corresponding to minimum eigenvalue is normal vector direction (lambda3 direction)
    normal = eigenvectors[:, 2]  # Normal vector direction (minimum eigenvalue)
    
    # Calculate projection of all points along normal vector direction
    projections = np.dot(points_centered, normal)
    
    # Plane thickness = range of projection values
    thickness = projections.max() - projections.min()
    
    return thickness


def evaluate_geometry(pcd_path, config):
    """
    Evaluate geometric accuracy of point cloud
    
    Args:
        pcd_path: Point cloud file path
        config: Configuration dictionary
    
    Returns:
        results: Dictionary containing evaluation results
    """
    # Read configuration parameters
    resolution = config['geometry']['resolution']
    voxel_size = config['geometry']['voxel_size']
    
    # Load point cloud
    print(f"Loading point cloud: {pcd_path}")
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    points = np.asarray(pcd.points)
    
    if len(points) == 0:
        print("Error: Point cloud is empty")
        return None
    
    print(f"Point cloud points: {len(points)}")
    print(f"Point cloud range: x=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Voxelize
    print(f"\nVoxelizing point cloud (voxel_size={voxel_size})...")
    voxel_dict = voxelize_point_cloud(points, voxel_size)
    print(f"Number of voxels: {len(voxel_dict)}")
    
    # Get planarity, curvature, thickness and point count thresholds
    planarity_threshold = config['geometry'].get('planarity_threshold', 0.5)
    curvature_threshold = config['geometry'].get('curvature_threshold', 0.1)
    min_thickness_threshold = config['geometry'].get('min_thickness_threshold', 0.0)
    min_points_per_voxel = config['geometry'].get('min_points_per_voxel', 15)
    print(f"Planarity threshold: {planarity_threshold}")
    print(f"Curvature threshold: {curvature_threshold}")
    print(f"Minimum thickness threshold: {min_thickness_threshold}")
    print(f"Minimum points per voxel: {min_points_per_voxel}")
    
    # Calculate plane thickness, planarity and curvature for each voxel
    print("\nCalculating plane thickness, planarity and curvature for each voxel...")
    voxel_thicknesses = []
    voxel_geometric_accuracies = []
    voxel_planarities = []
    voxel_curvatures = []
    valid_voxels = 0
    filtered_by_points = 0
    filtered_by_planarity = 0
    filtered_by_curvature = 0
    filtered_by_thickness = 0
    valid_voxel_keys = []  # Store list of valid voxel keys for matching after 3-sigma filtering
    valid_voxel_dict = {}  # Store valid voxel information for visualization
    
    for voxel_key, point_indices in tqdm(voxel_dict.items(), desc="Processing voxels", ncols=80):
        # Filter out voxels with fewer points than threshold
        if len(point_indices) < min_points_per_voxel:
            filtered_by_points += 1
            continue
        
        points_in_voxel = points[point_indices]
        
        # Calculate planarity and curvature
        planarity, curvature, eigenvalues = calculate_planarity_and_curvature(points_in_voxel)
        
        if planarity is None or curvature is None:
            continue
        
        # Only count voxels with planarity greater than threshold
        if planarity <= planarity_threshold:
            filtered_by_planarity += 1
            continue
        
        # Only count voxels with curvature less than threshold (smaller curvature means closer to plane)
        if curvature >= curvature_threshold:
            filtered_by_curvature += 1
            continue
        
        # Calculate plane thickness
        thickness = calculate_plane_thickness(points_in_voxel)
        
        if thickness is None or thickness <= min_thickness_threshold:
            filtered_by_thickness += 1
            continue
        
        # Record valid voxel data
        voxel_thicknesses.append(thickness)
        voxel_planarities.append(planarity)
        voxel_curvatures.append(curvature)
        
        # Calculate geometric accuracy: (plane thickness - resolution) / resolution
        geometric_accuracy = (thickness - resolution) / resolution
        voxel_geometric_accuracies.append(geometric_accuracy)
        valid_voxels += 1
        
        # Record valid voxel information (before 3-sigma filtering)
        valid_voxel_keys.append(voxel_key)
        valid_voxel_dict[voxel_key] = point_indices
    
    if len(voxel_thicknesses) == 0:
        print("Error: No valid voxels")
        return None
    
    # Convert to numpy arrays
    voxel_thicknesses = np.array(voxel_thicknesses)
    voxel_geometric_accuracies = np.array(voxel_geometric_accuracies)
    voxel_planarities = np.array(voxel_planarities)
    voxel_curvatures = np.array(voxel_curvatures)
    
    # Use 3-sigma rule to remove thickness outliers
    sigma_multiplier = config['geometry'].get('sigma_multiplier', 3.0)
    mean_thickness = np.mean(voxel_thicknesses)
    std_thickness = np.std(voxel_thicknesses)
    thickness_lower = mean_thickness - sigma_multiplier * std_thickness
    thickness_upper = mean_thickness + sigma_multiplier * std_thickness
    
    # Filter outliers
    inlier_mask = (voxel_thicknesses >= thickness_lower) & (voxel_thicknesses <= thickness_upper)
    filtered_by_sigma = np.sum(~inlier_mask)
    
    if filtered_by_sigma > 0:
        print(f"\nUsing {sigma_multiplier}-sigma rule to remove outliers:")
        print(f"  Mean thickness: {mean_thickness:.6f} m")
        print(f"  Std dev: {std_thickness:.6f} m")
        print(f"  Thickness range: [{thickness_lower:.6f}, {thickness_upper:.6f}] m")
        print(f"  Number of outliers removed: {filtered_by_sigma}")
    
    # Keep only inliers
    voxel_thicknesses = voxel_thicknesses[inlier_mask]
    voxel_geometric_accuracies = voxel_geometric_accuracies[inlier_mask]
    voxel_planarities = voxel_planarities[inlier_mask]
    voxel_curvatures = voxel_curvatures[inlier_mask]
    valid_voxels = len(voxel_thicknesses)
    
    # Filter valid voxel dictionary, keep only voxels passing 3-sigma filter
    filtered_valid_voxel_dict = {}
    for idx, voxel_key in enumerate(valid_voxel_keys):
        if inlier_mask[idx]:
            filtered_valid_voxel_dict[voxel_key] = valid_voxel_dict[voxel_key]
    
    if len(voxel_thicknesses) == 0:
        print("Error: No valid voxels after 3-sigma filtering")
        return None
    
    results = {
        'total_voxels': len(voxel_dict),
        'valid_voxels': valid_voxels,
        'filtered_by_points': filtered_by_points,
        'filtered_by_planarity': filtered_by_planarity,
        'filtered_by_curvature': filtered_by_curvature,
        'filtered_by_thickness': filtered_by_thickness,
        'filtered_by_sigma': int(filtered_by_sigma),
        'planarity_threshold': planarity_threshold,
        'curvature_threshold': curvature_threshold,
        'min_thickness_threshold': min_thickness_threshold,
        'min_points_per_voxel': min_points_per_voxel,
        'sigma_multiplier': sigma_multiplier,
        'mean_thickness': np.mean(voxel_thicknesses),
        'std_thickness': np.std(voxel_thicknesses),
        'min_thickness': np.min(voxel_thicknesses),
        'max_thickness': np.max(voxel_thicknesses),
        'mean_planarity': np.mean(voxel_planarities),
        'std_planarity': np.std(voxel_planarities),
        'min_planarity': np.min(voxel_planarities),
        'max_planarity': np.max(voxel_planarities),
        'mean_curvature': np.mean(voxel_curvatures),
        'std_curvature': np.std(voxel_curvatures),
        'min_curvature': np.min(voxel_curvatures),
        'max_curvature': np.max(voxel_curvatures),
        'mean_geometric_accuracy': np.mean(voxel_geometric_accuracies),
        'std_geometric_accuracy': np.std(voxel_geometric_accuracies),
        'min_geometric_accuracy': np.min(voxel_geometric_accuracies),
        'max_geometric_accuracy': np.max(voxel_geometric_accuracies),
        'resolution': resolution,
        'voxel_size': voxel_size,
        'valid_voxel_dict': filtered_valid_voxel_dict,  # For visualization
        'points': points  # Original point cloud, for visualization
    }
    
    return results


def visualize_valid_voxels(results):
    """
    Visualize points in all valid voxels participating in geometric accuracy calculation
    Adjacent voxels are displayed in different colors
    
    Args:
        results: Result dictionary returned by evaluate_geometry
    """
    if 'valid_voxel_dict' not in results or 'points' not in results:
        print("Error: Results missing data required for visualization")
        return
    
    valid_voxel_dict = results['valid_voxel_dict']
    points = results['points']
    voxel_size = results['voxel_size']
    
    if len(valid_voxel_dict) == 0:
        print("Error: No valid voxels to visualize")
        return
    
    print(f"\nPreparing to visualize {len(valid_voxel_dict)} valid voxels...")
    
    # Collect all points in valid voxels
    all_valid_points = []
    all_valid_colors = []
    
    # Generate a unique color for each voxel
    # Use HSV color space, generate different hues through voxel index
    voxel_keys_list = list(valid_voxel_dict.keys())
    
    for idx, (voxel_key, point_indices) in enumerate(valid_voxel_dict.items()):
        # Use voxel index to generate HSV color (hue cycles between 0-1)
        # Use hash to ensure adjacent voxels have different colors
        hue = (hash(voxel_key) % 360) / 360.0  # Hue: 0-1
        saturation = 0.8 + (hash(voxel_key) % 20) / 100.0  # Saturation: 0.8-1.0
        value = 0.7 + (hash(voxel_key) % 30) / 100.0  # Value: 0.7-1.0
        
        # Convert to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Get all points in this voxel
        voxel_points = points[point_indices]
        all_valid_points.append(voxel_points)
        
        # Assign colors to these points
        num_points = len(point_indices)
        voxel_colors = np.tile(rgb, (num_points, 1))
        all_valid_colors.append(voxel_colors)
    
    # Merge all points and colors
    if len(all_valid_points) > 0:
        all_valid_points = np.vstack(all_valid_points)
        all_valid_colors = np.vstack(all_valid_colors)
        
        # Create Open3D point cloud
        pcd_visualization = o3d.geometry.PointCloud()
        pcd_visualization.points = o3d.utility.Vector3dVector(all_valid_points)
        pcd_visualization.colors = o3d.utility.Vector3dVector(all_valid_colors)
        
        print(f"Visualization point cloud contains {len(all_valid_points)} points")
        print(f"From {len(valid_voxel_dict)} valid voxels")
        print("\nOpening visualization window...")
        print("Note: Each voxel displayed in different color, adjacent voxels have different colors")
        
        # Visualize
        o3d.visualization.draw_geometries([pcd_visualization],
                                          window_name=f"Valid Voxel Visualization (Total {len(valid_voxel_dict)} voxels)",
                                          width=1920,
                                          height=1080)
    else:
        print("Error: No valid points to visualize")


def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("Geometric Accuracy Evaluation Results")
    print("="*60)
    print(f"Configuration parameters:")
    print(f"  Resolution: {results['resolution']:.4f} m")
    print(f"  Voxel size: {results['voxel_size']:.2f} m")
    print(f"\nVoxel statistics:")
    print(f"  Total voxels: {results['total_voxels']}")
    print(f"  Valid voxels: {results['valid_voxels']}")
    print(f"  Filtered by insufficient points: {results['filtered_by_points']}")
    print(f"  Filtered by insufficient planarity: {results['filtered_by_planarity']}")
    print(f"  Filtered by excessive curvature: {results['filtered_by_curvature']}")
    print(f"  Filtered by insufficient thickness: {results['filtered_by_thickness']}")
    print(f"  Filtered by 3-sigma outliers: {results['filtered_by_sigma']}")
    print(f"  Minimum points per voxel: {results['min_points_per_voxel']}")
    print(f"  Planarity threshold: {results['planarity_threshold']}")
    print(f"  Curvature threshold: {results['curvature_threshold']}")
    print(f"  Minimum thickness threshold: {results['min_thickness_threshold']}")
    print(f"  Sigma multiplier: {results['sigma_multiplier']}")
    print(f"\nPlanarity statistics:")
    print(f"  Mean planarity: {results['mean_planarity']:.6f}")
    print(f"  Std dev: {results['std_planarity']:.6f}")
    print(f"  Min planarity: {results['min_planarity']:.6f}")
    print(f"  Max planarity: {results['max_planarity']:.6f}")
    print(f"\nCurvature statistics:")
    print(f"  Mean curvature: {results['mean_curvature']:.6f}")
    print(f"  Std dev: {results['std_curvature']:.6f}")
    print(f"  Min curvature: {results['min_curvature']:.6f}")
    print(f"  Max curvature: {results['max_curvature']:.6f}")
    print(f"\nPlane thickness statistics:")
    print(f"  Mean thickness: {results['mean_thickness']:.6f} m")
    print(f"  Std dev: {results['std_thickness']:.6f} m")
    print(f"  Min thickness: {results['min_thickness']:.6f} m")
    print(f"  Max thickness: {results['max_thickness']:.6f} m")
    print(f"\nGeometric accuracy statistics (thickness-resolution)/resolution:")
    print(f"  Mean geometric accuracy: {results['mean_geometric_accuracy']:.4f}")
    print(f"  Std dev: {results['std_geometric_accuracy']:.4f}")
    print(f"  Min geometric accuracy: {results['min_geometric_accuracy']:.4f}")
    print(f"  Max geometric accuracy: {results['max_geometric_accuracy']:.4f}")
    print("="*60)


def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = load_config(config_path)
    
    # Read point cloud paths from config file
    colormap_path = config['common'].get('colormap', 'output/colormap.pcd')
    map_path = config['common'].get('map', 'output/map.pcd')
    
    # Convert to absolute paths
    base_dir = Path(__file__).parent.parent
    colormap_path = base_dir / colormap_path
    map_path = base_dir / map_path
    
    # Prefer colormap.pcd, use map.pcd if it doesn't exist
    if colormap_path.exists():
        pcd_path = colormap_path
        print(f"Using colored point cloud: {pcd_path}")
    elif map_path.exists():
        pcd_path = map_path
        print(f"Using full point cloud: {pcd_path}")
    else:
        print(f"Error: Cannot find point cloud file")
        print(f"  Tried to find: {colormap_path}")
        print(f"  Tried to find: {map_path}")
        print("Please run utils/projection.py first to generate colored point cloud")
        return
    
    # Evaluate geometric accuracy
    results = evaluate_geometry(pcd_path, config)
    
    if results is not None:
        print_results(results)
        
        # Optional: Save results to file
        output_dir = Path(__file__).parent.parent / "output"
        results_path = output_dir / "geometry_evaluation_results.txt"
        
        # Create result copy, exclude visualization data (avoid saving large amounts of data)
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ['valid_voxel_dict', 'points']}
        
        with open(results_path, 'w') as f:
            f.write("Geometric Accuracy Evaluation Results\n")
            f.write("="*60 + "\n")
            for key, value in results_to_save.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        print(f"\nResults saved to: {results_path}")
        
        # Visualize valid voxels
        print("\n" + "="*60)
        visualize_valid_voxels(results)
    else:
        print("Evaluation failed")


if __name__ == "__main__":
    main()
