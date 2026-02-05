import numpy as np
import open3d as o3d
import yaml
from pathlib import Path
from collections import defaultdict


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


def evaluate_coverage(config):
    """
    Evaluate point cloud coverage
    
    Args:
        config: Configuration dictionary
    
    Returns:
        results: Dictionary containing evaluation results
    """
    # Read configuration parameters
    voxel_size = config['coverage']['voxel_size']
    colormap_path = config['common'].get('colormap', 'output/colormap.pcd')
    map_path = config['common'].get('map', 'output/map.pcd')
    
    # Convert to absolute paths
    base_dir = Path(__file__).parent.parent
    colormap_path = base_dir / colormap_path
    map_path = base_dir / map_path
    
    # Load point clouds
    print(f"Loading colored point cloud: {colormap_path}")
    pcd_colored = o3d.io.read_point_cloud(str(colormap_path))
    points_colored = np.asarray(pcd_colored.points)
    
    print(f"Loading full point cloud: {map_path}")
    pcd_map = o3d.io.read_point_cloud(str(map_path))
    points_map = np.asarray(pcd_map.points)
    
    if len(points_colored) == 0 or len(points_map) == 0:
        print("Error: Point cloud is empty")
        return None
    
    print(f"Colored point cloud points: {len(points_colored)}")
    print(f"Full point cloud points: {len(points_map)}")
    
    # Voxelize
    print(f"\nVoxelizing point clouds (voxel_size={voxel_size})...")
    voxel_dict_colored = voxelize_point_cloud(points_colored, voxel_size)
    voxel_dict_map = voxelize_point_cloud(points_map, voxel_size)
    
    colored_grids = len(voxel_dict_colored)
    map_grids = len(voxel_dict_map)
    
    print(f"Colored point cloud grids: {colored_grids}")
    print(f"Full point cloud grids: {map_grids}")
    
    # Calculate coverage
    # coverage = (colored_points/lidar_points) / (colored_grids / lidar_grids)
    point_ratio = len(points_colored) / len(points_map)
    grid_ratio = colored_grids / map_grids if map_grids > 0 else 0
    
    if grid_ratio > 0:
        coverage = point_ratio / grid_ratio
    else:
        coverage = 0.0
    
    print(f"\nCoverage calculation:")
    print(f"  Point ratio (colored/full): {point_ratio:.6f}")
    print(f"  Grid ratio (colored/full): {grid_ratio:.6f}")
    print(f"  Coverage: {coverage:.6f}")
    
    # Create visualization point clouds
    print("\nCreating visualization point clouds...")
    
    # Calculate average point cloud density to determine offset
    # Use statistics of nearest neighbor distances to estimate point cloud density
    if len(points_colored) > 0 and len(points_map) > 0:
        # Calculate point cloud range
        points_all = np.vstack([points_colored, points_map])
        point_spacing = np.mean([np.std(points_all[:, i]) for i in range(3)]) * 0.01
        # Offset set to 2-3 times point spacing, so green points display above
        offset_z = max(point_spacing * 2, 0.01)  # At least 0.01 meters
    else:
        offset_z = 0.01
    
    # Set colored point cloud to green, slightly offset upward for display
    pcd_colored_viz = o3d.geometry.PointCloud()
    points_colored_offset = points_colored.copy()
    points_colored_offset[:, 2] += offset_z  # Offset upward along z-axis
    pcd_colored_viz.points = o3d.utility.Vector3dVector(points_colored_offset)
    green_colors = np.zeros((len(points_colored), 3))
    green_colors[:, 1] = 1.0  # Green
    pcd_colored_viz.colors = o3d.utility.Vector3dVector(green_colors)
    
    # Set full point cloud to blue
    pcd_map_viz = o3d.geometry.PointCloud()
    pcd_map_viz.points = o3d.utility.Vector3dVector(points_map)
    blue_colors = np.zeros((len(points_map), 3))
    blue_colors[:, 2] = 1.0  # Blue
    pcd_map_viz.colors = o3d.utility.Vector3dVector(blue_colors)
    
    # Create grid map visualization (display center point of each grid)
    print("Creating grid map visualization...")
    grid_centers_colored = []
    grid_centers_map = []
    
    # Grid centers for colored point cloud
    for voxel_key in voxel_dict_colored.keys():
        # voxel_key is (x, y, z) index, convert to actual coordinates (grid center)
        center = np.array(voxel_key) * voxel_size + voxel_size / 2.0
        grid_centers_colored.append(center)
    
    # Grid centers for full point cloud
    for voxel_key in voxel_dict_map.keys():
        center = np.array(voxel_key) * voxel_size + voxel_size / 2.0
        grid_centers_map.append(center)
    
    if len(grid_centers_colored) > 0:
        grid_centers_colored = np.array(grid_centers_colored)
        pcd_grid_colored = o3d.geometry.PointCloud()
        pcd_grid_colored.points = o3d.utility.Vector3dVector(grid_centers_colored)
        green_grid_colors = np.zeros((len(grid_centers_colored), 3))
        green_grid_colors[:, 1] = 1.0  # Green
        pcd_grid_colored.colors = o3d.utility.Vector3dVector(green_grid_colors)
    else:
        pcd_grid_colored = o3d.geometry.PointCloud()
    
    if len(grid_centers_map) > 0:
        grid_centers_map = np.array(grid_centers_map)
        pcd_grid_map = o3d.geometry.PointCloud()
        pcd_grid_map.points = o3d.utility.Vector3dVector(grid_centers_map)
        blue_grid_colors = np.zeros((len(grid_centers_map), 3))
        blue_grid_colors[:, 2] = 1.0  # Blue
        pcd_grid_map.colors = o3d.utility.Vector3dVector(blue_grid_colors)
    else:
        pcd_grid_map = o3d.geometry.PointCloud()
    
    # Visualization
    print("\nVisualizing point clouds and grid map...")
    print(f"Note: Green point cloud has been offset upward by {offset_z*1000:.2f} mm to display above blue point cloud")
    
    # Use visualizer to set point size, make green points larger
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Coverage Visualization (Green=Colored Point Cloud, Blue=Full Point Cloud)",
                      width=1920, height=1080)
    
    # Add geometries
    vis.add_geometry(pcd_map_viz)  # Add blue point cloud first (bottom layer)
    vis.add_geometry(pcd_colored_viz)  # Then add green point cloud (top layer)
    if len(grid_centers_colored) > 0:
        vis.add_geometry(pcd_grid_colored)
    if len(grid_centers_map) > 0:
        vis.add_geometry(pcd_grid_map)
    
    # Set render options to make points larger and more visible
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Set point size (default is 1.0)
    
    # Run visualization
    vis.run()
    vis.destroy_window()
    
    # Save results
    results = {
        'colored_points': len(points_colored),
        'map_points': len(points_map),
        'colored_grids': colored_grids,
        'map_grids': map_grids,
        'point_ratio': point_ratio,
        'grid_ratio': grid_ratio,
        'coverage': coverage,
        'voxel_size': voxel_size
    }
    
    return results


def print_results(results):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("Coverage Evaluation Results")
    print("="*60)
    print(f"Configuration parameters:")
    print(f"  Voxel size (voxel_size): {results['voxel_size']:.2f} m")
    print(f"\nPoint cloud statistics:")
    print(f"  Colored point cloud points: {results['colored_points']}")
    print(f"  Full point cloud points: {results['map_points']}")
    print(f"  Point ratio (colored/full): {results['point_ratio']:.6f}")
    print(f"\nGrid statistics:")
    print(f"  Colored point cloud grids: {results['colored_grids']}")
    print(f"  Full point cloud grids: {results['map_grids']}")
    print(f"  Grid ratio (colored/full): {results['grid_ratio']:.6f}")
    print(f"\nCoverage:")
    print(f"  Coverage: {results['coverage']:.6f}")
    print("="*60)


def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = load_config(config_path)
    
    # Evaluate coverage
    results = evaluate_coverage(config)
    
    if results is not None:
        print_results(results)
        
        # Save results to file
        output_dir = Path(__file__).parent.parent / "output"
        results_path = output_dir / "coverage_evaluation_results.txt"
        with open(results_path, 'w') as f:
            f.write("Coverage Evaluation Results\n")
            f.write("="*60 + "\n")
            for key, value in results.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        print(f"\nResults saved to: {results_path}")
    else:
        print("Evaluation failed")


if __name__ == "__main__":
    main()
