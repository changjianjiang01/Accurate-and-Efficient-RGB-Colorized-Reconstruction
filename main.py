#!/usr/bin/env python3
"""
Main evaluation script for RISED project.
Evaluates three metrics: Coverage, Average Geometric Accuracy, and eta_tex.
"""

import sys
import os
from pathlib import Path
import yaml
import numpy as np
from contextlib import redirect_stdout
from io import StringIO
from tqdm import tqdm

# Add example directory to path
sys.path.insert(0, str(Path(__file__).parent / "example"))

from evaluate_coverage import evaluate_coverage, load_config as load_config_coverage
from evaluate_geometry import evaluate_geometry, load_config as load_config_geometry
from evaluate_projection import calculate_valid_pixel_ratio, load_config as load_config_projection


def load_config():
    """Load configuration file"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_coverage_silent(config):
    """Evaluate coverage without visualization"""
    import open3d as o3d
    from collections import defaultdict
    
    # Read config parameters
    voxel_size = config['coverage']['voxel_size']
    colormap_path = config['common'].get('colormap', 'output/colormap.pcd')
    map_path = config['common'].get('map', 'output/map.pcd')
    
    # Convert to absolute paths
    base_dir = Path(__file__).parent
    colormap_path = base_dir / colormap_path
    map_path = base_dir / map_path
    
    # Load point clouds
    pcd_colored = o3d.io.read_point_cloud(str(colormap_path))
    points_colored = np.asarray(pcd_colored.points)
    pcd_map = o3d.io.read_point_cloud(str(map_path))
    points_map = np.asarray(pcd_map.points)
    
    if len(points_colored) == 0 or len(points_map) == 0:
        return None
    
    # Voxelize
    def voxelize_point_cloud(points, voxel_size):
        voxel_indices = np.floor(points / voxel_size).astype(int)
        voxel_dict = defaultdict(list)
        for i, idx in enumerate(voxel_indices):
            voxel_key = tuple(idx)
            voxel_dict[voxel_key].append(i)
        return voxel_dict
    
    voxel_dict_colored = voxelize_point_cloud(points_colored, voxel_size)
    voxel_dict_map = voxelize_point_cloud(points_map, voxel_size)
    
    colored_grids = len(voxel_dict_colored)
    map_grids = len(voxel_dict_map)
    
    # Calculate coverage
    point_ratio = len(points_colored) / len(points_map)
    grid_ratio = colored_grids / map_grids if map_grids > 0 else 0
    
    if grid_ratio > 0:
        coverage = point_ratio / grid_ratio
    else:
        coverage = 0.0
    
    return {'coverage': coverage}


def evaluate_all():
    """Run all three evaluation methods and output final metrics"""
    
    print("="*60)
    print("RISED Evaluation")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    results = {}
    
    # 1. Evaluate Coverage
    print("\n[1/3] Evaluating Coverage...")
    try:
        with tqdm(total=100, desc="Coverage", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
            coverage_results = evaluate_coverage_silent(config)
            pbar.update(100)
        if coverage_results is not None:
            results['coverage'] = coverage_results['coverage']
            print(f"Coverage: {results['coverage']:.6f}")
        else:
            results['coverage'] = None
            print("Coverage evaluation failed")
    except Exception as e:
        print(f"Coverage evaluation error: {e}")
        results['coverage'] = None
    
    # 2. Evaluate Geometric Accuracy
    print("\n[2/3] Evaluating Geometric Accuracy...")
    try:
        # Get point cloud path from config
        base_dir = Path(__file__).parent
        colormap_path = base_dir / config['common'].get('colormap', 'output/colormap.pcd')
        map_path = base_dir / config['common'].get('map', 'output/map.pcd')
        
        # Prefer colormap.pcd, fallback to map.pcd
        if colormap_path.exists():
            pcd_path = colormap_path
        elif map_path.exists():
            pcd_path = map_path
        else:
            print(f"Error: Point cloud files not found")
            results['avg_geometric_accuracy'] = None
            pcd_path = None
        
        if pcd_path is not None:
            # Suppress print output during evaluation
            f = StringIO()
            with redirect_stdout(f):
                with tqdm(total=100, desc="Geometric Accuracy", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as pbar:
                    geometry_results = evaluate_geometry(pcd_path, config)
                    pbar.update(100)
            if geometry_results is not None:
                results['avg_geometric_accuracy'] = geometry_results['mean_geometric_accuracy']
                print(f"Average Geometric Accuracy: {results['avg_geometric_accuracy']:.6f}")
            else:
                results['avg_geometric_accuracy'] = None
                print("Geometric accuracy evaluation failed")
    except Exception as e:
        print(f"Geometric accuracy evaluation error: {e}")
        results['avg_geometric_accuracy'] = None
    
    # 3. Evaluate Projection Quality
    print("\n[3/3] Evaluating Projection Quality...")
    try:
        # Note: tqdm writes to stderr, so redirecting stdout won't affect it
        # We suppress stdout to reduce verbose output, but tqdm progress bars will still show
        f = StringIO()
        with redirect_stdout(f):
            projection_results = calculate_valid_pixel_ratio(config, max_images=None)
        if projection_results is not None:
            results['eta_tex'] = projection_results['eta_tex']
            results['rho_valid'] = projection_results['rho_valid']
            results['e_pho'] = projection_results['e_pho']
            print(f"rho_valid: {results['rho_valid']:.6f}")
            print(f"e_pho: {results['e_pho']:.6f}")
            print(f"eta_tex: {results['eta_tex']:.6f}")
        else:
            results['eta_tex'] = None
            results['rho_valid'] = None
            results['e_pho'] = None
            print("Projection quality evaluation failed")
    except Exception as e:
        print(f"Projection quality evaluation error: {e}")
        results['eta_tex'] = None
        results['rho_valid'] = None
        results['e_pho'] = None
    
    # Output final results
    print("\n" + "="*60)
    print("Final Evaluation Results")
    print("="*60)
    
    if results['coverage'] is not None:
        print(f"Coverage: {results['coverage']:.6f}")
    else:
        print("Coverage: N/A")
    
    if results['avg_geometric_accuracy'] is not None:
        print(f"Average Geometric Accuracy: {results['avg_geometric_accuracy']:.6f}")
    else:
        print("Average Geometric Accuracy: N/A")
    
    if results.get('rho_valid') is not None:
        print(f"rho_valid: {results['rho_valid']:.6f}")
    else:
        print("rho_valid: N/A")
    
    if results.get('e_pho') is not None:
        print(f"e_pho: {results['e_pho']:.6f}")
    else:
        print("e_pho: N/A")
    
    if results.get('eta_tex') is not None:
        print(f"eta_tex: {results['eta_tex']:.6f}")
    else:
        print("eta_tex: N/A")
    
    print("="*60)
    
    return results


if __name__ == "__main__":
    evaluate_all()
