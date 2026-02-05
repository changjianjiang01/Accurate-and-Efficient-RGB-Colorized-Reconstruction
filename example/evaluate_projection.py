import numpy as np
import yaml
from pathlib import Path
import sys
import cv2
import argparse
from tqdm import tqdm

# Add utils directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.projection import (
    load_config,
    load_poses,
    load_point_cloud,
    load_image,
    get_camera_intrinsic,
    project_points_to_image
)


def save_image_with_mask(image, valid_pixel_coords, output_dir, prefix, timestamp):
    """
    Save image with valid mask overlay
    
    Args:
        image: Original image (RGB format)
        valid_pixel_coords: Valid pixel coordinates (N, 2) [u, v]
        output_dir: Output directory
        prefix: Filename prefix ("highest" or "lowest")
        timestamp: Timestamp
    """
    if image is None or len(image.shape) < 2:
        print(f"    Warning: Invalid image format, skipping save")
        return
    
    height, width = image.shape[:2]
    
    # Create mask image
    mask_image = np.zeros((height, width), dtype=np.uint8)
    if len(valid_pixel_coords) > 0:
        mask_image[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]] = 255
    
    # Overlay mask on original image (red for valid pixels)
    overlay = image.copy()
    overlay[mask_image > 0] = [255, 0, 0]  # Red marker for valid pixels
    result_image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Save result
    mask_path = output_dir / f"{prefix}_valid_mask_{timestamp}.png"
    cv2.imwrite(str(mask_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print(f"    Saved mask image to: {mask_path}")


def save_photometric_error_images(original_image, projected_image, error_mask, output_dir, prefix, timestamp):
    """
    Save photometric error related images: original image, projected image, and error mask
    
    Args:
        original_image: Original image (RGB format)
        projected_image: Projected image (RGB format)
        error_mask: Error mask (bool array, True for valid pixels)
        output_dir: Output directory
        prefix: Filename prefix ("max_error" or "min_error")
        timestamp: Timestamp
    """
    if original_image is None or len(original_image.shape) < 2:
        print(f"    Warning: Invalid original image format, skipping save")
        return
    
    if projected_image is None or len(projected_image.shape) < 2:
        print(f"    Warning: Invalid projected image format, skipping save")
        return
    
    height, width = original_image.shape[:2]
    
    # Save original image
    original_path = output_dir / f"{prefix}_original_{timestamp}.png"
    cv2.imwrite(str(original_path), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    print(f"    Saved original image to: {original_path}")
    
    # Save projected image
    projected_path = output_dir / f"{prefix}_projected_{timestamp}.png"
    cv2.imwrite(str(projected_path), cv2.cvtColor(projected_image, cv2.COLOR_RGB2BGR))
    print(f"    Saved projected image to: {projected_path}")
    
    # Calculate L1 error for each pixel (for visualization)
    error_image = np.zeros((height, width, 3), dtype=np.float32)
    for c in range(3):
        diff = np.abs(projected_image[:, :, c].astype(float) - original_image[:, :, c].astype(float))
        error_image[:, :, c] = diff
    
    # Only show errors within valid mask
    error_mask_3d = np.stack([error_mask] * 3, axis=2)
    error_image[~error_mask_3d] = 0
    
    # Normalize to 0-255 range for visualization
    error_max = np.max(error_image)
    if error_max > 0:
        error_image_normalized = (error_image / error_max * 255).astype(np.uint8)
    else:
        error_image_normalized = error_image.astype(np.uint8)
    
    # Convert error image to heatmap (using OpenCV colormap)
    error_gray = cv2.cvtColor(error_image_normalized, cv2.COLOR_RGB2GRAY)
    error_heatmap = cv2.applyColorMap(error_gray, cv2.COLORMAP_JET)
    
    # Save error mask (heatmap)
    error_mask_path = output_dir / f"{prefix}_error_mask_{timestamp}.png"
    cv2.imwrite(str(error_mask_path), error_heatmap)
    print(f"    Saved error mask to: {error_mask_path}")


def calculate_valid_pixel_ratio(config, max_images=None):
    """
    Calculate valid pixel ratio rho_valid and photometric error e_pho
    
    Valid pixel ratio = Sum of pixels with point cloud projection across all images / Total pixels across all images
    Photometric error e_pho = (1 / (255NC)) * Σ_{i=1}^{N} Σ_{j=1}^{C} Σ_{k=1}^{R} (E_i(k) * ||I_{ij}(k) - Ĩ_{ij}(k)||_1) / M_i
    Texture quality metric ηtex = e_pho + (1 - ρvalid)
    
    Args:
        config: Configuration dictionary
    
    Returns:
        results: Dictionary containing the following fields:
            rho_valid: Valid pixel ratio
            e_pho: Photometric error
            eta_tex: Texture quality metric
            total_valid_pixels: Sum of valid pixels across all images
            total_pixels: Total pixels across all images
            num_images: Number of processed images
    """
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
        return None
    
    # If max_images is specified, only process first N images
    if max_images is not None and max_images > 0:
        common_timestamps = common_timestamps[:max_images]
        print(f"Limiting to first {max_images} images (for testing)")
    
    # Get image dimensions from config
    cam_width = projection_config.get('cam_width', 1280)
    cam_height = projection_config.get('cam_height', 1024)
    scale = projection_config.get('scale', 0.5)
    
    # Get projection window size (number of frames on each side)
    window_size = projection_config.get('projection_window_size', 10)
    
    # Calculate actual image dimensions (considering scale)
    actual_width = int(cam_width * scale)
    actual_height = int(cam_height * scale)
    
    print(f"Image dimensions: {actual_width} x {actual_height}")
    print(f"Projection window size: {window_size} frames on each side (total {2 * window_size + 1} point clouds)")
    
    # Load first image to get camera intrinsics
    first_image_path = image_dir / f"{common_timestamps[0]}.png"
    if not first_image_path.exists():
        print(f"Error: Cannot find first image: {first_image_path}")
        return None, None, None, 0
    
    first_image = load_image(first_image_path)
    K, dist_coeffs = get_camera_intrinsic(projection_config, first_image.shape)
    
    # Statistics for valid pixels and photometric error
    total_valid_pixels = 0
    num_images_processed = 0
    num_images_with_projection = 0
    total_pixels_all_images = 0  # Total pixels across all images (including those without projection)
    
    # Accumulator for photometric error calculation
    total_photometric_error = 0.0  # Accumulate photometric error for all images
    
    # Record valid pixel count for each image to find highest and lowest
    image_valid_pixel_counts = []  # [(timestamp, valid_pixel_count, total_points_projected, image_size, image, valid_pixel_coords), ...]
    
    # Record photometric error for each image to find maximum and minimum
    image_photometric_errors = []  # [(timestamp, photometric_error, original_image, projected_image, error_mask), ...]
    
    print(f"\nStarting to process {len(common_timestamps)} images...")
    print(f"For each image, will use {window_size} frames on each side for projection (total up to {2 * window_size + 1} frames)")
    
    # Iterate through each image
    for img_idx, image_timestamp in enumerate(tqdm(common_timestamps, desc="Processing images", ncols=80)):
        
        image_path = image_dir / f"{image_timestamp}.png"
        
        if not image_path.exists():
            continue
        
        try:
            # Load image
            image = load_image(image_path)
            height, width = image.shape[:2]
            total_pixels_all_images += width * height
            num_images_processed += 1
            
            # Get image pose
            T_camera = image_poses[image_timestamp]
            
            # For current image, collect valid pixels and color information from all point cloud projections
            all_valid_pixels_set = set()  # Use set for deduplication, count all valid pixels projected to this image
            total_points_projected = 0  # Count total points projected to this image
            
            # For generating projected image: record color for each pixel position (from images corresponding to point clouds in window)
            # Use dictionary storage: {(u, v): color}, if multiple points project to same pixel, take the last one
            pixel_color_map = {}  # {(u, v): [r, g, b]}, color values in range [0, 255]
            
            # Determine point cloud range to use (window_size frames on each side)
            start_idx = max(0, img_idx - window_size)
            end_idx = min(len(common_timestamps), img_idx + window_size + 1)
            
            # Iterate through point clouds in window, project to current image
            for pcd_idx in range(start_idx, end_idx):
                pcd_timestamp = common_timestamps[pcd_idx]
                pcd_path = pcd_dir / f"{pcd_timestamp}.pcd"
                
                if not pcd_path.exists():
                    continue
                
                try:
                    # Load point cloud
                    points_lidar = load_point_cloud(pcd_path)
                    T_lidar = lidar_poses[pcd_timestamp]
                    
                    # Project point cloud to current image
                    points_camera, pixel_coords, valid_mask = project_points_to_image(
                        points_lidar, T_lidar, T_camera, Rcl, Pcl, K, dist_coeffs
                    )
                    
                    # Filter valid pixels within image bounds
                    pixel_coords_int = pixel_coords.astype(int)
                    in_bounds = (pixel_coords_int[:, 0] >= 0) & (pixel_coords_int[:, 0] < width) & \
                               (pixel_coords_int[:, 1] >= 0) & (pixel_coords_int[:, 1] < height)
                    
                    # Final valid mask: in front of camera and within image bounds
                    final_valid_mask = valid_mask & in_bounds
                    
                    if np.sum(final_valid_mask) > 0:
                        # Get valid pixel coordinates
                        valid_pixel_coords = pixel_coords_int[final_valid_mask]
                        
                        # Record valid pixel coordinates (for rho_valid statistics)
                        for idx, (u, v) in enumerate(valid_pixel_coords):
                            # Add to set (automatic deduplication)
                            all_valid_pixels_set.add((u, v))
                        
                        total_points_projected += np.sum(final_valid_mask)
                        
                        # Get colors from image corresponding to timestamp (for generating projected image)
                        # Need to project point cloud back to source image to get colors from source image
                        source_image_path = image_dir / f"{pcd_timestamp}.png"
                        if source_image_path.exists():
                            try:
                                source_image = load_image(source_image_path)
                                source_height, source_width = source_image.shape[:2]
                                
                                # Get source image pose
                                T_camera_source = image_poses[pcd_timestamp]
                                
                                # Project point cloud back to source image to get pixel coordinates and colors
                                points_camera_source, pixel_coords_source, valid_mask_source = project_points_to_image(
                                    points_lidar, T_lidar, T_camera_source, Rcl, Pcl, K, dist_coeffs
                                )
                                
                                # Filter valid pixels within source image bounds
                                pixel_coords_source_int = pixel_coords_source.astype(int)
                                in_bounds_source = (pixel_coords_source_int[:, 0] >= 0) & (pixel_coords_source_int[:, 0] < source_width) & \
                                                  (pixel_coords_source_int[:, 1] >= 0) & (pixel_coords_source_int[:, 1] < source_height)
                                
                                # Final valid mask: in front of camera and within source image bounds
                                final_valid_mask_source = valid_mask_source & in_bounds_source
                                
                                # Get points valid in both images
                                valid_in_both = final_valid_mask & final_valid_mask_source
                                
                                if np.sum(valid_in_both) > 0:
                                    # Get pixel coordinates of points valid in both images
                                    source_pixel_coords = pixel_coords_source_int[valid_in_both]
                                    current_pixel_coords = pixel_coords_int[valid_in_both]
                                    
                                    # Get colors from source image, map to current image projection positions
                                    for (u_source, v_source), (u_current, v_current) in zip(source_pixel_coords, current_pixel_coords):
                                        if 0 <= v_source < source_height and 0 <= u_source < source_width and \
                                           0 <= v_current < height and 0 <= u_current < width:
                                            # Get color from source image
                                            color = source_image[v_source, u_source].copy()
                                            # Map to current image projection position
                                            pixel_color_map[(u_current, v_current)] = color
                            except Exception as e:
                                # If projection fails, use current image color as fallback
                                for u, v in valid_pixel_coords:
                                    if 0 <= v < height and 0 <= u < width:
                                        pixel_color_map[(u, v)] = image[v, u].copy()
                        else:
                            # If source image doesn't exist, use current image color
                            for u, v in valid_pixel_coords:
                                if 0 <= v < height and 0 <= u < width:
                                    pixel_color_map[(u, v)] = image[v, u].copy()
                
                except Exception as e:
                    # Single point cloud projection failure doesn't affect overall process
                    continue
            
            # Count valid pixels for current image
            unique_pixel_count = len(all_valid_pixels_set)
            
            # Calculate photometric error for current image (if there are valid pixels)
            image_photometric_error = 0.0
            if unique_pixel_count > 0:
                # Generate projected image: project point cloud colors onto image
                projected_image = np.zeros_like(image)  # Projected image, initialized to 0 (black)
                
                # Fill projected image with colors of valid pixels
                for (u, v), color in pixel_color_map.items():
                    projected_image[v, u] = color
                
                # Calculate valid mask (E_i)
                valid_mask_image = np.zeros((height, width), dtype=bool)
                for u, v in all_valid_pixels_set:
                    valid_mask_image[v, u] = True
                
                # M_i = ||E_i||_1 (number of valid pixels)
                M_i = unique_pixel_count
                
                if M_i > 0:
                    # Calculate L1 color difference: ||I_{ij}(k) - Ĩ_{ij}(k)||_1
                    # I_{ij}(k): projected image (projected_image)
                    # Ĩ_{ij}(k): reference image (image, original image)
                    # Only calculate pixels within valid mask
                    
                    # Calculate L1 difference for each channel
                    for c in range(3):  # RGB three channels
                        # Get L1 difference of valid pixels
                        diff = np.abs(projected_image[:, :, c].astype(float) - image[:, :, c].astype(float))
                        # Only sum pixels within valid mask
                        valid_diff_sum = np.sum(diff[valid_mask_image])
                        # Accumulate: E_i(k) * ||I_{ij}(k) - Ĩ_{ij}(k)||_1 / M_i
                        # Note: E_i(k) is 1 within mask, so just sum directly
                        # Formula: Σ_{k=1}^{R} (E_i(k) * ||I_{ij}(k) - Ĩ_{ij}(k)||_1) / M_i
                        # For each channel j, sum first then divide by M_i
                        image_photometric_error += valid_diff_sum / M_i
                    
                    # Record photometric error information (for finding max and min later)
                    image_photometric_errors.append((
                        image_timestamp,
                        image_photometric_error,
                        image.copy(),  # Original image
                        projected_image.copy(),  # Projected image
                        valid_mask_image.copy()  # Error mask (valid pixel mask)
                    ))
                
                # Accumulate to total photometric error
                total_photometric_error += image_photometric_error
            
            if unique_pixel_count > 0:
                # Convert set to numpy array for visualization
                valid_pixel_coords_array = np.array(list(all_valid_pixels_set))
                
                # Record information
                image_valid_pixel_counts.append((
                    image_timestamp, 
                    unique_pixel_count, 
                    total_points_projected,
                    width * height,
                    image.copy(),  # Save image for visualization
                    valid_pixel_coords_array  # Save valid pixel coordinates
                ))
                
                # Accumulate valid pixel count
                total_valid_pixels += unique_pixel_count
                num_images_with_projection += 1
                
                # Output detailed information every 50 images
                if (img_idx + 1) % 50 == 0:
                    num_pcds_used = end_idx - start_idx
                    print(f"  Image {img_idx+1} ({image_timestamp}):")
                    print(f"    Point clouds used: {num_pcds_used}, Points projected: {total_points_projected}, Valid pixels: {unique_pixel_count}")
            else:
                # Record images without projection
                image_valid_pixel_counts.append((
                    image_timestamp,
                    0,
                    0,
                    width * height,
                    image.copy(),
                    np.empty((0, 2), dtype=int)  # Empty valid pixel coordinates
                ))
            
        except Exception as e:
            print(f"Error processing image {image_timestamp}: {e}")
            continue
    
    if num_images_processed == 0:
        print("Error: No images processed successfully")
        return None
    
    # Use total pixels from all processed images
    total_pixels = total_pixels_all_images
    
    # Calculate valid pixel ratio
    rho_valid = total_valid_pixels / total_pixels if total_pixels > 0 else 0.0
    
    # Calculate photometric error e_pho
    # e_pho = (1 / (255NC)) * Σ_{i=1}^{N} Σ_{j=1}^{C} Σ_{k=1}^{R} (E_i(k) * ||I_{ij}(k) - Ĩ_{ij}(k)||_1) / M_i
    # Where:
    #   N: Number of images (num_images_processed)
    #   C: Number of channels (3, RGB)
    #   R: Image resolution (pixels per image)
    #   We have already calculated the inner sum: Σ_{j=1}^{C} Σ_{k=1}^{R} (E_i(k) * ||I_{ij}(k) - Ĩ_{ij}(k)||_1) / M_i
    #   Stored in total_photometric_error
    #   Now just need to divide by (255 * N * C)
    C = 3  # RGB three channels
    N = num_images_processed
    if N > 0:
        e_pho = total_photometric_error / (255.0 * N * C)
    else:
        e_pho = 0.0
    
    # Calculate ηtex = e_pho + (1 - ρvalid)
    eta_tex = e_pho + (1.0 - rho_valid)
    
    # Calculate average image dimensions
    avg_width = int(np.sqrt(total_pixels / num_images_processed * (actual_width / actual_height)))
    avg_height = int(np.sqrt(total_pixels / num_images_processed * (actual_height / actual_width)))
    
    # Find images with highest and lowest valid pixel counts
    image_valid_pixel_counts_sorted = sorted(image_valid_pixel_counts, key=lambda x: x[1], reverse=True)
    
    print(f"\nProjection evaluation calculation completed:")
    print(f"  Number of images processed: {num_images_processed}")
    print(f"  Number of images with projection: {num_images_with_projection}")
    print(f"  Average image dimensions: {avg_width} x {avg_height} (based on actual images)")
    print(f"  Total pixels (all images): {total_pixels}")
    print(f"  Valid pixels (sum of valid pixels across all images): {total_valid_pixels}")
    print(f"  Valid pixel ratio rho_valid: {rho_valid:.6f}")
    print(f"  Photometric error e_pho: {e_pho:.6f}")
    print(f"  Texture quality metric ηtex = e_pho + (1 - ρvalid): {eta_tex:.6f}")
    
    # Output detailed statistics
    if len(image_valid_pixel_counts) > 0:
        valid_counts = [x[1] for x in image_valid_pixel_counts if x[1] > 0]
        if len(valid_counts) > 0:
            print(f"\nDetailed statistics:")
            print(f"  Valid pixel count statistics:")
            print(f"    Maximum: {max(valid_counts)}")
            print(f"    Minimum: {min(valid_counts)}")
            print(f"    Mean: {np.mean(valid_counts):.1f}")
            print(f"    Median: {np.median(valid_counts):.1f}")
            print(f"    Std dev: {np.std(valid_counts):.1f}")
            
            # Statistics for point cloud projection
            total_points_projected_list = [x[2] for x in image_valid_pixel_counts if x[2] > 0]
            
            print(f"\n  Point cloud projection statistics:")
            if len(total_points_projected_list) > 0:
                print(f"    Average points projected to image: {np.mean(total_points_projected_list):.1f}")
                print(f"    Maximum projection points: {max(total_points_projected_list):.0f}")
                print(f"    Minimum projection points: {min(total_points_projected_list):.0f}")
            
            # Find highest and lowest images
            print(f"\n  Image with highest valid pixel count:")
            top_image = image_valid_pixel_counts_sorted[0]
            print(f"    Timestamp: {top_image[0]}")
            print(f"    Valid pixel count: {top_image[1]}")
            print(f"    Total points projected to image: {top_image[2]}")
            image_size = top_image[3]
            # Calculate image dimensions (assuming rectangular)
            height = int(np.sqrt(image_size * (actual_height / actual_width)))
            width = int(np.sqrt(image_size * (actual_width / actual_height)))
            print(f"    Image dimensions: {width} x {height}")
            
            # Find image with lowest valid pixel count (excluding 0)
            bottom_image = None
            for img_info in reversed(image_valid_pixel_counts_sorted):
                if img_info[1] > 0:
                    bottom_image = img_info
                    break
            
            if bottom_image:
                print(f"\n  Image with lowest valid pixel count (with projection):")
                print(f"    Timestamp: {bottom_image[0]}")
                print(f"    Valid pixel count: {bottom_image[1]}")
                print(f"    Total points projected to image: {bottom_image[2]}")
                image_size = bottom_image[3]
                height = int(np.sqrt(image_size * (actual_height / actual_width)))
                width = int(np.sqrt(image_size * (actual_width / actual_height)))
                print(f"    Image dimensions: {width} x {height}")
            
            # Save valid masks for highest and lowest images
            output_dir = Path(__file__).parent.parent / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Save mask for image with highest valid pixel count
            if top_image[1] > 0 and len(top_image) > 5:
                if len(top_image[5]) > 0:
                    save_image_with_mask(top_image[4], top_image[5], output_dir, "highest", top_image[0])
            
            # Save mask for image with lowest valid pixel count
            if bottom_image and bottom_image[1] > 0 and len(bottom_image) > 5:
                if len(bottom_image[5]) > 0:
                    save_image_with_mask(bottom_image[4], bottom_image[5], output_dir, "lowest", bottom_image[0])
            
            # Find images with maximum and minimum photometric error
            if len(image_photometric_errors) > 0:
                # Sort by photometric error
                image_photometric_errors_sorted = sorted(image_photometric_errors, key=lambda x: x[1], reverse=True)
                
                # Maximum photometric error
                max_error_image = image_photometric_errors_sorted[0]
                print(f"\n  Image with maximum photometric error:")
                print(f"    Timestamp: {max_error_image[0]}")
                print(f"    Photometric error: {max_error_image[1]:.6f}")
                save_photometric_error_images(
                    max_error_image[2],  # Original image
                    max_error_image[3],  # Projected image
                    max_error_image[4],  # Error mask
                    output_dir,
                    "max_error",
                    max_error_image[0]
                )
                
                # Minimum photometric error
                min_error_image = image_photometric_errors_sorted[-1]
                print(f"\n  Image with minimum photometric error:")
                print(f"    Timestamp: {min_error_image[0]}")
                print(f"    Photometric error: {min_error_image[1]:.6f}")
                save_photometric_error_images(
                    min_error_image[2],  # Original image
                    min_error_image[3],  # Projected image
                    min_error_image[4],  # Error mask
                    output_dir,
                    "min_error",
                    min_error_image[0]
                )
    
    return {
        'rho_valid': rho_valid,
        'e_pho': e_pho,
        'eta_tex': eta_tex,
        'total_valid_pixels': total_valid_pixels,
        'total_pixels': total_pixels,
        'num_images': num_images_processed
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate projection quality: calculate valid pixel ratio and photometric error')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Limit number of images to process (for testing, e.g., --max-images 50)')
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    config = load_config(config_path)
    
    # If max_images is specified in command line, override config file setting
    max_images = args.max_images
    
    # Calculate valid pixel ratio
    print("="*60)
    print("Calculate valid pixel ratio rho_valid and photometric error e_pho")
    print("="*60)
    
    results = calculate_valid_pixel_ratio(config, max_images=max_images)
    
    if results is not None and 'rho_valid' in results:
        # Save results to file
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        results_path = output_dir / "projection_evaluation_results.txt"
        
        with open(results_path, 'w') as f:
            f.write("Projection Evaluation Results\n")
            f.write("="*60 + "\n")
            f.write(f"Valid pixel ratio (rho_valid): {results['rho_valid']:.6f}\n")
            f.write(f"Photometric error (e_pho): {results['e_pho']:.6f}\n")
            f.write(f"Texture quality metric (eta_tex): {results['eta_tex']:.6f}\n")
            f.write(f"Valid pixels (sum across all images): {results['total_valid_pixels']}\n")
            f.write(f"Total pixels (all images): {results['total_pixels']}\n")
            f.write(f"Number of images processed: {results['num_images']}\n")
        
        print(f"\nResults saved to: {results_path}")
    else:
        print("Calculation failed")


if __name__ == "__main__":
    main()
