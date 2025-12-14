import os
import numpy as np
from glob import glob
import threading
import matplotlib.pyplot as plt
import tifffile
from scipy.stats import skew
import argparse
from pathlib import Path
import pandas as pd
import sys

# Ignore userwarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Try to import data generator, adjust path if necessary
try:
    from Model.aSN2N_datagen_sliding_aug import data_generator
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Model.aSN2N_datagen_sliding_aug import data_generator

# region Helper Functions

def _extract_patches(image, patch_size, stride):
    """
    Extract patches from a single image.
    Returns a dictionary: {'coords': [(y,x)...], 'patches': [patch_array...]}
    """
    patches = {'coords': [], 'patches': []}

    img_h, img_w = image.shape
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride

    for y in range(0, img_h - patch_h + 1, stride_h):
        for x in range(0, img_w - patch_w + 1, stride_w):
            patch = image[y:y+patch_h, x:x+patch_w]
            patches['coords'].append((y, x))
            patches['patches'].append(patch)
    return patches

def _calculate_patch_metrics(patch):
    """
    Calculate statistical metrics for a single patch (Mean, Std, Skewness).
    """
    if patch.size == 0:
        return {'mean': 0.0, 'std': 0.0, 'skewness': 0.0}
    
    patch_float = patch.astype(np.float32)

    mu_i = np.mean(patch_float)
    sigma_i = np.std(patch_float)
    
    # Skewness requires at least 3 points and non-zero standard deviation
    if patch_float.size > 2 and sigma_i > 1e-6: 
        s_i = skew(patch_float.flatten(), bias=False)
    else:
        s_i = 0.0 
    
    return {
        'mean': mu_i,
        'std': sigma_i,
        'skewness': s_i,
    }

def _assess_patch_risk(patch_metrics, thresholds):
    """
    Assess patch risk level based on threshold rules.
    """
    mu_i = patch_metrics['mean']
    sigma_i = patch_metrics['std']
    s_i = patch_metrics['skewness']

    risk = 'LOW'  # Default to low risk

    # --- Rule 1: Detect low signal, low contrast background patches ---
    # Logic: (Low Mean AND Low Std) OR (Low Skewness AND Very Low Std)
    r1_mu = thresholds.get("R1_mu", 0.8)
    r1_sigma = thresholds.get("R1_sigma", 0.8)
    r1_skew = thresholds.get("R1_skewness", 0.2)
    r1_sigma_secondary = thresholds.get("R1_sigma_secondary", 0.05)

    condition_mu = mu_i <= r1_mu
    condition_sigma = sigma_i <= r1_sigma
    condition_skewness = s_i <= r1_skew

    if (condition_mu and condition_sigma) or (condition_skewness and sigma_i <= r1_sigma_secondary):
        return 'HIGH_BACKGROUND'

    # --- Rule 2: Detect sparse, high signal patches (Low Mean, High Contrast, High Skewness) ---
    # Logic: Low Mean AND High Std AND High Skewness
    r2_mu = thresholds.get("R2_mu", 0.7)
    r2_sigma = thresholds.get("R2_sigma", 1.5)
    r2_skew = thresholds.get("R2_skewness", 1.5)
    
    condition_mu_2 = mu_i <= r2_mu
    condition_sigma_2 = sigma_i >= r2_sigma
    condition_skewness_2 = s_i >= r2_skew

    if condition_mu_2 and condition_sigma_2 and condition_skewness_2:
        return 'HIGH_SPARSE'
    
    return risk

# endregion

# region Core Logic

def get_adaptive_normalization_mode(
    image_folder_path, 
    patch_size, 
    stride, 
    thresholds, 
    high_risk_patch_ratio_threshold_per_image,
    global_priority_decision_logic='any',
    output_dir=None,
    visualize_patches=False,
    visualize_overlay=False,
    export_csv=False
):
    """
    Analyze image folder to determine the normalization strategy ('global' or 'local') for the entire dataset.
    Supports visualization and data export.
    """
    
    # Determine image type (for output folder naming, optional)
    current_type = "Unknown"
    for t in ["CCP", "ER", "F_actin", "MT", "Microtubules"]:
        if t in image_folder_path:
            current_type = t if t != "Microtubules" else "MT"
            break
            
    img_files = glob(os.path.join(image_folder_path, '*.tif')) + \
                glob(os.path.join(image_folder_path, '*.tiff'))
    
    if not img_files:
        print(f"Warning: No TIFF images found in {image_folder_path}. Defaulting to 'local'.")
        return 'local'

    print(f"Starting analysis of {len(img_files)} images in {image_folder_path}...")

    # Set output directory
    if output_dir is None:
        output_dir = Path("debug_output") / current_type
    else:
        output_dir = Path(output_dir) / current_type
    
    if visualize_patches or visualize_overlay or export_csv:
        output_dir.mkdir(parents=True, exist_ok=True)

    relative_params_list = []
    num_images_suggesting_global = 0
    image_decisions = []

    # Global statistical variables
    grand_total_patches = 0
    grand_total_high_risk = 0

    for i, img_path in enumerate(img_files):
        # print(f"  Processing image {i+1}/{len(img_files)}: {os.path.basename(img_path)}")
        
        try:
            image = tifffile.imread(img_path)
            # Handle multi-dimensional images, ensure 2D
            if image.ndim == 3 and image.shape[0] == 1: image = image.squeeze(0)
            elif image.ndim > 2:
                image = image[0] 
            
            if image.ndim != 2:
                continue
            
            image_float = image.astype(np.float64)

            # Preprocessing: Remove extremely bright noise (99.99% truncation) and normalize to [0, 1]
            upper_percentile = np.percentile(image_float, 99.99)
            image_float = np.clip(image_float, None, upper_percentile)
            image_float = (image_float - np.min(image_float)) / (np.max(image_float) - np.min(image_float) + 1e-6)

        except Exception as e:
            print(f"  Error processing {os.path.basename(img_path)}: {e}")
            continue

        patches = _extract_patches(image_float, patch_size, stride)
        
        if not patches['patches']:
            continue

        high_risk_bg_count = 0
        high_risk_sparse_count = 0
        
        high_bg_coords = []
        high_sparse_coords = []
        normal_coords = []

        # Iterate through patches
        for patch_idx, patch in enumerate(patches['patches']):
            metrics = _calculate_patch_metrics(patch)
            
            # Collect data for export
            if export_csv:
                relative_params_list.append([
                    os.path.basename(img_path),
                    patch_idx,
                    metrics['mean'],
                    metrics['std'],
                    metrics['skewness']
                ])

            # Risk assessment
            risk = _assess_patch_risk(metrics, thresholds)

            y, x = patches['coords'][patch_idx]

            if risk == 'HIGH_BACKGROUND':
                high_risk_bg_count += 1
                high_bg_coords.append((x, y))
            elif risk == 'HIGH_SPARSE':
                high_risk_sparse_count += 1
                high_sparse_coords.append((x, y))
            else:
                normal_coords.append((x, y))

            # Visualize single patch
            if visualize_patches:
                patch_save_dir = output_dir / "patches" / os.path.basename(img_path)[:-4]
                patch_save_dir.mkdir(parents=True, exist_ok=True)
                
                # Limit saved quantity to avoid too many files
                if patch_idx < 50:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(patch, cmap='gray', vmin=0, vmax=1)
                    ax.set_title(f"Risk: {risk}\nMean:{metrics['mean']:.2f}, Std:{metrics['std']:.2f}")
                    ax.axis('off')
                    fig.savefig(patch_save_dir / f"patch_{patch_idx:04d}.png", dpi=100, bbox_inches='tight')
                    plt.close(fig)

        # Visualize full image (overlay on base image, not just boxes)
        if visualize_overlay:
            overlay_save_dir = output_dir / "overlay"
            overlay_save_dir.mkdir(parents=True, exist_ok=True)
            
            img_h, img_w = image_float.shape
            fig, ax = plt.subplots(figsize=(8, 8))
            plt.imshow(image_float, cmap='gray', vmin=0, vmax=1)

            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)
            ax.set_aspect('equal')
            
            # Draw rectangles
            for (x, y) in high_bg_coords:
                ax.add_patch(plt.Rectangle((x, y), patch_size[1], patch_size[0], linewidth=2, edgecolor='red', facecolor='none'))
            for (x, y) in high_sparse_coords:
                ax.add_patch(plt.Rectangle((x, y), patch_size[1], patch_size[0], linewidth=2, edgecolor='yellow', facecolor='none'))
            for (x, y) in normal_coords:
                ax.add_patch(plt.Rectangle((x, y), patch_size[1], patch_size[0], linewidth=1, edgecolor='green', facecolor='none', alpha=0.3))
            
            ax.axis('off')
            plt.savefig(overlay_save_dir / f"{os.path.basename(img_path)}_overlay.png", dpi=200, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close(fig)

        # Statistical decision (per image)
        total_patches_img = len(patches['patches'])
        high_risk_patches_img = high_risk_bg_count + high_risk_sparse_count
        ratio_high_risk = high_risk_patches_img / total_patches_img if total_patches_img > 0 else 0
        
        current_decision = 'local'
        if ratio_high_risk > high_risk_patch_ratio_threshold_per_image:
            current_decision = 'global'
            num_images_suggesting_global += 1
        
        image_decisions.append(current_decision)
        
        # Update global statistics
        grand_total_patches += total_patches_img
        grand_total_high_risk += high_risk_patches_img
        
        # print(f"    Image: {os.path.basename(img_path)} -> High Risk Ratio: {ratio_high_risk:.2%} ({current_decision})")

    # Export CSV
    if export_csv and relative_params_list:
        df = pd.DataFrame(relative_params_list, columns=['Image', 'Patch_Index', 'Mean', 'Std', 'Skewness'])
        df.to_csv(output_dir / f"{current_type}_patch_metrics.csv", index=False)
        print(f"Metrics exported to {output_dir}")

    # Final decision logic
    if grand_total_patches == 0:
        return 'local'
        
    grand_ratio = grand_total_high_risk / grand_total_patches
    print(f"Overall Statistics: {grand_total_high_risk}/{grand_total_patches} high-risk patches ({grand_ratio:.2%})")
    print(f"Threshold: {high_risk_patch_ratio_threshold_per_image:.2%}")
    
    final_decision = 'local'
    if grand_ratio > high_risk_patch_ratio_threshold_per_image:
        final_decision = 'global'
    
    print(f"Decision Summary: Overall High-Risk Ratio {grand_ratio:.2%} -> {final_decision.upper()}")
    return final_decision

# endregion


# region Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adaptive SN2N Data Generation Script')
    
    # Path arguments
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to the training images folder')
    parser.add_argument('--output_base_path', type=str, default='./output', help='Base path for output data')
    
    # Feature switches
    parser.add_argument('--vis_patches', action='store_true', help='Enable visualization of individual patches')
    parser.add_argument('--vis_overlay', action='store_true', help='Enable visualization of risk overlay on images')
    parser.add_argument('--export_csv', action='store_true', help='Export patch metrics to CSV')
    
    # Generation configuration
    parser.add_argument('--both_modes', action='store_true', help='Generate BOTH global and local datasets regardless of decision')
    
    args = parser.parse_args()

    # Configuration parameters
    patch_size_config = (64, 64)
    stride_config = (64, 64)
    
    # Threshold configuration (Updated from test file)
    thresholds_config = {
        "R1_mu": 0.2,
        "R1_sigma": 0.028,
        "R1_skewness": 0.0,
        "R1_sigma_secondary": 0.05,
        "R2_mu": 0.3,
        "R2_sigma": 0.075,
        "R2_skewness": 3.0,
    }
    
    high_risk_ratio_config = 0.10
    decision_logic_config = 'any'

    # 2. Normal data generation mode
    print(f"\nAnalyzing dataset at: {args.train_data_path}")
    
    adaptive_mode = get_adaptive_normalization_mode(
        args.train_data_path,
        patch_size_config,
        stride_config,
        thresholds_config,
        high_risk_ratio_config,
        decision_logic_config,
        output_dir=os.path.join(args.output_base_path, "debug"),
        visualize_patches=args.vis_patches,
        visualize_overlay=args.vis_overlay,
        export_csv=args.export_csv
    )
    
    print(f"\n>>> Adaptive Mode Determined: {adaptive_mode.upper()} <<<\n")

    # Prepare paths for generated data
    global_save_path = os.path.join(args.output_base_path, "global")
    local_save_path = os.path.join(args.output_base_path, "local")

    # Define generation task
    def run_generator(mode, save_path):
        print(f"Starting {mode} generation...")
        # Note: This calls the data_generator class from Model
        # Assuming the class accepts img_path, save_path, gen_mode etc.
        d = data_generator(
            img_path=args.train_data_path, 
            save_path=save_path,
            augment_mode=0, 
            pre_augment_mode=0, 
            img_res=patch_size_config, 
            ifx2=True, 
            inter_method='Fourier', 
            sliding_interval=32, # This might need adjustment based on stride_config, or keep as is
            gen_mode=mode
        )
        d.savedata4folder_agument(times=1, roll=1, threshold_mode=1, threshold=-1)
        print(f"Finished {mode} generation.")

    threads = []
    
    # Start threads based on decision
    if args.both_modes or adaptive_mode == 'global':
        os.makedirs(global_save_path, exist_ok=True)
        t = threading.Thread(target=run_generator, args=('global', global_save_path))
        threads.append(t)
        
    if args.both_modes or adaptive_mode == 'local':
        os.makedirs(local_save_path, exist_ok=True)
        t = threading.Thread(target=run_generator, args=('local', local_save_path))
        threads.append(t)

    if not threads:
        print("No generation tasks started.")
    else:
        print(f"Starting {len(threads)} generation threads...")
        for t in threads: t.start()
        for t in threads: t.join()
        print("All data generation tasks completed.")

# endregion
