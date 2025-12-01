import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Optional
from .config import Config
from .image_io import load_pngs


def create_flat_field_correction(
    white_images: np.ndarray,
    mask: Optional[np.ndarray] = None,
    smooth_sigma: float = 15.0
) -> np.ndarray:
    """
    Create flat-field correction maps from images of uniform white surface.
    
    Args:
        white_images: (H, W, K) images of flat white surface under K lights
        mask: Optional mask of valid region
        smooth_sigma: Gaussian smoothing sigma for correction map
        
    Returns:
        correction_maps: (H, W, K) multiplicative correction factors
    """
    H, W, K = white_images.shape
    
    # For each light, normalize by its mean intensity
    correction_maps = np.zeros_like(white_images, dtype=np.float32)
    
    for k in range(K):
        img = white_images[..., k].astype(np.float32)
        
        if mask is not None:
            # Use only masked region for normalization
            mean_intensity = img[mask > 0].mean()
        else:
            mean_intensity = img.mean()
        
        # Correction factor: target / observed
        # Where observed is bright -> correction < 1 (darken)
        # Where observed is dim -> correction > 1 (brighten)
        correction = mean_intensity / (img + 1e-6)
        
        # Smooth the correction map to avoid amplifying noise
        correction_smooth = cv2.GaussianBlur(
            correction, 
            (0, 0), 
            sigmaX=smooth_sigma, 
            sigmaY=smooth_sigma
        )
        
        correction_maps[..., k] = correction_smooth
    
    return correction_maps


def apply_flat_field_correction(
    images: np.ndarray,
    correction_maps: np.ndarray
) -> np.ndarray:
    """Apply flat-field correction to images."""
    return (images * correction_maps).astype(np.float32)


def estimate_brdf_from_sphere(
    sphere_images: np.ndarray,
    mask: np.ndarray,
    light_dirs: np.ndarray,
    sphere_center: Tuple[float, float],
    sphere_radius: float
) -> dict:
    """
    Estimate simplified BRDF parameters from images of a sphere.
    Uses a simple cosine + specular model.
    
    Args:
        sphere_images: (H, W, K) images of sphere
        mask: Sphere mask
        light_dirs: (K, 3) light directions
        sphere_center: (cx, cy) sphere center in pixels
        sphere_radius: Sphere radius in pixels
        
    Returns:
        brdf_params: Dict with 'kd' (diffuse) and 'ks' (specular) coefficients
    """
    H, W, K = sphere_images.shape
    cx, cy = sphere_center
    
    # Compute surface normals on sphere
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    dx = x - cx
    dy = y - cy
    
    # Points on sphere
    dz = np.sqrt(np.maximum(0, sphere_radius**2 - dx**2 - dy**2))
    
    # Normals (pointing outward)
    normals = np.stack([dx, dy, dz], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norms
    
    # Fit BRDF: I = kd * max(n·l, 0) + ks * max(n·h, 0)^alpha
    # Simplified: just estimate kd and ks per light
    
    kd_list = []
    ks_list = []
    
    valid = mask.astype(bool)
    
    for k in range(K):
        L = light_dirs[k]
        
        # Lambertian component
        cos_theta = np.maximum(0, np.dot(normals, L))
        
        # View direction (camera at origin looking down -Z)
        V = np.array([0, 0, -1])
        H = (L + V) / (np.linalg.norm(L + V) + 1e-8)
        cos_alpha = np.maximum(0, np.dot(normals, H))
        
        # Observed intensity
        I_obs = sphere_images[..., k][valid]
        cos_theta_valid = cos_theta[valid]
        cos_alpha_valid = cos_alpha[valid]
        
        # Fit I ≈ kd * cos_theta + ks * cos_alpha^32
        A = np.column_stack([cos_theta_valid, cos_alpha_valid**32])
        coeffs, _, _, _ = np.linalg.lstsq(A, I_obs, rcond=None)
        
        kd_list.append(coeffs[0])
        ks_list.append(coeffs[1])
    
    return {
        'kd': np.array(kd_list),
        'ks': np.array(ks_list),
        'alpha': 32  # Specular exponent (fixed for simplicity)
    }


def correct_brdf_effects(
    images: np.ndarray,
    normals: np.ndarray,
    light_dirs: np.ndarray,
    brdf_params: dict,
    target_brdf: str = 'lambertian'
) -> np.ndarray:
    """
    Correct images for BRDF effects to make surface appear Lambertian.
    
    Args:
        images: (H, W, K) intensity images
        normals: (H, W, 3) surface normals
        light_dirs: (K, 3) light directions
        brdf_params: BRDF parameters from estimate_brdf_from_sphere
        target_brdf: 'lambertian' to remove specular highlights
        
    Returns:
        corrected_images: (H, W, K) BRDF-corrected images
    """
    H, W, K = images.shape
    corrected = np.zeros_like(images)
    
    kd = brdf_params['kd']
    ks = brdf_params['ks']
    alpha = brdf_params['alpha']
    
    V = np.array([0, 0, -1])  # Camera view direction
    
    for k in range(K):
        L = light_dirs[k]
        
        # Compute expected intensity components
        cos_theta = np.maximum(0, np.sum(normals * L, axis=-1))
        
        H_dir = (L + V) / (np.linalg.norm(L + V) + 1e-8)
        cos_alpha = np.maximum(0, np.sum(normals * H_dir, axis=-1))
        
        # Observed ≈ kd * cos_theta + ks * cos_alpha^alpha
        I_diffuse = kd[k] * cos_theta
        I_specular = ks[k] * (cos_alpha ** alpha)
        I_predicted = I_diffuse + I_specular
        
        if target_brdf == 'lambertian':
            # Remove specular, keep only diffuse
            I_target = I_diffuse
        else:
            I_target = I_predicted
        
        # Correction factor
        correction = I_target / (I_predicted + 1e-6)
        correction = np.clip(correction, 0.5, 2.0)  # Avoid extreme corrections
        
        corrected[..., k] = images[..., k] * correction
    
    return corrected


def multi_height_calibration(
    images_by_height: dict,
    heights: List[float],
    reference_height: float = 100.0
) -> np.ndarray:
    """
    Create distance-dependent illumination correction from multiple heights.
    
    Args:
        images_by_height: Dict[height_mm: np.ndarray(H,W,K)]
        heights: List of heights in mm
        reference_height: Reference height for normalization
        
    Returns:
        distance_model: Function parameters for 1/r^2 correction
    """
    # For each light, fit intensity vs distance
    # I(h) = I_ref * (h_ref / h)^2
    
    K = list(images_by_height.values())[0].shape[-1]
    
    distance_params = []
    
    for k in range(K):
        intensities = []
        for h in heights:
            img = images_by_height[h]
            mean_intensity = img[..., k].mean()
            intensities.append(mean_intensity)
        
        intensities = np.array(intensities)
        heights_arr = np.array(heights)
        
        # Fit I = a / h^2
        # log(I) = log(a) - 2*log(h)
        log_I = np.log(intensities + 1e-8)
        log_h = np.log(heights_arr + 1e-8)
        
        A = np.column_stack([np.ones_like(log_h), log_h])
        coeffs = np.linalg.lstsq(A, log_I, rcond=None)[0]
        
        # a = exp(coeffs[0]), exponent = coeffs[1]
        distance_params.append({
            'a': np.exp(coeffs[0]),
            'exp': coeffs[1],
            'I_ref': intensities[heights.index(reference_height)]
        })
    
    return distance_params


def create_distance_correction_map(
    image_shape: Tuple[int, int],
    camera_center: Tuple[float, float],
    pixel_size: float,
    working_height: float,
    distance_params: List[dict]
) -> np.ndarray:
    """
    Create spatially-varying distance correction map.
    
    Args:
        image_shape: (H, W)
        camera_center: (cx, cy) camera center in pixels
        pixel_size: Pixel size in mm/pixel
        working_height: Working distance in mm
        distance_params: Parameters from multi_height_calibration
        
    Returns:
        correction_map: (H, W, K) distance-based correction
    """
    H, W = image_shape
    K = len(distance_params)
    
    y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    cx, cy = camera_center
    
    # Distance from center in mm
    dx_mm = (x - cx) * pixel_size
    dy_mm = (y - cy) * pixel_size
    
    # 3D distance from camera (assuming flat surface at working_height)
    distance = np.sqrt(dx_mm**2 + dy_mm**2 + working_height**2)
    
    correction_map = np.zeros((H, W, K), dtype=np.float32)
    
    for k in range(K):
        params = distance_params[k]
        
        # Expected intensity at this distance
        I_expected = params['a'] / (distance ** abs(params['exp']))
        
        # Normalize by reference
        correction = params['I_ref'] / (I_expected + 1e-6)
        
        correction_map[..., k] = correction
    
    return correction_map


# ============================================================================
# Complete Calibration Pipeline
# ============================================================================

def calibrate_illumination(
    white_napkin_path: str,
    heights: List[float] = [100, 70, 40],
    reference_height: float = 100.0,
    output_dir: str = "calibration"
) -> dict:
    """
    Complete illumination calibration pipeline.
    
    Args:
        white_napkin_path: Path pattern for white napkin images, e.g.
                          "calib/white_{height}mm/*.png"
        heights: List of heights in mm
        reference_height: Reference height for normalization
        output_dir: Where to save calibration data
        
    Returns:
        calibration_data: Dict with all calibration parameters
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ILLUMINATION CALIBRATION PIPELINE")
    print("=" * 60)
    
    # Load white napkin images at different heights
    images_by_height = {}
    
    for h in heights:
        path = white_napkin_path.format(height=h)
        print(f"\nLoading white napkin images at {h}mm: {path}")
        I, files = load_pngs(path)
        images_by_height[h] = I
        print(f"  Loaded {len(files)} images, shape: {I.shape}")
    
    # 1. Create flat-field correction from reference height
    print(f"\n[1/3] Creating flat-field correction from {reference_height}mm...")
    white_ref = images_by_height[reference_height]
    flat_field_correction = create_flat_field_correction(white_ref)
    
    # Visualize flat-field correction
    for k in range(flat_field_correction.shape[-1]):
        correction_vis = flat_field_correction[..., k]
        cv2.imwrite(
            str(Path(output_dir) / f"flat_field_light_{k}.png"),
            (correction_vis * 50).clip(0, 255).astype(np.uint8)
        )
    
    print("  ✓ Flat-field correction maps saved")
    
    # 2. Estimate distance falloff model
    print("\n[2/3] Estimating distance-dependent falloff...")
    distance_params = multi_height_calibration(
        images_by_height, 
        heights, 
        reference_height
    )
    
    for k, params in enumerate(distance_params):
        print(f"  Light {k}: I = {params['a']:.2f} / h^{abs(params['exp']):.2f}")
    
    # 3. Package calibration data
    print("\n[3/3] Packaging calibration data...")
    calibration_data = {
        'flat_field_correction': flat_field_correction,
        'distance_params': distance_params,
        'reference_height': reference_height,
        'calibration_heights': heights,
    }
    
    # Save calibration
    np.save(
        Path(output_dir) / "illumination_calibration.npy",
        calibration_data,
        allow_pickle=True
    )
    
    print(f"\n✓ Calibration complete! Saved to {output_dir}/")
    print("=" * 60)
    
    return calibration_data


def apply_illumination_calibration(
    images: np.ndarray,
    calibration_data: dict,
    working_height: float,
    camera_center: Optional[Tuple[float, float]] = None,
    pixel_size: float = 0.1  # mm/pixel, adjust for your camera
) -> np.ndarray:
    """
    Apply complete illumination correction to images.
    
    Args:
        images: (H, W, K) input images
        calibration_data: Output from calibrate_illumination()
        working_height: Current working distance in mm
        camera_center: (cx, cy), defaults to image center
        pixel_size: Pixel size in mm/pixel
        
    Returns:
        corrected_images: (H, W, K) corrected images
    """
    H, W, K = images.shape
    
    if camera_center is None:
        camera_center = (W // 2, H // 2)
    
    # 1. Apply flat-field correction
    flat_field = calibration_data['flat_field_correction']
    images_corrected = apply_flat_field_correction(images, flat_field)
    
    # 2. Apply distance-dependent correction if height differs
    ref_height = calibration_data['reference_height']
    if abs(working_height - ref_height) > 1.0:  # More than 1mm difference
        distance_map = create_distance_correction_map(
            (H, W),
            camera_center,
            pixel_size,
            working_height,
            calibration_data['distance_params']
        )
        images_corrected = images_corrected * distance_map
    
    return images_corrected
