import numpy as np
from .config import Config

### Core logic for computing a photometric stereo image.
### TODO: Acquire the exact LED positions in mm and the exact position and angle/tilt of the camera to get much more accurate photometric computations

def R_from_euler_xyz(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
    px, py, pz = np.deg2rad([pitch_deg, yaw_deg, roll_deg])
    Rx = np.array([[1,0,0],[0,np.cos(px),-np.sin(px)],[0,np.sin(px),np.cos(px)]], dtype=np.float32)
    Ry = np.array([[np.cos(py),0,np.sin(py)],[0,1,0],[-np.sin(py),0,np.cos(py)]], dtype=np.float32)
    Rz = np.array([[np.cos(pz),-np.sin(pz),0],[np.sin(pz),np.cos(pz),0],[0,0,1]], dtype=np.float32)
    return (Rx @ Ry @ Rz).astype(np.float32)

"""
The ridge on the rock now shows up more clearly which means the new light geometry is now geometrically consistent with how the real LEDs illuminate the surface. 
This is because the def effectively compensates for the 18° camera tilt and the slight camera-ring offset (need to get accurate numbers on the offset)
the solver is now seeing shading variation from the correct angles instead of interpreting those differences as “fake” surface concavity.
"""
def build_light_dirs_point(
    angles_deg=[0,60,120,180,240,300],
    r=0.02, h=0.02,
    cam_tilt_deg=(18.0,0.0,0.0),
    cam_offset_rig=(0.0, 0.0, 0.0),
    z_ref=0.10
) -> np.ndarray:
    # LED positions in rig frame (N,3)
    led_pos = np.array(
        [[r*np.cos(np.deg2rad(a)), r*np.sin(np.deg2rad(a)), h] for a in angles_deg],
        dtype=np.float32
    )  # (N,3)

    R = R_from_euler_xyz(*cam_tilt_deg)                          # (3,3) camera<-rig
    t = np.array(cam_offset_rig, dtype=np.float32).reshape(1,3)  # (1,3)

    # ref point in camera frame -> to rig frame
    X_cam = np.array([[0.0, 0.0, z_ref]], dtype=np.float32)      # (1,3)
    X_rig = (X_cam @ R) + t                                      # (1,3)

    # vectors rig: LED - X_rig  ->  camera frame, normalize
    v_rig = led_pos - X_rig                                      # (N,3)
    v_cam = (v_rig @ R.T).astype(np.float32)                     # (N,3)
    norms = np.linalg.norm(v_cam, axis=1, keepdims=True) + 1e-12
    v_cam /= norms                                               # (N,3) unit

    return v_cam

def build_light_dirs_tilted(
    angles_deg=[0,60,120,180,240,300],
    z_tilt=1.5, #TO-DO! # See if this applies an angle of the LED's light direction towards center or not
    cam_tilt_deg=(18.0, 0.0, 0.0)
) -> np.ndarray:
    # lights in rig frame
    L_rows = []
    for a in angles_deg:
        v = np.array([np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a)), z_tilt], dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        L_rows.append(v)                         # (3,)
    L_rig = np.stack(L_rows, axis=0).astype(np.float32)  # (N,3)

    # rotate to camera frame
    R = R_from_euler_xyz(*cam_tilt_deg)         # (3,3)
    L_cam = (L_rig @ R.T).astype(np.float32)    # (N,3)
    return L_cam

# This one works
def build_light_dirs(angles_deg: list = Config.LIGHT_ANGLES, z_tilt: float = Config.Z_TILT) -> np.ndarray:
    """Build light directions for a ring around the camera."""
    angles = np.deg2rad(angles_deg)
    xy = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    z = np.ones((len(angles), 1)) * z_tilt
    L = np.concatenate([xy, z], axis=1).astype(np.float32)
    L /= np.linalg.norm(L, axis=1, keepdims=True) + 1e-12
    return L

def solve_photometric_stereo(I: np.ndarray, L: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve for albedo and normals using photometric stereo."""
    H, W, K = I.shape
    if K != L.shape[0] or K != Config.NUM_IMAGES:
        # If we aren't using 6 images for Depth from Shade, then we wont compute a photometric stereo image
        raise ValueError(f"Expected {Config.NUM_IMAGES} images and lights, got {K} images and {L.shape[0]} lights")
    LT = L.T
    pinv = np.linalg.inv(LT @ L) @ LT
    I_reshaped = I.reshape(-1, K).T
    g = (pinv @ I_reshaped).T.reshape(H, W, 3)
    albedo = np.linalg.norm(g, axis=-1)
    n = np.zeros_like(g, dtype=np.float32)
    nz = albedo > 1e-8
    n[nz] = (g[nz] / albedo[nz, None]).astype(np.float32)
    m = mask.astype(bool)
    albedo[~m] = 0.0
    n[~m] = 0.0
    flip = n[..., 2] < 0
    n[flip] = -n[flip]
    return albedo.astype(np.float32), n.astype(np.float32)