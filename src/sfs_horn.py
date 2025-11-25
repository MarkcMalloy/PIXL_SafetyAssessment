from __future__ import annotations
import numpy as np


# -------------------------------------------------------------------------
# LED ring geometry: measured 6-LED ring around the camera
# -------------------------------------------------------------------------

def build_led_dirs_measured() -> np.ndarray:
    """
    Build light directions for the real LED ring.

    We model each LED as a directional source whose optical axis points
    inward toward the image center, tilted by the measured 'inward_tilt'
    with respect to the +z viewing axis.
    """

    LED_data_height_m = 0.008975 + 0.005  # not used for direction here

    led_data = [
        (320, 0.033675, LED_data_height_m, 23.03),  # G1
        (10,  0.033675, LED_data_height_m, 23.03),  # G2
        (60,  0.033675, LED_data_height_m, 23.03),  # G3
        (110, 0.033675, LED_data_height_m, 23.03),  # G4
        (160, 0.033675, LED_data_height_m, 23.03),  # G5
        (205, 0.033675, LED_data_height_m, 23.03),  # G6
    ]

    dirs = []
    for angle_deg, radius_m, height_m, inward_tilt_deg in led_data:
        phi = np.deg2rad(angle_deg)
        theta = np.deg2rad(inward_tilt_deg)

        # Unit vector pointing radially inward in the xy-plane
        ex = -np.cos(phi)
        ey = -np.sin(phi)

        # LED axis tilted by theta away from +z toward the center
        s = np.array(
            [np.sin(theta) * ex,
             np.sin(theta) * ey,
             np.cos(theta)],
            dtype=np.float32,
        )
        s /= np.linalg.norm(s)
        dirs.append(s)

    L = np.stack(dirs, axis=0)  # (6,3)
    return L


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def _laplacian(X: np.ndarray) -> np.ndarray:
    """5-point Laplacian with periodic wrap-around."""
    lap = -4.0 * X
    lap += np.roll(X, 1, axis=0)
    lap += np.roll(X, -1, axis=0)
    lap += np.roll(X, 1, axis=1)
    lap += np.roll(X, -1, axis=1)
    return lap


def integrate_slopes_poisson(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Integrate slope fields p = dz/dx and q = dz/dy into a height map z by
    solving the Poisson equation Δz = ∂p/∂x + ∂q/∂y with an FFT solver.
    """
    H, W = p.shape

    dpdx = np.zeros_like(p, dtype=np.float32)
    dpdx[:, :-1] = p[:, 1:] - p[:, :-1]
    dpdx[:, -1] = dpdx[:, -2]

    dqdy = np.zeros_like(q, dtype=np.float32)
    dqdy[:-1, :] = q[1:, :] - q[:-1, :]
    dqdy[-1, :] = dqdy[-2, :]

    f = dpdx + dqdy

    F = np.fft.rfftn(f)

    ky = np.fft.fftfreq(H) * 2.0 * np.pi
    kx = np.fft.rfftfreq(W) * 2.0 * np.pi
    ky2 = ky[:, None] ** 2
    kx2 = kx[None, :] ** 2
    denom = kx2 + ky2
    denom[0, 0] = 1.0

    Z = -F / denom
    Z[0, 0] = 0.0

    z = np.fft.irfftn(Z, s=(H, W))
    return z.astype(np.float32)


# -------------------------------------------------------------------------
# Single-image SFS (as before, using averaged LED illumination)
# -------------------------------------------------------------------------

def _reflectance_ring_single(p, q, L, rho=1.0, eps=1e-6):
    """
    Lambertian reflectance for multiple lights averaged into one brightness.
    Used by the single-image SFS variant.
    """
    p = p.astype(np.float32)
    q = q.astype(np.float32)
    H, W = p.shape
    N = L.shape[0]

    w = np.sqrt(1.0 + p * p + q * q + eps)
    w2 = w * w
    w3 = w2 * w

    p_exp = p[..., None]
    q_exp = q[..., None]

    Lx = L[:, 0].reshape(1, 1, N)
    Ly = L[:, 1].reshape(1, 1, N)
    Lz = L[:, 2].reshape(1, 1, N)

    a = -p_exp * Lx + -q_exp * Ly + 1.0 * Lz
    d = a / w[..., None]

    mask_pos = d > 0.0
    d_clipped = np.where(mask_pos, d, 0.0)

    R = rho * d_clipped.mean(axis=-1)

    w2_exp = w2[..., None]
    w3_exp = w3[..., None]

    num_p = -Lx * w2_exp - a * p_exp
    num_q = -Ly * w2_exp - a * q_exp

    ddp = np.where(mask_pos, num_p / w3_exp, 0.0)
    ddq = np.where(mask_pos, num_q / w3_exp, 0.0)

    dR_dp = rho * ddp.mean(axis=-1)
    dR_dq = rho * ddq.mean(axis=-1)

    return R, dR_dp, dR_dq


def estimate_slopes_from_shading(
    E: np.ndarray,
    rho: float = 1.0,
    n_iters: int = 80,
    step: float = 0.05,
    lam_smooth: float = 0.2,
    p0: np.ndarray | None = None,
    q0: np.ndarray | None = None,
    light_dirs: np.ndarray | None = None,
    random_seed: int = 0,
    verbose: bool = False,
):
    """
    Single-image SFS using an averaged LED ring reflectance map.
    """
    E = E.astype(np.float32)
    H, W = E.shape

    if light_dirs is None:
        light_dirs = build_led_dirs_measured()
    light_dirs = light_dirs.astype(np.float32)

    rng = np.random.default_rng(random_seed)

    if p0 is None:
        p = 0.05 * rng.standard_normal(size=(H, W), dtype=np.float32)
    else:
        p = p0.astype(np.float32).copy()

    if q0 is None:
        q = 0.05 * rng.standard_normal(size=(H, W), dtype=np.float32)
    else:
        q = q0.astype(np.float32).copy()

    for it in range(n_iters):
        R, dR_dp, dR_dq = _reflectance_ring_single(p, q, light_dirs, rho=rho)
        e = R - E

        grad_p = e * dR_dp
        grad_q = e * dR_dq

        grad_p -= lam_smooth * _laplacian(p)
        grad_q -= lam_smooth * _laplacian(q)

        p -= step * grad_p
        q -= step * grad_q

        if verbose and (it % 20 == 0 or it == n_iters - 1):
            loss = 0.5 * float(np.mean(e ** 2))
            print(f"[SFS single] iter {it:4d}  loss={loss:.6f}")

    return p, q


def shape_from_shading(
    E: np.ndarray,
    rho: float = 1.0,
    n_iters: int = 80,
    step: float = 0.05,
    lam_smooth: float = 0.2,
    p0: np.ndarray | None = None,
    q0: np.ndarray | None = None,
    light_dirs: np.ndarray | None = None,
    random_seed: int = 0,
    verbose: bool = False,
):
    """
    Wrapper for the single-image SFS variant.
    """
    p, q = estimate_slopes_from_shading(
        E, rho, n_iters, step, lam_smooth,
        p0=p0, q0=q0,
        light_dirs=light_dirs,
        random_seed=random_seed,
        verbose=verbose,
    )
    z = integrate_slopes_poisson(p, q)
    return z, p, q


# -------------------------------------------------------------------------
# Multi-light SFS with per-pixel albedo (recommended for your data)
# -------------------------------------------------------------------------

def estimate_slopes_from_shading_multilight(
    I: np.ndarray,
    light_dirs: np.ndarray | None = None,
    n_iters: int = 80,
    step: float = 0.01,
    lam_smooth: float = 1.0,
    random_seed: int = 0,
    verbose: bool = False,
):
    """
    Multi-light SFS using all K images (one per LED) and recovering a
    per-pixel albedo map.

    Minimizes, with alternating updates,

        L = 1/2 sum_i (rho * max(0, n·s_i) - I_i)^2
            + lam_smooth * (||∇p||^2 + ||∇q||^2)

    where I_i is the image for light i, s_i its direction, n the normal
    from slopes p,q, and rho is per-pixel albedo updated at each iteration
    by least squares.
    """
    I = I.astype(np.float32)
    H, W, K = I.shape

    if light_dirs is None:
        light_dirs = build_led_dirs_measured()
    L = light_dirs.astype(np.float32)
    assert L.shape[0] == K, "Number of lights must match number of images"

    rng = np.random.default_rng(random_seed)
    p = 0.02 * rng.standard_normal(size=(H, W), dtype=np.float32)
    q = 0.02 * rng.standard_normal(size=(H, W), dtype=np.float32)

    # Pre-broadcast light directions
    Lx = L[:, 0].reshape(1, 1, K)
    Ly = L[:, 1].reshape(1, 1, K)
    Lz = L[:, 2].reshape(1, 1, K)

    for it in range(n_iters):
        # --- Geometry: normals & dot products ---
        w = np.sqrt(1.0 + p * p + q * q + 1e-6)          # (H,W)
        p_exp = p[..., None]                             # (H,W,1)
        q_exp = q[..., None]                             # (H,W,1)
        w_exp = w[..., None]

        # v = (-p, -q, 1), a_i = v · s_i
        a = -p_exp * Lx + -q_exp * Ly + 1.0 * Lz         # (H,W,K)

        d = a / w_exp                                    # (H,W,K)
        g = np.maximum(d, 0.0)                           # clipped Lambertian term

        # --- Albedo update (per pixel, LS over lights) ---
        num = np.sum(g * I, axis=-1)                     # (H,W)
        den = np.sum(g * g, axis=-1) + 1e-6
        rho = num / den                                  # (H,W)
        rho = np.clip(rho, 0.0, 2.0)                     # clamp to sane range
        rho_exp = rho[..., None]                         # (H,W,1)

        # --- Predicted intensities & residual ---
        R = rho_exp * g                                  # (H,W,K)
        e = R - I                                        # (H,W,K)

        # --- Derivatives of d_i wrt p,q (only where d>0) ---
        w2 = w * w
        w3 = w2 * w
        w2_exp = w2[..., None]
        w3_exp = w3[..., None]

        num_p = -Lx * w2_exp - a * p_exp
        num_q = -Ly * w2_exp - a * q_exp

        mask_pos = d > 0.0
        ddp = np.where(mask_pos, num_p / w3_exp, 0.0)    # (H,W,K)
        ddq = np.where(mask_pos, num_q / w3_exp, 0.0)

        # --- Gradients of loss wrt p,q ---
        # d(1/2 * sum_i e_i^2)/dp = sum_i e_i * dR_i/dp = sum_i e_i * (rho * ddp_i)
        grad_p = np.sum(e * (rho_exp * ddp), axis=-1)    # (H,W)
        grad_q = np.sum(e * (rho_exp * ddq), axis=-1)

        grad_p -= lam_smooth * _laplacian(p)
        grad_q -= lam_smooth * _laplacian(q)

        p -= step * grad_p
        q -= step * grad_q

        if verbose and (it % 20 == 0 or it == n_iters - 1):
            loss = 0.5 * float(np.mean(e ** 2))
            print(f"[SFS multi] iter {it:4d}  loss={loss:.6f}")

    return p, q, rho


def shape_from_shading_multilight(
    I: np.ndarray,
    light_dirs: np.ndarray | None = None,
    n_iters: int = 80,
    step: float = 0.01,
    lam_smooth: float = 1.0,
    random_seed: int = 0,
    verbose: bool = False,
):
    """
    Full multi-light SFS pipeline.

    Parameters
    ----------
    I : (H, W, K)
        Stack of K images, one per LED, already normalized to [0,1].
    light_dirs : (K,3)
        Corresponding LED directions (unit vectors). If None, uses
        build_led_dirs_measured().
    """
    p, q, rho = estimate_slopes_from_shading_multilight(
        I,
        light_dirs=light_dirs,
        n_iters=n_iters,
        step=step,
        lam_smooth=lam_smooth,
        random_seed=random_seed,
        verbose=verbose,
    )
    z = integrate_slopes_poisson(p, q)
    return z, p, q, rho


def estimate_slopes_from_shading_multilight_const_albedo(
    I: np.ndarray,
    light_dirs: np.ndarray | None = None,
    rho: float = 1.0,
    n_iters: int = 120,
    step: float = 0.01,
    lam_smooth: float = 1.0,
    random_seed: int = 0,
    verbose: bool = False,
):
    """
    Multi-light SFS with a *constant* albedo rho.

    This forces the algorithm to explain brightness variation via shape
    (p,q) rather than via per-pixel albedo. It uses all K images and the
    measured LED directions.
    """
    I = I.astype(np.float32)
    H, W, K = I.shape

    if light_dirs is None:
        light_dirs = build_led_dirs_measured()
    L = light_dirs.astype(np.float32)
    assert L.shape[0] == K, "Number of lights must match number of images"

    rng = np.random.default_rng(random_seed)
    p = 0.02 * rng.standard_normal(size=(H, W), dtype=np.float32)
    q = 0.02 * rng.standard_normal(size=(H, W), dtype=np.float32)

    Lx = L[:, 0].reshape(1, 1, K)
    Ly = L[:, 1].reshape(1, 1, K)
    Lz = L[:, 2].reshape(1, 1, K)

    for it in range(n_iters):
        w = np.sqrt(1.0 + p * p + q * q + 1e-6)
        p_exp = p[..., None]
        q_exp = q[..., None]
        w_exp = w[..., None]

        # v = (-p, -q, 1), a_i = v · s_i
        a = -p_exp * Lx + -q_exp * Ly + 1.0 * Lz
        d = a / w_exp
        g = np.maximum(d, 0.0)              # Lambertian term (H,W,K)

        # Predicted intensities with constant rho
        R = rho * g
        e = R - I                           # residual (H,W,K)

        # Derivatives ddp, ddq as before
        w2 = w * w
        w3 = w2 * w
        w2_exp = w2[..., None]
        w3_exp = w3[..., None]

        num_p = -Lx * w2_exp - a * p_exp
        num_q = -Ly * w2_exp - a * q_exp

        mask_pos = d > 0.0
        ddp = np.where(mask_pos, num_p / w3_exp, 0.0)
        ddq = np.where(mask_pos, num_q / w3_exp, 0.0)

        # Gradients wrt p,q
        grad_p = np.sum(e * (rho * ddp), axis=-1)  # (H,W)
        grad_q = np.sum(e * (rho * ddq), axis=-1)

        grad_p -= lam_smooth * _laplacian(p)
        grad_q -= lam_smooth * _laplacian(q)

        p -= step * grad_p
        q -= step * grad_q

        if verbose and (it % 20 == 0 or it == n_iters - 1):
            loss = 0.5 * float(np.mean(e ** 2))
            print(f"[SFS multi const-rho] iter {it:4d}  loss={loss:.6f}")

    return p, q


def shape_from_shading_multilight_const_albedo(
    I: np.ndarray,
    light_dirs: np.ndarray | None = None,
    rho: float = 1.0,
    n_iters: int = 120,
    step: float = 0.01,
    lam_smooth: float = 1.0,
    random_seed: int = 0,
    verbose: bool = False,
):
    """
    Full pipeline with constant albedo:
        - estimate p,q with all lights
        - integrate to z
        - subtract best-fit plane so relief is easier to see
    """
    p, q = estimate_slopes_from_shading_multilight_const_albedo(
        I,
        light_dirs=light_dirs,
        rho=rho,
        n_iters=n_iters,
        step=step,
        lam_smooth=lam_smooth,
        random_seed=random_seed,
        verbose=verbose,
    )
    z = integrate_slopes_poisson(p, q)

    # Remove best-fit plane (this makes the local shape pop out)
    H, W = z.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    A = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, z.ravel(), rcond=None)
    plane = (coeffs[0] * xx + coeffs[1] * yy + coeffs[2])
    z_rel = z - plane

    return z_rel.astype(np.float32), p, q
