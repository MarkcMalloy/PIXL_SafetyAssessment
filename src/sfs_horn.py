import numpy as np
from typing import Optional

# -------------------------------------------------------------------------
# LED ring geometry (6 LEDs, rotationally symmetric, 25° tilt)
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# LED ring geometry: measured 6-LED ring around the camera
# -------------------------------------------------------------------------

def build_led_dirs_measured() -> np.ndarray:
    """
    Build light directions for the real LED ring.

    We model each LED as a directional source whose optical axis points
    inward toward the image center, tilted by the measured 'inward_tilt'
    with respect to the +z viewing axis.

    The radius and height matter mainly for intensity falloff, which we
    ignore here; the *direction* of the LED axis is what we need for SFS.
    """

    LED_data_height_m = 0.008975 + 0.005  # not used directly in direction
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

        # LED axis tilted by theta away from +z toward the center.
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
# Reflectance map for the LED ring (Lambertian)
# -------------------------------------------------------------------------

def _reflectance_ring(p, q, L, rho=1.0, eps=1e-6):
    """
    Lambertian reflectance map for a surface with slopes p, q illuminated by
    multiple point lights in L (N,3), assuming orthographic projection.

    Surface normal for slope (p,q) is

        n = (-p, -q, 1) / sqrt(1 + p^2 + q^2)

    Brightness for LED i is

        B_i = rho * max(0, n · s_i)

    and the observed brightness (what we feed into SFS) is the average over
    all LEDs:

        R(p,q) = (rho / N) * sum_i max(0, n · s_i).

    This function returns R along with partial derivatives dR/dp, dR/dq.

    Parameters
    ----------
    p, q : (H, W) float32
        Slopes dz/dx, dz/dy.
    L : (N, 3) float32
        Light directions (unit vectors).
    rho : float
        Albedo (assumed constant here).
    eps : float
        Small term to avoid division by zero.

    Returns
    -------
    R : (H, W) float32
    dR_dp : (H, W) float32
    dR_dq : (H, W) float32
    """
    p = p.astype(np.float32)
    q = q.astype(np.float32)
    H, W = p.shape
    N = L.shape[0]

    # Geometry factors
    w = np.sqrt(1.0 + p * p + q * q + eps)  # (H,W)
    w2 = w * w
    w3 = w2 * w

    p_exp = p[..., None]  # (H,W,1)
    q_exp = q[..., None]  # (H,W,1)

    Lx = L[:, 0].reshape(1, 1, N)  # (1,1,N)
    Ly = L[:, 1].reshape(1, 1, N)
    Lz = L[:, 2].reshape(1, 1, N)

    # v = (-p, -q, 1), a_i = v · s_i
    a = -p_exp * Lx + -q_exp * Ly + 1.0 * Lz  # (H,W,N)

    # Raw dot product with normal: d_i = a / |v| = a / w
    d = a / w[..., None]  # (H,W,N)

    # Only LEDs with positive contribution are active
    mask_pos = d > 0.0
    d_clipped = np.where(mask_pos, d, 0.0)

    # Reflectance = average contribution over LEDs
    R = rho * d_clipped.mean(axis=-1)  # (H,W)

    # Derivatives of d_i wrt p, q when d_i > 0
    #
    #   d_i = a / w
    #   ∂a/∂p = -s_ix,   ∂a/∂q = -s_iy
    #   ∂w/∂p =  p / w,  ∂w/∂q =  q / w
    #
    #   ∂d_i/∂p = ( (∂a/∂p) w - a (∂w/∂p) ) / w^2
    #           = (-s_ix w^2 - a p) / w^3
    #
    #   ∂d_i/∂q = (-s_iy w^2 - a q) / w^3
    #
    w2_exp = w2[..., None]  # (H,W,1)
    w3_exp = w3[..., None]

    num_p = -Lx * w2_exp - a * p_exp
    num_q = -Ly * w2_exp - a * q_exp

    ddp = num_p / w3_exp
    ddq = num_q / w3_exp

    ddp = np.where(mask_pos, ddp, 0.0)
    ddq = np.where(mask_pos, ddq, 0.0)

    dR_dp = rho * ddp.mean(axis=-1)
    dR_dq = rho * ddq.mean(axis=-1)

    return R, dR_dp, dR_dq


# -------------------------------------------------------------------------
# Optimization of slopes p, q
# -------------------------------------------------------------------------

def _laplacian(X):
    """5-point Laplacian with periodic wrap-around."""
    lap = -4.0 * X
    lap += np.roll(X, 1, axis=0)
    lap += np.roll(X, -1, axis=0)
    lap += np.roll(X, 1, axis=1)
    lap += np.roll(X, -1, axis=1)
    return lap


def estimate_slopes_from_shading(
    E: np.ndarray,
    rho: float = 1.0,
    n_iters: int = 80,
    step: float = 0.05,
    lam_smooth: float = 0.2,
    p0: Optional[np.ndarray] = None,
    q0: Optional[np.ndarray] = None,
    light_dirs: Optional[np.ndarray] = None,
    random_seed: int = 0,
    verbose: bool = False,
):
    """
    Estimate slopes p, q from a single brightness image E(x,y) using a
    Horn-style variational SFS, with a rotationally symmetric LED ring model.

        L(p,q) = 1/2 * (R(p,q) - E)^2
                 + lam_smooth * ( ||∇p||^2 + ||∇q||^2 )

    Parameters
    ----------
    E : (H, W) float32
        Input brightness / irradiance image (ideally linear, in [0,1]).
    rho : float
        Albedo (assumed constant).
    n_iters : int
        Number of gradient descent iterations.
    step : float
        Gradient descent step size.
    lam_smooth : float
        Smoothness weight on p and q.
    p0, q0 : (H, W) or None
        Optional initial slope guesses. If None, we start from small random
        slopes to avoid getting stuck at the flat solution p=q=0.
    light_dirs : (N,3) or None
        Optional LED directions. If None, a default 6-LED ring is used.
    random_seed : int
        Seed for the random initialization.
    verbose : bool
        If True, print loss every ~20 iterations.

    Returns
    -------
    p, q : (H, W) float32
        Estimated slopes dz/dx and dz/dy.
    """
    E = E.astype(np.float32)
    H, W = E.shape

    if light_dirs is None:
        light_dirs = build_led_dirs_measured()

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
        R, dR_dp, dR_dq = _reflectance_ring(p, q, light_dirs, rho=rho)

        e = R - E  # brightness residual

        # Data term gradient
        grad_p = e * dR_dp
        grad_q = e * dR_dq

        # Smoothness term: ~ lam_smooth * (-Δp), lam_smooth * (-Δq)
        grad_p -= lam_smooth * _laplacian(p)
        grad_q -= lam_smooth * _laplacian(q)

        # Gradient descent update
        p -= step * grad_p
        q -= step * grad_q

        if verbose and (it % 20 == 0 or it == n_iters - 1):
            loss = 0.5 * float(np.mean(e ** 2))
            print(f"[SFS] iter {it:4d}  loss={loss:.6f}")

    return p, q


# -------------------------------------------------------------------------
# Integration of slopes to a depth map (Poisson solver)
# -------------------------------------------------------------------------

def integrate_slopes_poisson(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Integrate slope fields p = dz/dx and q = dz/dy into a height map z by
    solving the Poisson equation

        Δz = ∂p/∂x + ∂q/∂y

    using an FFT-based solver, with mean(z) = 0.

    Parameters
    ----------
    p, q : (H, W) float32
        Surface slopes.

    Returns
    -------
    z : (H, W) float32
        Relative height map (additive constant fixed to 0).
    """
    H, W = p.shape

    dpdx = np.zeros_like(p, dtype=np.float32)
    dpdx[:, :-1] = p[:, 1:] - p[:, :-1]
    dpdx[:, -1] = dpdx[:, -2]

    dqdy = np.zeros_like(q, dtype=np.float32)
    dqdy[:-1, :] = q[1:, :] - q[:-1, :]
    dqdy[-1, :] = dqdy[-2, :]

    f = dpdx + dqdy  # RHS of Poisson equation

    F = np.fft.rfftn(f)

    ky = np.fft.fftfreq(H) * 2.0 * np.pi
    kx = np.fft.rfftfreq(W) * 2.0 * np.pi
    ky2 = ky[:, None] ** 2
    kx2 = kx[None, :] ** 2
    denom = kx2 + ky2
    denom[0, 0] = 1.0  # avoid divide-by-zero at DC

    Z = -F / denom
    Z[0, 0] = 0.0  # fix mean(z) = 0

    z = np.fft.irfftn(Z, s=(H, W))
    return z.astype(np.float32)


# -------------------------------------------------------------------------
# Public API: single-call SFS
# -------------------------------------------------------------------------

def shape_from_shading(
    E: np.ndarray,
    rho: float = 1.0,
    n_iters: int = 80,
    step: float = 0.05,
    lam_smooth: float = 0.2,
    p0: Optional[np.ndarray] = None,
    q0: Optional[np.ndarray] = None,
    light_dirs: Optional[np.ndarray] = None,
    random_seed: int = 0,
    verbose: bool = False,
):
    """
    Full shape-from-shading pipeline:

        1. Estimate slopes p, q from a single brightness image E(x,y),
           using a reflectance map derived from a 6-LED ring.
        2. Integrate the slopes to obtain a relative depth map z.

    Parameters
    ----------
    E : (H, W) float32
        Brightness / irradiance image (ideally linear, in [0,1]).
    rho, n_iters, step, lam_smooth, p0, q0, light_dirs, random_seed, verbose :
        See estimate_slopes_from_shading().

    Returns
    -------
    z : (H, W) float32
        Relative depth (mean(z) = 0).
    p, q : (H, W) float32
        Slopes dz/dx and dz/dy.
    """
    p, q = estimate_slopes_from_shading(
        E,
        rho=rho,
        n_iters=n_iters,
        step=step,
        lam_smooth=lam_smooth,
        p0=p0,
        q0=q0,
        light_dirs=light_dirs,
        random_seed=random_seed,
        verbose=verbose,
    )
    z = integrate_slopes_poisson(p, q)
    return z, p, q
