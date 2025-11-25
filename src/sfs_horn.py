import numpy as np


# ---------- Reflectance model (rotationally symmetric Lambertian) ---------- #

def lambertian_headlight_reflectance(p, q, rho=1.0, eps=1e-6):
    """
    Rotationally symmetric Lambertian reflectance map.

    For orthographic projection and a headlight source (light along +z, same
    as viewing direction), the unit surface normal is

        n = (-p, -q, 1) / sqrt(1 + p^2 + q^2)

    and the Lambertian brightness is

        R(p,q) = rho * max(0, n·s) = rho / sqrt(1 + p^2 + q^2).

    Parameters
    ----------
    p, q : ndarray (H, W)
        Surface slopes dz/dx and dz/dy.
    rho : float
        Albedo (assumed constant).
    eps : float
        Small term to avoid division by zero.

    Returns
    -------
    R : ndarray (H, W)
        Predicted brightness.
    """
    return rho / np.sqrt(1.0 + p * p + q * q + eps)


# ---------- SFS: estimate slopes from shading by gradient descent ---------- #

def _laplacian(X):
    """5-point Laplacian with periodic wrap-around (simple, fast)."""
    lap = -4.0 * X
    lap += np.roll(X, 1, axis=0)
    lap += np.roll(X, -1, axis=0)
    lap += np.roll(X, 1, axis=1)
    lap += np.roll(X, -1, axis=1)
    return lap


def estimate_slopes_from_shading(
    E,
    rho=1.0,
    n_iters=200,
    step=0.1,
    lam_smooth=0.1,
    p0=None,
    q0=None,
):
    """
    Horn-style shape-from-shading for a rotationally symmetric Lambertian
    reflectance map (LED ring around camera approximated as headlight).

    Minimizes over p,q:

        L(p,q) = 1/2 * (R(p,q) - E)^2 + lam_smooth * ||∇p||^2 + ||∇q||^2

    using simple gradient descent.

    Parameters
    ----------
    E : ndarray (H, W), float32/float64
        Brightness / irradiance image.
        - Should be linear (gamma removed).
        - Typically normalized to [0, 1].
        - Brightest front-facing patch ≈ rho.
    rho : float
        Assumed constant albedo.
    n_iters : int
        Number of gradient descent iterations.
    step : float
        Gradient descent step size.
    lam_smooth : float
        Weight of the smoothness prior on slopes p and q.
    p0, q0 : ndarray or None
        Optional initial guesses for p and q. If None, start from zeros.

    Returns
    -------
    p, q : ndarray (H, W)
        Estimated slopes dz/dx and dz/dy.
    """
    E = E.astype(np.float32)
    H, W = E.shape

    if p0 is None:
        p = np.zeros_like(E, dtype=np.float32)
    else:
        p = p0.astype(np.float32).copy()

    if q0 is None:
        q = np.zeros_like(E, dtype=np.float32)
    else:
        q = q0.astype(np.float32).copy()

    rho_eff = float(rho)

    for _ in range(n_iters):
        # Predicted brightness and residual
        R = lambertian_headlight_reflectance(p, q, rho=rho_eff)
        e = R - E  # brightness residual

        # Derivatives of R w.r.t. p and q (from the reflectance map):
        #   R = rho / sqrt(1 + u), with u = p^2 + q^2
        #   dR/dp = -rho * p * (1+u)^(-3/2)
        # but (1+u)^(-3/2) = (R/rho)^3, so
        #   dR/dp = -p * (R/rho)^3
        R_over_rho = R / (rho_eff + 1e-6)
        R_cubed = R_over_rho ** 3

        dR_dp = -p * R_cubed
        dR_dq = -q * R_cubed

        # Data-term gradient: d(1/2 e^2)/dp = e * dR_dp
        grad_p = e * dR_dp
        grad_q = e * dR_dq

        # Add smoothness term: roughly lam_smooth * Laplacian on p and q
        grad_p -= lam_smooth * _laplacian(p)
        grad_q -= lam_smooth * _laplacian(q)

        # Gradient descent update
        p -= step * grad_p
        q -= step * grad_q

    return p, q


# ---------- Integrate slopes to a height map via Poisson solver ---------- #

def integrate_slopes_poisson(p, q):
    """
    Integrate slopes p = dz/dx and q = dz/dy to obtain height map z by
    solving the Poisson equation

        Δz = ∂p/∂x + ∂q/∂y

    with Neumann-like boundary conditions and mean(z) = 0.

    This is a standard least-squares integration of a gradient field, using
    an FFT-based Poisson solver.

    Parameters
    ----------
    p, q : ndarray (H, W)
        Surface slopes.

    Returns
    -------
    z : ndarray (H, W)
        Reconstructed height map (relative; additive constant fixed to 0).
    """
    H, W = p.shape

    # Finite differences for divergence of slope field
    dpdx = np.zeros_like(p, dtype=np.float32)
    dpdx[:, :-1] = p[:, 1:] - p[:, :-1]
    dpdx[:, -1] = dpdx[:, -2]

    dqdy = np.zeros_like(q, dtype=np.float32)
    dqdy[:-1, :] = q[1:, :] - q[:-1, :]
    dqdy[-1, :] = dqdy[-2, :]

    f = dpdx + dqdy  # RHS of Poisson equation

    # Solve Δz = f in Fourier domain
    F = np.fft.rfftn(f)

    ky = np.fft.fftfreq(H) * 2.0 * np.pi
    kx = np.fft.rfftfreq(W) * 2.0 * np.pi
    ky2 = ky[:, None] ** 2
    kx2 = kx[None, :] ** 2
    denom = kx2 + ky2
    denom[0, 0] = 1.0  # avoid division by zero at DC

    Z = -F / denom
    Z[0, 0] = 0.0  # fix mean(z) = 0

    z = np.fft.irfftn(Z, s=(H, W))
    return z.astype(np.float32)


# ---------- Convenience wrapper ---------- #

def shape_from_shading(
    E,
    rho=1.0,
    n_iters=200,
    step=0.1,
    lam_smooth=0.1,
    p0=None,
    q0=None,
):
    """
    Full SFS pipeline for a single image using Horn's reflectance-map
    formulation with a rotationally symmetric Lambertian model.

        1. Estimate slopes p, q from shading (brightness E).
        2. Integrate slopes to a height map z via Poisson integration.

    Parameters
    ----------
    E : ndarray (H, W)
        Input brightness image (linear).
    rho, n_iters, step, lam_smooth, p0, q0 :
        Passed to estimate_slopes_from_shading().

    Returns
    -------
    z : ndarray (H, W)
        Relative height map (mean(z) = 0).
    p, q : ndarray (H, W)
        Surface slopes dz/dx, dz/dy.
    """
    p, q = estimate_slopes_from_shading(
        E,
        rho=rho,
        n_iters=n_iters,
        step=step,
        lam_smooth=lam_smooth,
        p0=p0,
        q0=q0,
    )
    z = integrate_slopes_poisson(p, q)
    return z, p, q
