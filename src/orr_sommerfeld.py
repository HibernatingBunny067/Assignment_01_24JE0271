# Orr-Sommerfeld stability of the Blasius boundary layer (parallel-flow approximation)
#
# Perturbation streamfunction  psi = phi(y) exp(i alpha (x - c t)),  lengths scaled by delta*,
# velocities by U_inf, Re = U_inf delta* / nu.
#
#   (U - c)(phi'' - a^2 phi) - U'' phi = (1 / (i a Re)) (phi'''' - 2 a^2 phi'' + a^4 phi)
#
#   BCs:  phi(0) = phi'(0) = 0                  (no-slip, wall)
#         phi, phi' -> 0  as  y -> infinity     (disturbance decays in the freestream)
#
# Method: shooting from the freestream DOWN to the wall.
#   * For y >= y_inf the profile is uniform (U = 1, U'' = 0) and the OS equation has
#     constant coefficients. Its two decaying solutions are
#         phi_1 = exp(-alpha y),   phi_2 = exp(-gamma y),   gamma^2 = alpha^2 + i alpha Re (1 - c)
#     and these are used as the starting values at y = y_inf.
#   * Both solutions are integrated toward the wall with fixed-step RK4 (written from scratch).
#     phi_2 grows ~exp(gamma (y_inf - y)) and would swamp phi_1 (parasitic growth), so the two
#     solutions are carried through their 2x2 minors (compound-matrix method), which keeps the
#     linear independence exactly.
#   * The wall conditions phi(0) = phi'(0) = 0 for a combination A phi_1 + B phi_2 require
#         D(alpha, Re, c) = phi_1(0) phi_2'(0) - phi_2(0) phi_1'(0) = 0,
#     i.e. the first compound variable must vanish at the wall. D(c) = 0 is solved for the
#     complex wave speed c by Newton-Raphson with a finite-difference derivative.
#   * Neutral curve: c_i(alpha, Re) = 0. Found by sweeping a (Re, alpha) grid, contouring c_i = 0,
#     then refining each point with a 2x2 Newton solve for (alpha or Re, c_r) with c real.

import time
import numpy as np
from scipy.integrate import solve_ivp
import contourpy

from src.blasius_bvp_approach import solve_blasius_bvp


# --------------------------------------------------------------------------------------
# Base flow: Blasius profile rescaled to displacement-thickness units
# --------------------------------------------------------------------------------------
class BlasiusProfile:
    """
    U(y) and U''(y) with y = eta / delta*_eta (so that delta* = 1 and Re = Re_delta*).
    f''(0) comes from the solve_bvp Blasius solver; the profile itself is then re-integrated
    forward from the wall with a tight-tolerance DOP853 so it can be sampled anywhere.
    """

    def __init__(self, eta_end=20.0):
        s = solve_blasius_bvp(eta_max=12.0, tol=1e-10)['f_double_prime_0']
        self.f_double_prime_0 = s

        rhs = lambda eta, w: [w[1], w[2], -0.5 * w[0] * w[2]]
        self._sol = solve_ivp(rhs, (0.0, eta_end), [0.0, 0.0, s], method='DOP853',
                              rtol=1e-12, atol=1e-14, dense_output=True)
        self.eta_end = eta_end
        f_end = self._sol.y[0, -1]
        self.delta_star = eta_end - f_end    # delta*_eta = lim (eta - f), in eta units

    def __call__(self, y):
        """Return U, dU/dy, d2U/dy2 at the points y (delta* units)."""
        y = np.atleast_1d(np.asarray(y, dtype=float))
        eta = y * self.delta_star
        inside = eta < self.eta_end
        U   = np.ones_like(eta)
        dU  = np.zeros_like(eta)
        d2U = np.zeros_like(eta)
        if np.any(inside):
            f, df, d2f = self._sol.sol(eta[inside])
            U[inside]   = df
            dU[inside]  = d2f * self.delta_star
            d2U[inside] = -0.5 * f * d2f * self.delta_star**2   # f''' = -f f''/2
        return U, dU, d2U


# --------------------------------------------------------------------------------------
# Shooting solver (compound-matrix form, integrated from y_inf down to the wall)
# --------------------------------------------------------------------------------------
class OrrSommerfeldShooter:
    """
    Evaluates the wall determinant D(alpha, Re, c) for many (alpha, Re, c) at once:
    all cases share the same y-grid, so the RK4 loop runs once over y and every
    arithmetic operation acts on a vector of cases.
    """

    def __init__(self, profile=None, y_inf=8.0, n_steps=2000):
        self.profile = profile if profile is not None else BlasiusProfile()
        self.y_inf   = y_inf
        self.n_steps = n_steps
        self.h       = -y_inf / n_steps                 # negative: wall-ward integration

        y_nodes = np.linspace(y_inf, 0.0, n_steps + 1)
        y_mid   = y_nodes[:-1] + 0.5 * self.h
        U_n, _, U2_n = self.profile(y_nodes)
        U_m, _, U2_m = self.profile(y_mid)
        self._U_n, self._U2_n = U_n, U2_n
        self._U_m, self._U2_m = U_m, U2_m

    @staticmethod
    def _freestream_minors(alpha, gamma):
        # 2x2 minors of [phi_1, phi_2] built from [1, -lam, lam^2, -lam^3] e^{-lam y},
        # divided by the common factor (alpha - gamma) e^{-(alpha+gamma) y}.
        a, g = alpha, gamma
        return [np.ones_like(a), -(a + g), a*a + a*g + g*g, a*g, -a*g*(a + g), a*a*g*g]

    def wall_determinant(self, alpha, Re, c):
        alpha, Re, c = np.broadcast_arrays(np.asarray(alpha, dtype=complex),
                                           np.asarray(Re, dtype=complex),
                                           np.asarray(c, dtype=complex))
        shape = alpha.shape
        alpha, Re, c = alpha.ravel(), Re.ravel(), c.ravel()

        a2    = alpha * alpha
        a4    = a2 * a2
        K     = 1j * alpha * Re
        gamma = np.sqrt(a2 + K * (1.0 - c))           # principal root: Re(gamma) > 0
        shift = alpha + gamma                          # removes the dominant growth rate

        def rhs(U, U2, Y):
            y1, y2, y3, y4, y5, y6 = Y
            Uc = U - c
            A = 2.0 * a2 + K * Uc                      # phi'''' = A phi'' + B phi
            B = -a4 - K * (a2 * Uc + U2)
            return [y2 + shift*y1,
                    y3 + y4 + shift*y2,
                    A*y2 + y5 + shift*y3,
                    y5 + shift*y4,
                    y6 - B*y1 + A*y4 + shift*y5,
                    -B*y2 + shift*y6]

        Y = self._freestream_minors(alpha, gamma)
        h = self.h
        Un, U2n, Um, U2m = self._U_n, self._U2_n, self._U_m, self._U2_m
        for n in range(self.n_steps):
            k1 = rhs(Un[n],   U2n[n],   Y)
            k2 = rhs(Um[n],   U2m[n],   [Y[i] + 0.5*h*k1[i] for i in range(6)])
            k3 = rhs(Um[n],   U2m[n],   [Y[i] + 0.5*h*k2[i] for i in range(6)])
            k4 = rhs(Un[n+1], U2n[n+1], [Y[i] + h*k3[i]     for i in range(6)])
            Y = [Y[i] + (h/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]) for i in range(6)]

        return Y[0].reshape(shape)                     # D = phi_1 phi_2' - phi_2 phi_1' at y = 0

    # ---------------------------------------------------------------- eigenvalue c
    def solve_c(self, alpha, Re, c_guess, tol=1e-10, max_iter=30, eps=1e-7, max_step=0.05,
                return_history=False):
        """Newton-Raphson on D(c) = 0 (vectorised over cases). Returns c and a converged mask."""
        alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
        Re    = np.atleast_1d(np.asarray(Re, dtype=float))
        c     = np.atleast_1d(np.asarray(c_guess, dtype=complex)).copy()
        alpha, Re, c = np.broadcast_arrays(alpha, Re, c)
        c = c.copy()
        active = np.ones(c.shape, dtype=bool)
        history = []

        for _ in range(max_iter):
            idx = np.where(active)[0]
            if idx.size == 0:
                break
            al, R, cc = alpha[idx], Re[idx], c[idx]
            D = self.wall_determinant(np.concatenate([al, al]), np.concatenate([R, R]),
                                      np.concatenate([cc, cc + eps]))
            D0, D1 = D[:idx.size], D[idx.size:]
            dc = D0 * eps / (D1 - D0)
            big = np.abs(dc) > max_step                       # damp large jumps (mode hopping)
            dc[big] *= max_step / np.abs(dc[big])
            c[idx] = cc - dc
            history.append({'c': c[idx].copy(), 'dc': np.abs(dc), 'D': np.abs(D0)})
            active[idx[np.abs(dc) < tol]] = False

        converged = ~active
        if return_history:
            return c, converged, history
        return c, converged

    # ---------------------------------------------------------------- neutral points
    def refine_neutral(self, alpha, Re, c_r, fix='Re', tol=1e-10, max_iter=25):
        """
        Solve D(alpha, Re, c_r) = 0 with c real (neutral, c_i = 0) for two real unknowns:
          fix='Re'    -> unknowns (alpha, c_r) at fixed Re
          fix='alpha' -> unknowns (Re, c_r)    at fixed alpha
        `fix` may be an array of 'Re'/'alpha' strings, one per case.
        """
        alpha = np.atleast_1d(np.asarray(alpha, dtype=float)).copy()
        Re    = np.atleast_1d(np.asarray(Re, dtype=float)).copy()
        c_r   = np.atleast_1d(np.asarray(c_r, dtype=float)).copy()
        fix_alpha = np.broadcast_to(np.asarray(fix) == 'alpha', alpha.shape).copy()
        n = alpha.size
        active = np.ones(n, dtype=bool)

        for _ in range(max_iter):
            idx = np.where(active)[0]
            if idx.size == 0:
                break
            al, R, cr, fa = alpha[idx], Re[idx], c_r[idx], fix_alpha[idx]
            dp = np.where(fa, 1e-6 * R, 1e-7)                 # perturbation of the free parameter
            al_p = np.where(fa, al, al + dp)
            R_p  = np.where(fa, R + dp, R)
            ec = 1e-7
            D = self.wall_determinant(np.concatenate([al, al_p, al]),
                                      np.concatenate([R, R_p, R]),
                                      np.concatenate([cr, cr, cr + ec]))
            m = idx.size
            D0, Dp, Dc = D[:m], D[m:2*m], D[2*m:]
            J_p = (Dp - D0) / dp
            J_c = (Dc - D0) / ec
            # Real 2x2 system:  [Re J_p, Re J_c; Im J_p, Im J_c] [dp; dc] = -[Re D0; Im D0]
            det = J_p.real * J_c.imag - J_c.real * J_p.imag
            d_p = -( D0.real * J_c.imag - J_c.real * D0.imag) / det
            d_c = -( J_p.real * D0.imag - D0.real * J_p.imag) / det
            # damping
            lim_p = np.where(fa, 0.2 * R, 0.02)
            scale = np.minimum(1.0, lim_p / np.maximum(np.abs(d_p), 1e-300))
            scale = np.minimum(scale, 0.02 / np.maximum(np.abs(d_c), 1e-300))
            d_p *= scale
            d_c *= scale
            alpha[idx] = np.where(fa, al, al + d_p)
            Re[idx]    = np.where(fa, R + d_p, R)
            c_r[idx]   = cr + d_c
            rel = np.where(fa, np.abs(d_p) / R, np.abs(d_p))
            active[idx[(rel < tol) & (np.abs(d_c) < tol)]] = False

        return alpha, Re, c_r, ~active


# --------------------------------------------------------------------------------------
# Parameter sweep over (Re, alpha) with continuation of the Tollmien-Schlichting mode
# --------------------------------------------------------------------------------------
def march(shooter, alphas, Res, c0):
    """Scalar continuation of one eigenvalue along a path of (alpha, Re) points."""
    out = np.empty(len(alphas), dtype=complex)
    prev = []
    for k, (a, R) in enumerate(zip(alphas, Res)):
        if not prev:
            guess = c0
        elif len(prev) == 1:
            guess = prev[-1]
        else:
            guess = 2*prev[-1] - prev[-2]               # linear extrapolation
        c, ok = shooter.solve_c(a, R, guess)
        if not ok[0]:
            raise RuntimeError(f"Continuation failed at alpha={a:.4f}, Re={R:.1f}")
        out[k] = c[0]
        prev.append(c[0])
    return out


# Benchmark TS eigenvalue (Jordinson 1970 / Mack 1976), in delta* units:
# Re_d* = 998.06, alpha d* = 0.30802  ->  c = 0.36412286 + 0.00795972 i
SEED = (0.30802, 998.06, 0.36412286 + 0.00795972j)


def sweep_grid(shooter, Re_vals, alpha_vals, alpha_row=0.15, n_sub=4, verbose=True):
    """
    Computes c(alpha, Re) for the TS mode on the whole grid.
    1. Start from the benchmark eigenvalue SEED and continue in alpha (Re fixed) to the
       grid row closest to alpha_row, which stays near the unstable band at all Re.
    2. March along Re on that row (n_sub sub-steps between grid points, extrapolated guesses).
    3. March up and down in alpha, solving a whole row of Re values at once.
    """
    nR, nA = len(Re_vals), len(alpha_vals)
    C = np.full((nA, nR), np.nan + 1j*np.nan)
    j0 = int(np.argmin(np.abs(alpha_vals - alpha_row)))
    i0 = int(np.argmin(np.abs(np.log(Re_vals / SEED[1]))))

    # step 1: seed -> (alpha_vals[j0], Re_vals[i0])
    n1 = 25
    path_a = np.linspace(SEED[0], alpha_vals[j0], n1)
    path_R = np.exp(np.linspace(np.log(SEED[1]), np.log(Re_vals[i0]), n1))
    C[j0, i0] = march(shooter, path_a, path_R, SEED[2])[-1]

    # step 2: along Re on row j0
    for direction in (+1, -1):
        idx = np.arange(i0, nR) if direction > 0 else np.arange(i0, -1, -1)
        if idx.size < 2:
            continue
        logR = np.log(Re_vals[idx])
        fine = np.interp(np.linspace(0, idx.size - 1, (idx.size - 1)*n_sub + 1),
                         np.arange(idx.size), logR)
        cf = march(shooter, np.full(fine.size, alpha_vals[j0]), np.exp(fine), C[j0, i0])
        C[j0, idx] = cf[::n_sub]
    if verbose:
        print(f"  seed row alpha = {alpha_vals[j0]:.4f} done")

    # step 3: rows above / below, whole row at once, n_sub alpha sub-steps per grid row
    for direction in (+1, -1):
        j = j0 + direction
        c_prev, c_curr = None, C[j0].copy()
        while 0 <= j < nA:
            a_from, a_to = alpha_vals[j - direction], alpha_vals[j]
            for a in np.linspace(a_from, a_to, n_sub + 1)[1:]:
                guess = c_curr if c_prev is None else 2*c_curr - c_prev
                guess = np.where(np.isnan(guess), np.nanmean(c_curr), guess)
                c, ok = shooter.solve_c(np.full(nR, a), Re_vals, guess)
                c[~ok] = np.nan
                c_prev, c_curr = c_curr, c
            C[j] = c_curr
            if verbose:
                print(f"  row alpha = {alpha_vals[j]:.4f}: {np.isfinite(c_curr).sum()}/{nR} converged")
            j += direction
    return C


def neutral_curve_from_grid(shooter, Re_vals, alpha_vals, C, n_points=60):
    """Contour c_i = 0 on the grid, then refine every contour point onto the exact neutral curve."""
    logR = np.log10(Re_vals)
    ci = np.where(np.isnan(C.imag), -1.0, C.imag)
    gen = contourpy.contour_generator(logR, alpha_vals, ci)
    lines = gen.lines(0.0)
    if not lines:
        raise RuntimeError("No neutral curve found in the grid")
    line = max(lines, key=len)                      # main TS neutral loop
    cr_grid = np.where(np.isnan(C.real), np.nanmean(C.real), C.real)

    # resample uniformly along the contour (in scaled coordinates)
    xs = (line[:, 0] - logR.min()) / np.ptp(logR)
    ys = (line[:, 1] - alpha_vals.min()) / np.ptp(alpha_vals)
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    t = np.linspace(0, s[-1], n_points)
    lr = np.interp(t, s, line[:, 0])
    al = np.interp(t, s, line[:, 1])
    dx = np.gradient(np.interp(t, s, xs))
    dy = np.gradient(np.interp(t, s, ys))
    fix = np.where(np.abs(dy) < np.abs(dx), 'Re', 'alpha')   # cut the curve transversally

    # c_r guess: bilinear lookup of the grid
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((alpha_vals, logR), cr_grid)
    cr0 = interp(np.column_stack([al, lr]))

    a_n, R_n, cr_n, ok = shooter.refine_neutral(al, 10**lr, cr0, fix=fix)
    return {'alpha': a_n[ok], 'Re': R_n[ok], 'c_r': cr_n[ok],
            'alpha_raw': al, 'Re_raw': 10**lr, 'n_failed': int((~ok).sum())}


def critical_point(shooter, neutral):
    """Minimum Re on the neutral curve: fixed-alpha solves around the nose + parabola fit."""
    k = int(np.argmin(neutral['Re']))
    a0, R0, c0 = neutral['alpha'][k], neutral['Re'][k], neutral['c_r'][k]
    for width in (0.03, 0.006):
        al = a0 + np.linspace(-width, width, 9)
        a_n, R_n, cr_n, ok = shooter.refine_neutral(al, np.full(9, R0), np.full(9, c0), fix='alpha')
        p = np.polyfit(a_n[ok], R_n[ok], 2)
        a0 = -p[1] / (2*p[0])
        j = int(np.argmin(R_n[ok]))
        R0, c0 = R_n[ok][j], cr_n[ok][j]
    a_c, R_c, cr_c, ok = shooter.refine_neutral([a0], [R0], [c0], fix='alpha')
    return {'alpha': a_c[0], 'Re': R_c[0], 'c_r': cr_c[0], 'converged': bool(ok[0])}


if __name__ == '__main__':
    t0 = time.perf_counter()
    sh = OrrSommerfeldShooter()
    print(f"delta*_eta = {sh.profile.delta_star:.8f},  f''(0) = {sh.profile.f_double_prime_0:.10f}")
    # Benchmark (Jordinson 1970 / Mack 1976): Re_l = 580, alpha_l = 0.179 in Blasius-length units
    d = sh.profile.delta_star
    c, ok, hist = sh.solve_c(0.179 * d, 580.0 * d, 0.36 + 0.01j, return_history=True)
    print(f"Re_d* = {580*d:.2f}, alpha d* = {0.179*d:.5f}")
    print(f"c = {c[0].real:.8f} {c[0].imag:+.8f}i   (ref 0.36412286 +0.00795972i)  iters={len(hist)}")
    print(f"time {time.perf_counter()-t0:.2f} s")
