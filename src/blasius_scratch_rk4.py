# Blasius Boundary Layer: Approach 3
# Fully from scratch: RK4 integrator + Newton-Raphson shooting
# No scipy used for integration.
#
# ODE: f''' + (1/2) f f'' = 0
# BCs: f(0) = 0, f'(0) = 0, f'(eta_inf) = 1
#
# The generic RK4 integrator and the boundary-layer integrals are reused from
# the Falkner-Skan scratch solver; only the right-hand side changes.

import time
import numpy as np

from src.scratch_rk4_newton import rk4_integrate, boundary_layer_properties

BENCHMARK = 0.332057336215


# Augmented 6-state system: [f, f', f'', z0, z1, z2]
# where z = [df/ds, df'/ds, df''/ds] are the sensitivity variables (z(0) = [0,0,1])
def derivatives(eta, w):
    f, df, d2f = w[0], w[1], w[2]
    z0, z1, z2 = w[3], w[4], w[5]

    d3f = -0.5 * f * d2f
    dz2 = -0.5 * (z0 * d2f + f * z2)

    return np.array([df, d2f, d3f, z1, z2, dz2])


def solve_blasius_scratch(eta_max=10.0, h=0.01, s_init=0.3, tol=1e-10, max_iter=30):
    start_time = time.perf_counter()

    s = float(s_init)
    history  = []
    converged = False
    eta_arr, w_arr = None, None

    for k in range(1, max_iter + 1):
        w0 = np.array([0.0, 0.0, s, 0.0, 0.0, 1.0])
        eta_arr, w_arr = rk4_integrate(derivatives, w0, (0.0, eta_max), h)

        residual = w_arr[1, -1] - 1.0  # f'(eta_inf) - 1
        dphi_ds  = w_arr[4, -1]         # z1(eta_inf), exact sensitivity
        step     = residual / dphi_ds

        history.append({'iteration': k, 's': s, 'residual': residual,
                        'dphi_ds': dphi_ds, 'step': step})

        if abs(residual) < tol:
            converged = True
            break

        s -= step

    elapsed = time.perf_counter() - start_time

    if not converged:
        raise RuntimeError(f"Did not converge in {max_iter} iterations")

    bl = boundary_layer_properties(eta_arr, w_arr[1])

    return {
        'method': 'Scratch RK4 + Newton',
        'converged': converged,
        'iterations': len(history),
        'f_double_prime_0': s,
        'history': history,
        'eta': eta_arr,
        'f': w_arr[0],
        'df': w_arr[1],
        'd2f': w_arr[2],
        'h': h,
        'elapsed_time': elapsed,
        'eta_max': eta_max,
        **bl
    }


if __name__ == '__main__':
    res = solve_blasius_scratch()

    print(f"{'Iter':<6}{'s':<18}{'Residual':<16}{'dPhi/ds':<14}{'Step'}")
    print("-" * 65)
    for row in res['history']:
        print(f"{row['iteration']:<6}{row['s']:<18.10f}{row['residual']:<16.3e}"
              f"{row['dphi_ds']:<14.6f}{row['step']:.3e}")

    print(f"\nf''(0)         = {res['f_double_prime_0']:.10f}")
    print(f"Benchmark      = {BENCHMARK:.10f}")
    print(f"Absolute error = {abs(res['f_double_prime_0'] - BENCHMARK):.3e}")
    print(f"delta_99       = {res['delta_99']:.4f}  (ref: 4.91)")
    print(f"delta*         = {res['delta_star']:.4f}  (ref: 1.7208)")
    print(f"theta          = {res['theta']:.4f}  (ref: 0.6641)")
    print(f"H              = {res['shape_factor']:.4f}  (ref: 2.5911)")
    print(f"Time           = {res['elapsed_time']*1000:.1f} ms")
