# Assignment 01 - Falkner-Skan Equation: Approach 3
# Fully from scratch: RK4 integrator + Newton-Raphson shooting
# No scipy used for integration.
#
# ODE: f''' + f*f'' - (f')^2 + 1 = 0
# BCs: f(0) = 0, f'(0) = 0, f'(eta_inf) = 1

import time
import numpy as np


# Augmented 6-state system: [f, f', f'', z0, z1, z2]
# where z = [df/ds, df'/ds, df''/ds] are the sensitivity variables (z(0) = [0,0,1])
def derivatives(eta, w):
    f, df, d2f = w[0], w[1], w[2]
    z0, z1, z2 = w[3], w[4], w[5]

    d3f = df*df - f*d2f - 1.0
    dz2 = 2.0*df*z1 - z0*d2f - f*z2

    return np.array([df, d2f, d3f, z1, z2, dz2])


def rk4_step(f, eta, w, h):
    k1 = f(eta, w)
    k2 = f(eta + h/2, w + h/2 * k1)
    k3 = f(eta + h/2, w + h/2 * k2)
    k4 = f(eta + h,   w + h   * k3)
    return w + (h/6) * (k1 + 2*k2 + 2*k3 + k4)


def rk4_integrate(f, w0, eta_span, h):
    eta0, eta_max = eta_span
    n = int(round((eta_max - eta0) / h))
    h = (eta_max - eta0) / n  # adjust h to land exactly on eta_max

    eta_arr = np.zeros(n + 1)
    w_arr   = np.zeros((len(w0), n + 1))

    eta_arr[0]   = eta0
    w_arr[:, 0]  = w0
    eta = eta0
    w   = np.array(w0)

    for i in range(n):
        w    = rk4_step(f, eta, w, h)
        eta += h
        eta_arr[i+1]  = eta
        w_arr[:, i+1] = w

    return eta_arr, w_arr


def trapezoidal(y, x):
    total = 0.0
    for i in range(len(x) - 1):
        total += 0.5 * (y[i] + y[i+1]) * (x[i+1] - x[i])
    return total


def boundary_layer_properties(eta, df):
    # delta_99: interpolate where f' crosses 0.99
    idx = np.where(df >= 0.99)[0]
    if len(idx) > 0 and idx[0] > 0:
        i = idx[0]
        delta_99 = eta[i-1] + (0.99 - df[i-1]) * (eta[i] - eta[i-1]) / (df[i] - df[i-1])
    else:
        delta_99 = eta[-1]

    delta_star   = trapezoidal(1.0 - df, eta)
    theta        = trapezoidal(df * (1.0 - df), eta)
    shape_factor = delta_star / theta if theta > 1e-12 else 0.0

    return {'delta_99': delta_99, 'delta_star': delta_star,
            'theta': theta, 'shape_factor': shape_factor}


def solve_falkner_skan_scratch(eta_max=8.0, h=0.01, s_init=1.2, tol=1e-8, max_iter=30):
    start_time = time.perf_counter()

    s = float(s_init)
    history  = []
    converged = False
    eta_arr, w_arr = None, None

    for k in range(1, max_iter + 1):
        w0 = np.array([0.0, 0.0, s, 0.0, 0.0, 1.0])
        eta_arr, w_arr = rk4_integrate(derivatives, w0, (0.0, eta_max), h)

        residual = w_arr[1, -1] - 1.0  # f'(eta_inf) - 1
        dphi_ds  = w_arr[4, -1]         # z1(eta_inf)
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
    res = solve_falkner_skan_scratch()

    print(f"{'Iter':<6}{'s':<18}{'Residual':<16}{'dPhi/ds':<14}{'Step'}")
    print("-" * 65)
    for row in res['history']:
        print(f"{row['iteration']:<6}{row['s']:<18.10f}{row['residual']:<16.3e}"
              f"{row['dphi_ds']:<14.6f}{row['step']:.3e}")

    print(f"\nf''(0)         = {res['f_double_prime_0']:.10f}")
    print(f"Benchmark      = 1.2325876568")
    print(f"Absolute error = {abs(res['f_double_prime_0'] - 1.2325876568):.3e}")
    print(f"delta_99       = {res['delta_99']:.4f}")
    print(f"delta*         = {res['delta_star']:.4f}  (ref: 0.6479)")
    print(f"theta          = {res['theta']:.4f}  (ref: 0.2923)")
    print(f"H              = {res['shape_factor']:.4f}  (ref: 2.216)")
    print(f"Time           = {res['elapsed_time']*1000:.1f} ms")
