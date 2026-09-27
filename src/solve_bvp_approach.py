# Assignment 01 - Falkner-Skan Equation: Approach 1
# Solving the BVP using scipy's solve_bvp (collocation method)
#
# ODE: f''' + f*f'' - (f')^2 + 1 = 0
# BCs: f(0) = 0, f'(0) = 0, f'(eta_inf) = 1

import time
import numpy as np
from scipy.integrate import solve_bvp


# The ODE as a first-order system: y = [f, f', f'']
def ode_system(eta, y):
    f, df, d2f = y[0], y[1], y[2]
    d3f = df**2 - f * d2f - 1.0
    return np.vstack((df, d2f, d3f))


# Residuals for the boundary conditions
def bc(ya, yb):
    return np.array([
        ya[0],       # f(0) = 0
        ya[1],       # f'(0) = 0
        yb[1] - 1.0  # f'(eta_inf) = 1
    ])


def solve_falkner_skan_bvp(eta_max=8.0, n_nodes=100, tol=1e-8, max_nodes=50000):
    start_time = time.perf_counter()

    eta_mesh = np.linspace(0.0, eta_max, n_nodes)

    # Initial guess using tanh profile (smooth, satisfies BCs roughly)
    y_guess = np.zeros((3, n_nodes))
    y_guess[0] = np.log(np.cosh(eta_mesh))
    y_guess[1] = np.tanh(eta_mesh)
    y_guess[2] = 1.0 / np.cosh(eta_mesh)**2

    sol = solve_bvp(ode_system, bc, eta_mesh, y_guess, tol=tol, max_nodes=max_nodes, verbose=0)
    elapsed = time.perf_counter() - start_time

    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")

    return {
        'method': 'solve_bvp',
        'success': sol.success,
        'message': sol.message,
        'eta': sol.x,
        'f': sol.y[0],
        'df': sol.y[1],
        'd2f': sol.y[2],
        'f_double_prime_0': sol.y[2, 0],
        'n_nodes': len(sol.x),
        'elapsed_time': elapsed,
        'eta_max': eta_max
    }


if __name__ == '__main__':
    res = solve_falkner_skan_bvp()
    print(f"f''(0)         = {res['f_double_prime_0']:.8f}")
    print(f"Benchmark      = 1.23258765")
    print(f"Absolute error = {abs(res['f_double_prime_0'] - 1.23258765):.2e}")
    print(f"Mesh nodes     = {res['n_nodes']}")
    print(f"Time           = {res['elapsed_time']*1000:.1f} ms")
