# Assignment 01 - Falkner-Skan Equation: Approach 2
# Shooting method: guess f''(0) = s, integrate as IVP, use Newton-Raphson to
# correct s until f'(eta_inf) = 1
#
# ODE: f''' + f*f'' - (f')^2 + 1 = 0
# BCs: f(0) = 0, f'(0) = 0, f'(eta_inf) = 1

import time
import numpy as np
from scipy.integrate import solve_ivp


# State: y = [f, f', f'']
def ode(eta, y):
    f, df, d2f = y[0], y[1], y[2]
    return [df, d2f, df**2 - f*d2f - 1.0]


def solve_falkner_skan_ivp_newton(eta_max=8.0, s_init=1.2, tol=1e-6,
                                   max_iter=30, eps=1e-4):
    """
    Shooting method using finite differences for the Newton-Raphson Jacobian.
    eps: small perturbation used to approximate dPhi/ds numerically.
    """
    start_time = time.perf_counter()

    s = float(s_init)
    history  = []
    converged = False
    final_sol = None

    t_span = (0.0, eta_max)
    t_eval = np.linspace(0.0, eta_max, 500)

    for k in range(1, max_iter + 1):
        # Integrate with current guess s
        sol = solve_ivp(ode, t_span, [0.0, 0.0, s],
                        t_eval=t_eval, rtol=1e-8, atol=1e-10)

        # Integrate with perturbed guess s + eps
        sol_p = solve_ivp(ode, t_span, [0.0, 0.0, s + eps],
                          rtol=1e-8, atol=1e-10)

        residual = sol.y[1, -1] - 1.0                             # Phi(s)
        dphi_ds  = (sol_p.y[1, -1] - sol.y[1, -1]) / eps         # Phi'(s) via FD
        step     = residual / dphi_ds

        history.append({'iteration': k, 's': s, 'residual': residual,
                        'dphi_ds': dphi_ds, 'step': step})
        final_sol = sol

        if abs(residual) < tol:
            converged = True
            break

        s -= step

    elapsed = time.perf_counter() - start_time

    if not converged:
        raise RuntimeError(f"Newton-Raphson did not converge in {max_iter} iterations")

    return {
        'method': 'solve_ivp + Newton (FD)',
        'converged': converged,
        'iterations': len(history),
        'f_double_prime_0': s,
        'history': history,
        'eta': final_sol.t,
        'f': final_sol.y[0],
        'df': final_sol.y[1],
        'd2f': final_sol.y[2],
        'elapsed_time': elapsed,
        'eta_max': eta_max
    }


if __name__ == '__main__':
    res = solve_falkner_skan_ivp_newton()

    print(f"{'Iter':<6}{'s':<18}{'Residual':<16}{'dPhi/ds':<14}{'Step'}")
    print("-" * 65)
    for row in res['history']:
        print(f"{row['iteration']:<6}{row['s']:<18.10f}{row['residual']:<16.3e}"
              f"{row['dphi_ds']:<14.6f}{row['step']:.3e}")

    print(f"\nf''(0)         = {res['f_double_prime_0']:.8f}")
    print(f"Benchmark      = 1.23258766")
    print(f"Absolute error = {abs(res['f_double_prime_0'] - 1.23258766):.3e}")
    print(f"Iterations     = {res['iterations']}")
    print(f"Time           = {res['elapsed_time']*1000:.1f} ms")
