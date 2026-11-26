# Blasius boundary layer (base flow for the Orr-Sommerfeld analysis): runs the three
# solvers, prints tables, writes blasius_*.png figures and blasius_table.tex to
# latex_report/Orr_Sommerfeld/ (combined report).
#
# Called automatically as the first stage of:  python main.py orr-sommerfeld
#
# ODE: f''' + (1/2) f f'' = 0,  f(0) = f'(0) = 0,  f'(inf) = 1

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.blasius_bvp_approach import solve_blasius_bvp
from src.blasius_ivp_newton   import solve_blasius_ivp_newton
from src.blasius_scratch_rk4  import solve_blasius_scratch, boundary_layer_properties

BENCHMARK  = 0.332057336215
LIT        = {'delta_99': 4.91, 'delta_star': 1.7208, 'theta': 0.6641, 'H': 2.5911}
ETA_MAX    = 10.0
OUT_DIR    = os.path.join("latex_report", "Orr_Sommerfeld")
FIG_DIR    = os.path.join(OUT_DIR, "figures")


def run_all(eta_max=ETA_MAX):
    print("Running all three Blasius approaches...\n")

    res_bvp = solve_blasius_bvp(eta_max=eta_max)
    res_bvp.update(boundary_layer_properties(res_bvp['eta'], res_bvp['df']))

    res_ivp = solve_blasius_ivp_newton(eta_max=eta_max)
    res_ivp.update(boundary_layer_properties(res_ivp['eta'], res_ivp['df']))

    res_scratch = solve_blasius_scratch(eta_max=eta_max)

    return res_bvp, res_ivp, res_scratch


def print_table(res_bvp, res_ivp, res_scratch):
    rows = [
        ("solve_bvp",          res_bvp,     res_bvp['n_nodes'],        "nodes"),
        ("solve_ivp + Newton", res_ivp,     res_ivp['iterations'],     "iters"),
        ("Scratch RK4 + NR",   res_scratch, res_scratch['iterations'], "iters"),
    ]

    fpp = "f''(0)"
    print(f"{'Method':<22}{fpp:<16}{'Error':<12}{'Count':<12}{'Time (ms)'}")
    print("-" * 72)
    for name, res, count, label in rows:
        err = abs(res['f_double_prime_0'] - BENCHMARK)
        t   = res['elapsed_time'] * 1000
        print(f"{name:<22}{res['f_double_prime_0']:<16.10f}{err:<12.2e}{str(count)+' '+label:<12}{t:.1f}")
    print(f"{'Benchmark':<22}{BENCHMARK:<16.10f}")

    print(f"\n{'Method':<22}{'delta_99':<12}{'delta*':<12}{'theta':<12}{'H'}")
    print("-" * 58)
    for name, res, _, _ in rows:
        print(f"{name:<22}{res['delta_99']:<12.4f}{res['delta_star']:<12.4f}"
              f"{res['theta']:<12.4f}{res['shape_factor']:.4f}")
    print(f"{'Literature':<22}{LIT['delta_99']:<12}{LIT['delta_star']:<12}{LIT['theta']:<12}{LIT['H']}")


def save_latex_table(res_bvp, res_ivp, res_scratch):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "blasius_table.tex")
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n  \\centering\n")
        f.write(f"  \\caption{{Blasius base flow: comparison of the three methods ($\\eta_\\infty = {ETA_MAX:g}$). "
                f"Benchmark $f''(0) = {BENCHMARK:.10f}$.}}\n")
        f.write("  \\label{tab:blasius}\n")
        f.write("  \\begin{tabular}{lcccccc}\n    \\toprule\n")
        f.write("    Method & $f''(0)$ & Error & $\\delta_{99}$ & $\\delta^*$ & $\\theta$ & $H$ \\\\\n    \\midrule\n")
        for name, res in [
            ("\\texttt{solve\\_bvp}", res_bvp),
            ("\\texttt{solve\\_ivp} + Newton", res_ivp),
            ("Scratch RK4 + Newton", res_scratch),
        ]:
            err = abs(res['f_double_prime_0'] - BENCHMARK)
            f.write(f"    {name} & {res['f_double_prime_0']:.10f} & {err:.2e} & {res['delta_99']:.4f} & "
                    f"{res['delta_star']:.4f} & {res['theta']:.4f} & {res['shape_factor']:.4f} \\\\\n")
        f.write(f"    \\midrule\n    Literature & {BENCHMARK:.10f} & --- & {LIT['delta_99']} & "
                f"{LIT['delta_star']} & {LIT['theta']} & {LIT['H']} \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    print(f"Saved {path}")


def savefig(name):
    path = os.path.join(FIG_DIR, "blasius_" + name)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_all(res_bvp, res_ivp, res_scratch, eta_max=ETA_MAX):
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'lines.linewidth': 1.8})

    # Figure 1: f, f', f'' profiles
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, key, ylabel, title in zip(
        axes,
        ['f', 'df', 'd2f'],
        [r'$f(\eta)$', r"$f'(\eta)$", r"$f''(\eta)$"],
        [r'Streamfunction $f(\eta)$', r"Velocity $f'(\eta)$", r"Shear $f''(\eta)$"]
    ):
        ax.plot(res_bvp['eta'],     res_bvp[key],     'k-',  label='solve_bvp')
        ax.plot(res_ivp['eta'],     res_ivp[key],     'b--', label='solve_ivp + NR')
        ax.plot(res_scratch['eta'], res_scratch[key], 'r:',  label='Scratch RK4')
        ax.set_xlabel(r'$\eta$')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.4)
        ax.legend()
    axes[1].axhline(1.0, color='gray', ls='-.', alpha=0.6, label=r"$f'=1$")
    axes[1].axvline(res_scratch['delta_99'], color='purple', ls=':', alpha=0.7,
                    label=rf"$\delta_{{99}}={res_scratch['delta_99']:.2f}$")
    axes[1].legend(fontsize=9)
    axes[2].scatter([0], [res_scratch['f_double_prime_0']], color='red', zorder=5,
                    label=rf"$f''(0) = {res_scratch['f_double_prime_0']:.4f}$")
    axes[2].legend(fontsize=9)
    savefig("profiles.png")

    # Figure 2: Newton-Raphson convergence
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for res, label, style in [
        (res_ivp,     'solve_ivp + NR (FD Jacobian)',   'bo-'),
        (res_scratch, 'Scratch RK4 + NR (sensitivity)', 'rs--')
    ]:
        iters  = [r['iteration'] for r in res['history']]
        resids = [max(abs(r['residual']), 1e-16) for r in res['history']]
        ax.semilogy(iters, resids, style, label=label)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$|\Phi(s)|$')
    ax.set_title('Newton-Raphson Convergence')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    savefig("newton_convergence.png")

    # Figure 3: Pointwise method difference
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    eta_grid = np.linspace(0.0, eta_max, 500)
    df_bvp = np.interp(eta_grid, res_bvp['eta'],     res_bvp['df'])
    df_ivp = np.interp(eta_grid, res_ivp['eta'],     res_ivp['df'])
    df_sc  = np.interp(eta_grid, res_scratch['eta'], res_scratch['df'])
    ax.semilogy(eta_grid, np.abs(df_ivp - df_bvp) + 1e-16, 'b-',  label='|IVP - BVP|')
    ax.semilogy(eta_grid, np.abs(df_sc  - df_bvp) + 1e-16, 'r--', label='|Scratch - BVP|')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r"$|f'_A - f'_B|$")
    ax.set_title("Pointwise Difference in Velocity")
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    savefig("method_difference.png")

    # Figure 4: eta_inf sensitivity
    print("Running eta_inf sensitivity study...")
    eta_list = [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    bvp_err, sc_err = [], []
    for e in eta_list:
        rb = solve_blasius_bvp(eta_max=e)
        rs = solve_blasius_scratch(eta_max=e, s_init=0.33)
        bvp_err.append(max(abs(rb['f_double_prime_0'] - BENCHMARK), 1e-16))
        sc_err.append(max(abs(rs['f_double_prime_0'] - BENCHMARK), 1e-16))

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.semilogy(eta_list, bvp_err, 'ko-',  label='solve_bvp')
    ax.semilogy(eta_list, sc_err,  'r^--', label='Scratch RK4')
    ax.set_xlabel(r'$\eta_\infty$')
    ax.set_ylabel(r"$|f''(0) - f''_{\mathrm{ref}}(0)|$")
    ax.set_title(r'Effect of domain truncation $\eta_\infty$')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    savefig("eta_inf_study.png")

    # Figure 5: RK4 order of convergence
    print("Running RK4 h-refinement study...")
    h_vals = [0.4, 0.2, 0.1, 0.05, 0.025]
    errors = []
    s_ref  = solve_blasius_scratch(eta_max=eta_max, h=0.002, s_init=0.33, tol=1e-13)['f_double_prime_0']
    for h in h_vals:
        rs = solve_blasius_scratch(eta_max=eta_max, h=h, s_init=0.33, tol=1e-13)
        errors.append(max(abs(rs['f_double_prime_0'] - s_ref), 1e-16))

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    h_arr  = np.array(h_vals)
    ref_ln = errors[1] * (h_arr / h_vals[1])**4
    ax.loglog(h_arr, errors, 'rs-',  label="RK4 error in $f''(0)$")
    ax.loglog(h_arr, ref_ln, 'k--', label=r'$O(h^4)$ reference')
    ax.set_xlabel(r'Step size $h$')
    ax.set_ylabel(r"Error in $f''(0)$")
    ax.set_title('RK4 Convergence Rate Verification')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    savefig("rk4_convergence.png")

    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print("Observed RK4 orders:", np.round(rates, 2))
    print("eta_inf errors (scratch):", [f"{e:.1e}" for e in sc_err])
    print(f"Blasius figures saved to {FIG_DIR}/blasius_*.png")


def run():
    res_bvp, res_ivp, res_scratch = run_all()
    print_table(res_bvp, res_ivp, res_scratch)
    save_latex_table(res_bvp, res_ivp, res_scratch)
    plot_all(res_bvp, res_ivp, res_scratch)
    print("\nBlasius base flow done.\n")


if __name__ == '__main__':
    run()
