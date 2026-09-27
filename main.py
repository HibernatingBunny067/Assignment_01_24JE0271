# Assignment 01 - Main comparison script
# Run this from the project root: python main.py

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.solve_bvp_approach import solve_falkner_skan_bvp
from src.solve_ivp_newton   import solve_falkner_skan_ivp_newton
from src.scratch_rk4_newton import solve_falkner_skan_scratch, boundary_layer_properties

BENCHMARK  = 1.232587656817
OUT_DIR    = "latex_report"
FIG_DIR    = os.path.join(OUT_DIR, "figures")


def run_all(eta_max=8.0):
    print("Running all three approaches...\n")

    res_bvp = solve_falkner_skan_bvp(eta_max=eta_max)
    res_bvp.update(boundary_layer_properties(res_bvp['eta'], res_bvp['df']))

    res_ivp = solve_falkner_skan_ivp_newton(eta_max=eta_max)
    res_ivp.update(boundary_layer_properties(res_ivp['eta'], res_ivp['df']))

    res_scratch = solve_falkner_skan_scratch(eta_max=eta_max)

    return res_bvp, res_ivp, res_scratch


def print_table(res_bvp, res_ivp, res_scratch):
    rows = [
        ("solve_bvp",          res_bvp,     res_bvp['n_nodes'],       "nodes"),
        ("solve_ivp + Newton", res_ivp,     res_ivp['iterations'],    "iters"),
        ("Scratch RK4 + NR",   res_scratch, res_scratch['iterations'], "iters"),
    ]

    print(f"{'Method':<22}{'f\'\'(0)':<16}{'Error':<12}{'Count':<12}{'Time (ms)'}")
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
    print(f"{'Literature':<22}{'~2.38':<12}{'0.6479':<12}{'0.2923':<12}2.216")


def save_latex_table(res_bvp, res_ivp, res_scratch):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "comparison_table.tex")
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n  \\centering\n")
        f.write("  \\caption{Comparison of the three methods. Benchmark $f''(0) = 1.2325876568$.}\n")
        f.write("  \\label{tab:comparison}\n")
        f.write("  \\begin{tabular}{lccccc}\n    \\toprule\n")
        f.write("    Method & $f''(0)$ & Error & $\\delta^*$ & $\\theta$ & $H$ \\\\\n    \\midrule\n")
        for name, res in [
            ("\\texttt{solve\\_bvp}", res_bvp),
            ("\\texttt{solve\\_ivp} + Newton", res_ivp),
            ("Scratch RK4 + Newton", res_scratch),
        ]:
            err = abs(res['f_double_prime_0'] - BENCHMARK)
            f.write(f"    {name} & {res['f_double_prime_0']:.8f} & {err:.2e} & "
                    f"{res['delta_star']:.4f} & {res['theta']:.4f} & {res['shape_factor']:.4f} \\\\\n")
        f.write(f"    \\midrule\n    Literature & {BENCHMARK:.8f} & --- & 0.6479 & 0.2923 & 2.216 \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    print(f"Saved {path}")


def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.savefig(path, dpi=200)
    plt.close()


def plot_all(res_bvp, res_ivp, res_scratch):
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
        (res_ivp,     'solve_ivp + NR', 'bo-'),
        (res_scratch, 'Scratch RK4',    'rs--')
    ]:
        iters  = [r['iteration'] for r in res['history']]
        resids = [abs(r['residual']) for r in res['history']]
        ax.semilogy(iters, resids, style, label=label)
    ax.axhline(1e-6, color='k', ls=':', label='tol')
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$|\Phi(s)|$')
    ax.set_title('Newton-Raphson Convergence')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    savefig("newton_convergence.png")

    # Figure 3: Pointwise method difference
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    eta_grid = np.linspace(0.0, 8.0, 500)
    df_bvp = np.interp(eta_grid, res_bvp['eta'],     res_bvp['df'])
    df_ivp = np.interp(eta_grid, res_ivp['eta'],     res_ivp['df'])
    df_sc  = np.interp(eta_grid, res_scratch['eta'], res_scratch['df'])
    ax.semilogy(eta_grid, np.abs(df_ivp - df_bvp), 'b-',  label='|IVP - BVP|')
    ax.semilogy(eta_grid, np.abs(df_sc  - df_bvp), 'r--', label='|Scratch - BVP|')
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r"$|f'_A - f'_B|$")
    ax.set_title("Pointwise Difference in Velocity")
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    savefig("method_difference.png")

    # Figure 4: eta_inf sensitivity
    print("Running eta_inf sensitivity study...")
    eta_list, bvp_vals, sc_vals = [], [], []
    for e in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]:
        rb = solve_falkner_skan_bvp(eta_max=e)
        rs = solve_falkner_skan_scratch(eta_max=e, s_init=1.23)
        eta_list.append(e)
        bvp_vals.append(rb['f_double_prime_0'])
        sc_vals.append(rs['f_double_prime_0'])

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(eta_list, bvp_vals, 'ko-',  label='solve_bvp')
    ax.plot(eta_list, sc_vals,  'r^--', label='Scratch RK4')
    ax.axhline(BENCHMARK, color='blue', ls=':', label='Benchmark')
    ax.set_xlabel(r'$\eta_\infty$')
    ax.set_ylabel(r"$f''(0)$")
    ax.set_title(r'Effect of domain truncation $\eta_\infty$')
    ax.legend()
    ax.grid(True, alpha=0.4)
    savefig("eta_inf_study.png")

    # Figure 5: RK4 order of convergence
    print("Running RK4 h-refinement study...")
    h_vals = [0.2, 0.1, 0.05, 0.025, 0.0125]
    errors = []
    s_ref  = solve_falkner_skan_scratch(eta_max=6.0, h=0.002, s_init=1.23, tol=1e-11)['f_double_prime_0']
    for h in h_vals:
        rs = solve_falkner_skan_scratch(eta_max=6.0, h=h, s_init=1.23, tol=1e-10)
        errors.append(max(abs(rs['f_double_prime_0'] - s_ref), 1e-15))

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

    print(f"All figures saved to {FIG_DIR}/")


if __name__ == '__main__':
    res_bvp, res_ivp, res_scratch = run_all()
    print_table(res_bvp, res_ivp, res_scratch)
    save_latex_table(res_bvp, res_ivp, res_scratch)
    plot_all(res_bvp, res_ivp, res_scratch)
    print(f"\nDone. Upload {OUT_DIR}/ to Overleaf to compile the report.")
