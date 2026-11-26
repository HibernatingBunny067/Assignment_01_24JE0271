# Orr-Sommerfeld stability of the Blasius boundary layer: driver.
# Stage 1: Blasius base flow with three methods (src/blasius_case.py).
# Stage 2: validates the OS shooting solver, sweeps the (Re, alpha) plane, extracts the
#          neutral curve and the critical point.
# Everything is written to latex_report/Orr_Sommerfeld/ (one combined report).
#
# Usage (from the project root):  python main.py orr-sommerfeld [--quick]

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src import blasius_case
from src.orr_sommerfeld import (BlasiusProfile, OrrSommerfeldShooter, sweep_grid,
                                neutral_curve_from_grid, critical_point)

OUT_DIR = os.path.join("latex_report", "Orr_Sommerfeld")
FIG_DIR = os.path.join(OUT_DIR, "figures")

# Reference values (parallel-flow Blasius, Re and alpha based on delta*)
C_BENCH  = 0.36412286 + 0.00795972j      # Jordinson (1970): Re_l = 580, alpha_l = 0.179 (Blasius length)
LIT_CRIT = {'Re': 520.0, 'alpha': 0.30, 'c_r': 0.40}   # approx., e.g. Schlichting & Gersten


def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, name), dpi=200)
    plt.close()


def validate(profile, y_inf):
    """Benchmark eigenvalue: Newton history and RK4 step-size convergence."""
    print("Validating against the benchmark eigenvalue...")
    # benchmark defined in Blasius-length units (Re_l = 580, alpha_l = 0.179) -> delta* units
    a, R = 0.179 * profile.delta_star, 580.0 * profile.delta_star
    sh = OrrSommerfeldShooter(profile, y_inf=y_inf, n_steps=2000)
    c, ok, hist = sh.solve_c(a, R, 0.36 + 0.01j, return_history=True)

    n_list = [250, 500, 1000, 2000, 4000]
    c_list = []
    for n in n_list:
        s = OrrSommerfeldShooter(profile, y_inf=y_inf, n_steps=n)
        c_list.append(s.solve_c(a, R, c[0])[0][0])
    c_ref = OrrSommerfeldShooter(profile, y_inf=y_inf, n_steps=8000).solve_c(a, R, c[0])[0][0]
    err = np.abs(np.array(c_list) - c_ref)

    y_list = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
    c_y = [OrrSommerfeldShooter(profile, y_inf=y, n_steps=int(250*y)).solve_c(a, R, c[0])[0][0]
           for y in y_list]

    print(f"  c = {c[0].real:.8f} {c[0].imag:+.8f}i  (reference {C_BENCH.real:.8f} {C_BENCH.imag:+.8f}i)")
    print(f"  |c - c_ref| = {abs(c[0]-C_BENCH):.2e},  Newton iterations = {len(hist)}")
    return {'c': c[0], 'alpha': a, 'Re': R, 'hist': hist, 'n_list': n_list, 'err': err,
            'y_list': y_list, 'c_y': np.array(c_y)}


def plot_base_flow(profile):
    y = np.linspace(0, 8, 600)
    U, dU, d2U = profile(y)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, q, lab in zip(axes, [U, dU, d2U], [r'$U$', r"$U'$", r"$U''$"]):
        ax.plot(q, y, 'k-')
        ax.set_xlabel(lab)
        ax.set_ylabel(r'$y/\delta^*$')
        ax.grid(True, alpha=0.4)
    axes[0].set_title('Blasius velocity')
    axes[1].set_title('Shear')
    axes[2].set_title(r"Curvature ($U''(0)=0$: no inflection point)")
    savefig("base_flow.png")


def plot_validation(val):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.semilogy(range(1, len(val['hist']) + 1), [h['dc'][0] for h in val['hist']], 'bo-', label=r'$|\Delta c|$')
    ax.semilogy(range(1, len(val['hist']) + 1), [h['D'][0] for h in val['hist']], 'rs--', label=r'$|D(c)|$')
    ax.set_xlabel('Newton iteration')
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title(r'Newton on $D(c)=0$ (benchmark point)')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)

    ax = axes[1]
    h = 8.0 / np.array(val['n_list'])
    ax.loglog(h, val['err'], 'rs-', label='RK4 error in $c$')
    ax.loglog(h, val['err'][2] * (h / h[2])**4, 'k--', label=r'$O(h^4)$')
    ax.set_xlabel(r'step size $h$ ($\delta^*$ units)')
    ax.set_ylabel(r'$|c_h - c_{\mathrm{ref}}|$')
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title('Step-size convergence')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)

    ax = axes[2]
    ax.semilogy(val['y_list'], np.abs(val['c_y'] - val['c_y'][-1]) + 1e-16, 'ko-')
    ax.set_xlabel(r'$y_\infty/\delta^*$ (start of the wall-ward integration)')
    ax.set_ylabel(r'$|c - c(y_\infty = 12)|$')
    ax.set_title(r'Sensitivity to $y_\infty$')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    savefig("os_validation.png")


def split_branches(neutral, crit):
    """
    The refined points are ordered along the contour, which runs from one large-Re end,
    round the nose, to the other. Split at the nose (minimum Re); the side with the larger
    alpha at a common Re is the upper branch.
    """
    a, R, cr = neutral['alpha'], neutral['Re'], neutral['c_r']
    k = int(np.argmin(R))
    sides = [np.arange(0, k + 1), np.arange(k, len(R))]
    branches = []
    for idx in sides:
        o = idx[np.argsort(R[idx])]
        br = {'alpha': a[o], 'Re': R[o], 'c_r': cr[o]}
        keep = br['Re'] > crit['Re']                   # replace the nose point by the exact critical point
        for key in br:
            br[key] = np.concatenate([[crit[key]], br[key][keep]])
        branches.append(br)
    R_common = min(branches[0]['Re'].max(), branches[1]['Re'].max())
    a_at = [np.interp(R_common, b['Re'], b['alpha']) for b in branches]
    lower, upper = (branches[0], branches[1]) if a_at[0] < a_at[1] else (branches[1], branches[0])
    return lower, upper


def plot_neutral(Re_vals, alpha_vals, C, lower, upper, crit):
    # c_i map with neutral curve
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ci = C.imag
    levels = np.linspace(-0.06, 0.03, 19)
    cs = ax.contourf(Re_vals, alpha_vals, np.clip(ci, -0.06, 0.03), levels=levels, cmap='RdBu_r',
                     vmin=-0.03, vmax=0.03)
    fig.colorbar(cs, ax=ax, label=r'$c_i$ (TS mode)')
    for br in (lower, upper):
        ax.plot(br['Re'], br['alpha'], 'k-', lw=2)
    ax.plot(crit['Re'], crit['alpha'], 'y*', ms=14, mec='k', label=rf"$Re_c = {crit['Re']:.1f}$")
    ax.set_xscale('log')
    ax.set_xlabel(r'$Re_{\delta^*}$')
    ax.set_ylabel(r'$\alpha\delta^*$')
    ax.set_title(r'Temporal growth rate $c_i(Re,\alpha)$ and neutral curve $c_i=0$')
    ax.legend(loc='upper right')
    savefig("ci_map.png")

    # neutral curve on its own (linear and log Re)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, logx in zip(axes, (False, True)):
        ax.plot(lower['Re'], lower['alpha'], 'b-', lw=2, label='lower branch')
        ax.plot(upper['Re'], upper['alpha'], 'r-', lw=2, label='upper branch')
        ax.plot(crit['Re'], crit['alpha'], 'k*', ms=13,
                label=rf"critical: $Re_c={crit['Re']:.1f}$, $\alpha_c={crit['alpha']:.4f}$")
        ax.plot(LIT_CRIT['Re'], LIT_CRIT['alpha'], 'go', mfc='none', ms=12, label=r'literature $\approx(520,\,0.30)$')
        ax.set_xlabel(r'$Re_{\delta^*}$')
        ax.set_ylabel(r'$\alpha\delta^*$')
        ax.grid(True, which='both', ls='--', alpha=0.5)
        if logx:
            ax.set_xscale('log')
            ax.set_title('Neutral curve (log scale)')
        else:
            ax.set_xlim(0, 10000)
            ax.set_title('Neutral curve, Blasius (parallel flow)')
            ax.text(3000, 0.17, 'UNSTABLE', color='gray')
        ax.legend(fontsize=9)
    savefig("neutral_curve.png")

    # phase speed along the neutral curve
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.semilogx(lower['Re'], lower['c_r'], 'b-', lw=2, label='lower branch')
    ax.semilogx(upper['Re'], upper['c_r'], 'r-', lw=2, label='upper branch')
    ax.semilogx(crit['Re'], crit['c_r'], 'k*', ms=12, label=rf"$c_r = {crit['c_r']:.4f}$ at $Re_c$")
    ax.set_xlabel(r'$Re_{\delta^*}$')
    ax.set_ylabel(r'$c_r$')
    ax.set_title('Phase speed of neutral TS waves')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    savefig("neutral_phase_speed.png")


def save_tables(val, crit, lower, upper, run_info):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "results_table.tex")
    c = val['c']
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n  \\centering\n")
        f.write("  \\caption{Validation eigenvalue and critical point of the Blasius neutral curve "
                "($Re$ and $\\alpha$ based on $\\delta^*$).}\n  \\label{tab:os_results}\n")
        f.write("  \\begin{tabular}{lccc}\n    \\toprule\n")
        f.write("    Quantity & Present & Reference & Difference \\\\\n    \\midrule\n")
        pt = f"$(Re,\\alpha)=({val['Re']:.2f},\\,{val['alpha']:.5f})$"
        f.write(f"    $c_r$ at {pt} & {c.real:.8f} & {C_BENCH.real:.8f} & {abs(c.real-C_BENCH.real):.1e} \\\\\n")
        f.write(f"    $c_i$ at {pt} & {c.imag:.8f} & {C_BENCH.imag:.8f} & {abs(c.imag-C_BENCH.imag):.1e} \\\\\n")
        f.write("    \\midrule\n")
        f.write(f"    Critical $Re_{{\\delta^*}}$ & {crit['Re']:.2f} & $\\approx {LIT_CRIT['Re']:.0f}$ & --- \\\\\n")
        f.write(f"    Critical $\\alpha\\delta^*$ & {crit['alpha']:.4f} & $\\approx {LIT_CRIT['alpha']:.2f}$ & --- \\\\\n")
        f.write(f"    Critical $c_r$ & {crit['c_r']:.4f} & $\\approx {LIT_CRIT['c_r']:.2f}$ & --- \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    print(f"Saved {path}")

    path = os.path.join(OUT_DIR, "neutral_points_table.tex")
    targets = [600, 1000, 2000, 5000, 10000]
    with open(path, "w") as f:
        f.write("\\begin{table}[htbp]\n  \\centering\n")
        f.write("  \\caption{Neutral wavenumbers and phase speeds on the two branches "
                "(interpolated from the refined neutral curve).}\n  \\label{tab:neutral_points}\n")
        f.write("  \\begin{tabular}{lcccc}\n    \\toprule\n")
        f.write("    $Re_{\\delta^*}$ & $\\alpha\\delta^*$ (lower) & $c_r$ (lower) & "
                "$\\alpha\\delta^*$ (upper) & $c_r$ (upper) \\\\\n    \\midrule\n")
        for R in targets:
            def at(br, key):
                if R < br['Re'].min() or R > br['Re'].max():
                    return '---'
                return f"{np.interp(np.log(R), np.log(br['Re']), br[key]):.4f}"
            f.write(f"    {R} & {at(lower,'alpha')} & {at(lower,'c_r')} & {at(upper,'alpha')} & {at(upper,'c_r')} \\\\\n")
        f.write("    \\bottomrule\n  \\end{tabular}\n\\end{table}\n")
    print(f"Saved {path}")

    path = os.path.join(OUT_DIR, "run_parameters.tex")
    with open(path, "w") as f:
        f.write("% auto-generated by main.py orr-sommerfeld\n")
        for k, v in run_info.items():
            f.write(f"\\newcommand{{\\os{k}}}{{{v}}}\n")
    print(f"Saved {path}")

    path = os.path.join(OUT_DIR, "neutral_curve.csv")
    with open(path, "w") as f:
        f.write("branch,Re_delta_star,alpha_delta_star,c_r\n")
        for name, br in (('lower', lower), ('upper', upper)):
            for R, a, cr in zip(br['Re'], br['alpha'], br['c_r']):
                f.write(f"{name},{R:.6f},{a:.8f},{cr:.8f}\n")
    print(f"Saved {path}")


def run(quick=False):
    print("-" * 20, "Stage 1: Blasius base flow", "-" * 20)
    blasius_case.run()

    print("-" * 20, "Stage 2: Orr-Sommerfeld equation", "-" * 20)
    t0 = time.perf_counter()
    os.makedirs(FIG_DIR, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'font.family': 'serif', 'lines.linewidth': 1.8})

    y_inf   = 8.0
    n_steps = 1000 if quick else 1500
    nR, nA  = (25, 25) if quick else (41, 41)
    Re_vals    = np.logspace(np.log10(200.0), np.log10(2.0e4), nR)
    alpha_vals = np.linspace(0.02, 0.42, nA)

    profile = BlasiusProfile()
    print(f"Blasius profile: f''(0) = {profile.f_double_prime_0:.10f}, "
          f"delta*_eta = {profile.delta_star:.8f}")
    plot_base_flow(profile)

    val = validate(profile, y_inf)
    plot_validation(val)

    shooter = OrrSommerfeldShooter(profile, y_inf=y_inf, n_steps=n_steps)
    print(f"Sweeping {nA} x {nR} (alpha, Re) grid, RK4 steps = {n_steps}, y_inf = {y_inf} delta* ...")
    C = sweep_grid(shooter, Re_vals, alpha_vals, n_sub=3, verbose=False)
    print(f"  grid done ({time.perf_counter()-t0:.0f} s), max c_i = {np.nanmax(C.imag):.5f}")

    neutral = neutral_curve_from_grid(shooter, Re_vals, alpha_vals, C)
    print(f"  neutral curve: {len(neutral['Re'])} refined points ({neutral['n_failed']} failed)")
    crit = critical_point(shooter, neutral)
    print(f"  critical point: Re_c = {crit['Re']:.2f}, alpha_c = {crit['alpha']:.4f}, c_r = {crit['c_r']:.4f}")

    lower, upper = split_branches(neutral, crit)
    plot_neutral(Re_vals, alpha_vals, C, lower, upper, crit)

    run_info = {'Yinf': f"{y_inf:g}", 'Nsteps': n_steps, 'NRe': nR, 'NAlpha': nA,
                'ReMin': f"{Re_vals[0]:g}", 'ReMax': f"{Re_vals[-1]:g}",
                'AlphaMin': f"{alpha_vals[0]:g}", 'AlphaMax': f"{alpha_vals[-1]:g}",
                'Rec': f"{crit['Re']:.2f}", 'Alphac': f"{crit['alpha']:.4f}", 'Crc': f"{crit['c_r']:.4f}",
                'Runtime': f"{time.perf_counter()-t0:.0f}"}
    save_tables(val, crit, lower, upper, run_info)
    print(f"All figures saved to {FIG_DIR}/  (total {time.perf_counter()-t0:.0f} s)")


if __name__ == '__main__':
    run()
