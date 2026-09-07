#!/usr/bin/env python3
"""
plot_markowitz_example.py

Standalone illustrative Markowitz mean-variance (mu vs sigma) diagram.
Shows approximate positions of major asset classes, a stylized Markowitz
bullet, efficient frontier, and Capital Allocation Line.

Numbers are rough empirical estimates for illustration only.

Usage:
    plot_markowitz_example.py
    plot_markowitz_example.py --outname my_plot
    plot_markowitz_example.py --scale-factor 0.5 --risk-free-rate 0.01
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive: no window, no Dock icon

import matplotlib.pyplot as plt
from matplotlib import patheffects

EFFICIENCY_FRONTIER_COLOR = "darkgray"

# ---------------------------------------------------------------------------
# Asset class positions (illustrative, annualized, in decimal)
# ---------------------------------------------------------------------------
ASSET_CLASSES = [
    # (name,          mu,    sigma, color,            marker)
    ("Cash",          0.03,  0.002, "#A8CBA0",        "o"),
    ("Bonds",         0.05,  0.07,  "teal",           "o"),
    ("Real Estate",   0.07,  0.11,  "mediumpurple",   "o"),
    ("Value Stocks",  0.09,  0.15,  "goldenrod",      "o"),
    ("Market Index",  0.10,  0.17,  "dimgray",        "o"),
    ("Growth Stocks", 0.12,  0.26,  "lightcoral",     "o"),
    ("Inefficient",   0.06,  0.21,  "xkcd:poop",      "o"),
    ("PE / VC",       0.17,  0.45,  "xkcd:hot pink",  "o"),
]

# Risk-free rate, and the two base assets the bullet is spanned by
DEFAULT_RISK_FREE_RATE = 0.03
BULLET_BONDS = (0.05, 0.07)
BULLET_MARKET = (0.10, 0.17)
BULLET_RHO = 0.05

def markowitz_bullet(mu1, s1, mu2, s2, rho, w_range=(-2.0, 3.0), n=600):
    """Parametric Markowitz bullet for two assets with correlation rho."""
    cov12 = rho * s1 * s2
    ws = np.linspace(w_range[0], w_range[1], n)
    mus = ws * mu1 + (1 - ws) * mu2
    sigmas = np.sqrt(
        ws**2 * s1**2
        + (1 - ws)**2 * s2**2
        + 2 * ws * (1 - ws) * cov12
    )
    return mus, sigmas

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot an illustrative Markowitz mean-variance diagram."
    )
    parser.add_argument(
        "-o", "--outname", default="markowitz-example",
        help="Output filename stem (default: markowitz-example).",
    )
    parser.add_argument(
        "-s", "--scale-factor", type=float, default=1.0,
        help="Multiply every asset return by this factor (default: 1.0). "
             "Does not touch the risk-free rate -- set that with -r.",
    )
    parser.add_argument(
        "-r", "--risk-free-rate", type=float,
        default=DEFAULT_RISK_FREE_RATE,
        help=f"Risk-free rate as a decimal, the CAL's intercept "
             f"(default: {DEFAULT_RISK_FREE_RATE}).",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    scale = args.scale_factor
    r_f = args.risk_free_rate

    # Every return on the plot is scaled; the sigmas and r_f are not
    asset_classes = [
        (name, scale * mu, sigma, color, marker)
        for name, mu, sigma, color, marker in ASSET_CLASSES
    ]

    # Frontier from two base assets: Bonds and Market Index
    mu_bonds,  s_bonds  = scale * BULLET_BONDS[0],  BULLET_BONDS[1]
    mu_market, s_market = scale * BULLET_MARKET[0], BULLET_MARKET[1]

    all_mu, all_sigma = markowitz_bullet(
        mu_bonds, s_bonds, mu_market, s_market, BULLET_RHO
    )

    # Minimum variance portfolio
    idx_min = np.argmin(all_sigma)
    mu_minvar    = all_mu[idx_min]
    sigma_minvar = all_sigma[idx_min]

    # Efficient frontier: upper half of bullet
    eff_mask  = all_mu >= mu_minvar
    eff_mu    = all_mu[eff_mask]
    eff_sigma = all_sigma[eff_mask]

    # Tangent portfolio (maximum Sharpe)
    sharpe      = (all_mu - r_f) / all_sigma
    idx_tangent = np.argmax(sharpe)
    mu_tan      = all_mu[idx_tangent]
    sigma_tan   = all_sigma[idx_tangent]

    # Capital Allocation Line
    max_sigma = max(sigma for _, _, sigma, _, _ in asset_classes)
    cml_sigmas = np.array([0.0, max_sigma * 1.2])
    cml_mus    = r_f + (mu_tan - r_f) / sigma_tan * cml_sigmas

    # --- Axis limits (square, matching plot_markowitz_portfolio logic) ---
    all_mus    = [mu    for _, mu, _,     _, _ in asset_classes]
    all_sigmas = [sigma for _, _,  sigma, _, _ in asset_classes]
    max_r = max(all_mus)
    min_r = min(all_mus + [0.0])
    max_s = max(all_sigmas)
    len_s = max_s
    len_r = max_r - min_r
    if 0.77 < len_r / len_s < 1.3:
        len_x = 1.05 * max(len_s, len_r, max_r)
        xlim = (0, 100 * len_x)
        ylim = (0, 100 * len_x) if min_r >= 0 else (
            100 * (min_r - 0.025 * len_x),
            100 * (min_r - 0.025 * len_x) + 100 * len_x,
        )
    else:
        xlim = (0, 100 * 1.05 * max_s)
        ylim = (0, 100 * 1.05 * max_r) if min_r >= 0 else (
            100 * (min_r - 0.025 * len_r),
            100 * (min_r - 0.025 * len_r) + 100 * 1.05 * len_r,
        )

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set physics-style plot parameters - simplified and safer
    plt.tick_params(direction="in", which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Full Markowitz bullet (opportunity set)
    ax.plot(
        100 * all_sigma, 100 * all_mu,
        color="lightgray", linewidth=1.5, zorder=1,
#        label="Opportunity Set",
    )

    # Efficient frontier (with ticked stroke matching plot_markowitz_portfolio)
    ax.plot(
        100 * eff_sigma, 100 * eff_mu,
        color=EFFICIENCY_FRONTIER_COLOR,
        linewidth=2,
        zorder=2,
        label="Efficient Frontier",
        path_effects=[
            patheffects.withTickedStroke(spacing=18, angle=90, length=0.4)
        ],
    )

    # Capital Allocation Line
    ax.plot(
        100 * cml_sigmas, 100 * cml_mus,
        color="black", linewidth=1.5, linestyle="dashed", zorder=2,
        label="Capital Allocation Line",
    )

    # Minimum variance portfolio
    ax.plot(
        100 * sigma_minvar, 100 * mu_minvar,
        color=EFFICIENCY_FRONTIER_COLOR,
        marker="D", markersize=8, linestyle="None",
        markeredgewidth=2,
        zorder=4,
        label="Min Var Portfolio",
    )

    # Tangent portfolio
    ax.plot(
        100 * sigma_tan, 100 * mu_tan,
        color="#1f77b4",
        marker="D", markersize=8, linestyle="None",
        zorder=4,
        label="Tangent Portfolio",
    )

    # Example portfolio: 70% Tangent + 30% Cash, lies on the CML
    w_example    = 0.7
    mu_example   = r_f + w_example * (mu_tan - r_f)
    sigma_example = w_example * sigma_tan
    ax.plot(
        100 * sigma_example, 100 * mu_example,
        color="black",
        marker="D", markersize=8, linestyle="None",
        zorder=5,
        label="Example Portfolio",
    )

    # Asset class markers -- named by the annotation, not by a legend entry
    for name, mu, sigma, color, marker in asset_classes:
        ax.plot(
            100 * sigma, 100 * mu,
            color=color, marker=marker, markersize=8,
            linestyle="None",
        )
        ax.annotate(
            name,
            xy=(100 * sigma, 100 * mu),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=11,
        )

    # Add details to the plot
    ax.set_xlabel("Standard Deviation [%]", fontsize=14)
    ax.set_ylabel("Return [%]", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="--")

    # Physics-style legend
    plt.legend(fontsize=14, loc="upper left", framealpha=1, edgecolor="black")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.tick_params(axis="both", labelsize=14)

    # Add minor ticks
    plt.minorticks_on()

    plt.tight_layout()
    plt.savefig(f"{args.outname}.pdf")
    plt.savefig(f"{args.outname}.png")
    print(f"{args.outname}.pdf/png written")

if __name__ == "__main__":
    main()
