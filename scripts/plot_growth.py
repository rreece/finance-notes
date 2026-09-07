#!/usr/bin/env python3
"""
plot_growth.py
"""


import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive: no window, no Dock icon

import matplotlib.pyplot as plt


def calculate_portfolio_value(
    initial_investment,
    annual_return,
    years,
    additional_contribution=0,
    contribution_frequency="monthly",
    dividend_yield=0,
):
    """
    Calculate portfolio value over time with compound interest

    Parameters:
    initial_investment (float): Starting investment amount
    annual_return (float): Annual rate of return (as decimal, e.g., 0.08 for 8%)
    years (int): Number of years to simulate
    additional_contribution (float): Regular contribution amount
    contribution_frequency (str): Frequency of contributions ("monthly", "quarterly", "annually")
    dividend_yield (float): Annual dividend yield (as decimal)

    Returns:
    tuple: (time_points, portfolio_values)
    """
    # Determine periods per year based on contribution frequency
    if contribution_frequency == "monthly":
        periods_per_year = 12
    elif contribution_frequency == "quarterly":
        periods_per_year = 4
    else:  # annual
        periods_per_year = 1

    periods = periods_per_year * years

    # Convert annual return and dividend to the appropriate periodic rate
    periodic_return = (1 + annual_return) ** (1 / periods_per_year) - 1
    periodic_dividend = (1 + dividend_yield) ** (1 / periods_per_year) - 1
    total_periodic_return = periodic_return + periodic_dividend

    # Initialize arrays
    time_points = np.linspace(0, years, periods + 1)
    portfolio_values = np.zeros(periods + 1)
    portfolio_values[0] = initial_investment

    # Calculate portfolio value at each time point
    for i in range(1, periods + 1):
        portfolio_values[i] = portfolio_values[i - 1] * (1 + total_periodic_return)
        # Add contribution at appropriate times
        if i % (periods / (periods_per_year * years)) == 0:
            portfolio_values[i] += additional_contribution

    return time_points, portfolio_values


def calculate_contributions(
    initial_investment,
    years,
    additional_contribution=0,
    contribution_frequency="monthly",
):
    """Calculate cumulative contributions over time"""
    if contribution_frequency == "monthly":
        periods_per_year = 12
    elif contribution_frequency == "quarterly":
        periods_per_year = 4
    else:  # annual
        periods_per_year = 1

    periods = periods_per_year * years
    time_points = np.linspace(0, years, periods + 1)

    contribution_values = np.zeros(len(time_points))
    contribution_values[0] = initial_investment

    for i in range(1, len(time_points)):
        # Calculate how many contributions have been made by this time point
        contributions_made = min(i, periods)
        contribution_values[i] = initial_investment + (
            additional_contribution * contributions_made
        )

    return time_points, contribution_values


def plot_multi_cagr_portfolio_growth(
    initial_investment,
    cagr_rates,
    years,
    additional_contribution=0,
    contribution_frequency="monthly",
    inflation_rate=0,
    dividend_yield=0,
    outname="compound-growth",
):
    """
    Plot the growth of a portfolio over time with multiple CAGR rates

    Parameters:
    initial_investment (float): Starting investment amount
    cagr_rates (list): List of annual rates of return to compare (as decimals)
    years (int): Number of years to simulate
    additional_contribution (float): Regular contribution amount
    contribution_frequency (str): Frequency of contributions
    inflation_rate (float): Annual inflation rate to adjust for inflation
    dividend_yield (float): Annual dividend yield (as decimal)
    outname (str): Output filename stem for the PDF and PNG
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Set physics-style plot parameters - simplified and safer
    ax = plt.gca()
    plt.tick_params(direction="in", which="both", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Calculate and plot contribution line (same for all CAGR rates)
    time_points, contribution_values = calculate_contributions(
        initial_investment, years, additional_contribution, contribution_frequency
    )
    plt.plot(
        time_points,
        contribution_values,
        "k:",
        linewidth=1.5,
        label=f"Total Contributions (${contribution_values[-1]:,.0f})",
    )

    # Colors for the different CAGR rates
    #    colors = {'0.08': 'red', '0.10': 'green', '0.12': 'blue'}
    colors = {"0.12": "red", "0.10": "green", "0.08": "blue", "0.04": "darkgray"}

    # Line styles for different data types
    styles = {
        "nominal": "-",  # Solid line
        "dividend": "--",  # Dashed line
        "inflation": ":",  # Dotted line
    }

    # Plot for each CAGR rate
    for cagr in cagr_rates:
        cagr_str = f"{cagr:.2f}"
        color = colors.get(cagr_str, "gray")  # Default to gray if not in our color map

        # Calculate nominal portfolio values (no dividend)
        time_points, portfolio_values = calculate_portfolio_value(
            initial_investment,
            cagr,
            years,
            additional_contribution,
            contribution_frequency,
        )

        # Plot nominal values
        plt.plot(
            time_points,
            portfolio_values,
            color=color,
            linestyle=styles["nominal"],
            linewidth=2,
            label=f"CAGR {cagr*100:.0f}% (${portfolio_values[-1]:,.0f})",
        )

        # Calculate with dividends if dividend_yield is provided
        if dividend_yield > 0:
            time_points, dividend_portfolio_values = calculate_portfolio_value(
                initial_investment,
                cagr,
                years,
                additional_contribution,
                contribution_frequency,
                dividend_yield,
            )

            # Plot with dividends
            plt.plot(
                time_points,
                dividend_portfolio_values,
                color=color,
                linestyle=styles["dividend"],
                linewidth=2,
                label=f"CAGR {cagr*100:.0f}% + {dividend_yield*100:.1f}% Div (${dividend_portfolio_values[-1]:,.0f})",
            )

            # Calculate inflation-adjusted values if requested
            if inflation_rate > 0:
                inflation_factors = (1 + inflation_rate) ** -time_points
                inflation_adjusted_values = (
                    dividend_portfolio_values * inflation_factors
                )

                # Plot inflation-adjusted values
                plt.plot(
                    time_points,
                    inflation_adjusted_values,
                    color=color,
                    linestyle=styles["inflation"],
                    linewidth=2,
                    label=f"CAGR {cagr*100:.0f}% Infl-Adj (${inflation_adjusted_values[-1]:,.0f})",
                )

    # Add details to the plot
    plt.xlabel("Years Invested", fontsize=14)
    plt.ylabel("Portfolio Value", fontsize=14)
    plt.grid(True, alpha=0.3, linestyle="--")

    # Physics-style legend - outside the plot area
    plt.legend(fontsize=14, loc="upper left", framealpha=1, edgecolor="black")

    ax.set_xlim([0, years])
    ax.set_ylim([0, None])

    # Set y-axis to logarithmic scale
    #    plt.yscale('log')

    ax.tick_params(axis="both", labelsize=14)

    # Format y-axis with dollar signs and commas
    plt.gca().yaxis.set_major_formatter(
        plt.matplotlib.ticker.StrMethodFormatter("${x:,.0f}")
    )

    # Add minor ticks
    plt.minorticks_on()

    # Minor adjustments to axes - simplified
    ax = plt.gca()
    # Removed the problematic lines related to spine position and aspect ratio

    plt.tight_layout()

    plt.savefig(f"{outname}.pdf")
    plt.savefig(f"{outname}.png")
    print(f"{outname}.pdf/png written")

    return fig, ax


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot compound portfolio growth for several CAGR rates."
    )
    parser.add_argument(
        "-o", "--outname", default="compound-growth",
        help="Output filename stem (default: compound-growth).",
    )
    parser.add_argument(
        "-i", "--initial-investment", type=float, default=10000,
        help="Starting investment amount in dollars (default: 10000).",
    )
    parser.add_argument(
        "-c", "--cagr-rates", type=float, nargs="+",
        default=[0.12, 0.10, 0.08, 0.04],
        help="Annual rates of return to compare, as decimals "
             "(default: 0.12 0.10 0.08 0.04). Rates outside that set are "
             "drawn in gray.",
    )
    parser.add_argument(
        "-y", "--years", type=int, default=30,
        help="Time horizon in years (default: 30).",
    )
    parser.add_argument(
        "-m", "--contribution", type=float, default=500,
        help="Contribution added each period (default: 500).",
    )
    parser.add_argument(
        "-f", "--contribution-frequency", default="monthly",
        choices=["monthly", "quarterly", "annually"],
        help="How often the contribution is made (default: monthly).",
    )
    parser.add_argument(
        "-n", "--inflation-rate", type=float, default=0,
        help="Annual inflation rate as a decimal (default: 0). Only drawn "
             "when a dividend yield is set.",
    )
    parser.add_argument(
        "-d", "--dividend-yield", type=float, default=0.01,
        help="Annual dividend yield as a decimal (default: 0.01).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    plot_multi_cagr_portfolio_growth(
        initial_investment=args.initial_investment,
        cagr_rates=args.cagr_rates,
        years=args.years,
        additional_contribution=args.contribution,
        contribution_frequency=args.contribution_frequency,
        inflation_rate=args.inflation_rate,
        dividend_yield=args.dividend_yield,
        outname=args.outname,
    )


if __name__ == "__main__":
    main()
