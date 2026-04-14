from __future__ import annotations

from pathlib import Path

from market.market import StageOneConfig, StageOneMarket, default_hospital_profiles


def fmt(value: float | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.3f}"


def render_results_table(results: list[dict[str, float | str]]) -> str:
    columns = [
        "hospital",
        "size",
        "ownership",
        "beta_h",
        "delta_h",
        "kappa",
        "trial_cost",
        "equilibrium",
        "u_try",
        "q_l",
        "adoption_rate",
        "delay_rate",
        "avg_cycle_of_adoption",
        "avg_hospital_payoff",
        "avg_firm_payoff",
        "share_c_Y",
        "share_c_S",
    ]
    widths = {
        column: max(len(column), max(len(fmt(row[column])) for row in results))
        for column in columns
    }

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    divider = "-+-".join("-" * widths[column] for column in columns)
    lines = [header, divider]
    for row in results:
        lines.append(" | ".join(fmt(row[column]).ljust(widths[column]) for column in columns))
    return "\n".join(lines)


def print_results_table(results: list[dict[str, float | str]]) -> None:
    print(render_results_table(results))


def build_variable_descriptions() -> str:
    descriptions = {
        "hospital": "Name of the hospital profile being simulated.",
        "size": "Organizational size assumption used for that profile.",
        "ownership": "Ownership type assumption used for that profile.",
        "beta_h": "Hospital present-bias parameter. Lower values mean the hospital undervalues future benefits more strongly.",
        "delta_h": "Hospital exponential discount factor. Lower values mean the hospital is more impatient over time.",
        "kappa": "Implementation and switching friction applied to contract cost.",
        "trial_cost": "Upfront cost of choosing Try before the signal is realized.",
        "equilibrium": "Whether the simulation is using the pure-strategy case or the mixed-strategy boundary case.",
        "u_try": "The hospital's stage-1 expected utility from choosing Try. Positive means Try is attractive; negative means Delay dominates.",
        "q_l": "In the mixed boundary case, the probability that the firm offers c_S after sigma_L. It is zero in pure-strategy rows.",
        "adoption_rate": "Share of simulation runs in which the hospital eventually adopts within the allowed number of cycles.",
        "delay_rate": "Share of runs in which the hospital delays forever within the simulation horizon.",
        "avg_cycle_of_adoption": "Average cycle in which adoption happens, conditional on adoption occurring.",
        "avg_hospital_payoff": "Average realized hospital payoff from adoption across simulation runs.",
        "avg_firm_payoff": "Average realized firm profit across simulation runs.",
        "share_c_Y": "Share of runs ending with the two-part tariff c_Y being the adopted contract.",
        "share_c_S": "Share of runs ending with the SaaS contract c_S being the adopted contract.",
    }
    lines = ["Output Variables", "----------------"]
    for name, description in descriptions.items():
        lines.append(f"{name}: {description}")
    return "\n".join(lines)


def build_results_explanation(results: list[dict[str, float | str]]) -> str:
    lines = [
        "Results Interpretation",
        "----------------------",
        "The main pattern is that hospitals with lower present bias, lower friction, and lower trial cost are much more likely to adopt.",
        "",
    ]

    for row in results:
        name = str(row["hospital"])
        u_try = float(row["u_try"])
        adoption_rate = float(row["adoption_rate"])
        delay_rate = float(row["delay_rate"])
        equilibrium = str(row["equilibrium"])
        q_l = float(row["q_l"])

        if equilibrium == "pure" and u_try > 0:
            lines.append(
                f"{name}: `u_try = {u_try:.3f}` is positive, so this profile prefers Try immediately. "
                f"That produces an adoption rate of {adoption_rate:.3f} and essentially no long-run delay."
            )
        elif equilibrium == "pure":
            lines.append(
                f"{name}: `u_try = {u_try:.3f}` is negative, so Delay dominates Try in the pure-strategy case. "
                f"That is why adoption is {adoption_rate:.3f} and delay is {delay_rate:.3f}."
            )
        else:
            lines.append(
                f"{name}: this is the mixed boundary case. `q_l = {q_l:.3f}` means that after `sigma_L` the firm offers `c_S` "
                f"with probability {q_l:.3f} and `c_Y` otherwise. Adoption still occurs often, but later on average because the game mixes."
            )

    lines.extend(
        [
            "",
            "How to read the current table",
            "-----------------------------",
            "Positive `u_try` means the hospital's discounted expected gain from trying exceeds the trial cost.",
            "Negative `u_try` means the hospital keeps delaying, so the firm earns zero in those scenarios.",
            "Higher `avg_firm_payoff` tends to appear exactly where adoption happens more often.",
            "The mixed row should be read as a boundary illustration rather than a clean calibration, because your mixed-strategy section requires an extra screening-surplus assumption.",
        ]
    )
    return "\n".join(lines)


def build_report(results: list[dict[str, float | str]], config: StageOneConfig) -> str:
    sections = [
        "Stage 1 Firm-Hospital Simulation Report",
        "=======================================",
        "",
        "Simulation Settings",
        "-------------------",
        f"v_bar: {config.v_bar}",
        f"delta_v: {config.delta_v}",
        f"p_l: {config.p_l}",
        f"gamma: {config.gamma}",
        f"cost_c: {config.cost_c}",
        f"delta_m: {config.delta_m}",
        f"loops: {config.loops}",
        f"runs: {config.runs}",
        f"seed: {config.seed}",
        f"boundary_tolerance: {config.boundary_tolerance}",
        "",
        build_results_explanation(results),
        "",
        build_variable_descriptions(),
        "",
        "Simulation Output Table",
        "-----------------------",
        render_results_table(results),
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    config = StageOneConfig(
        v_bar=10.0,
        delta_v=4.0,
        p_l=0.45,
        gamma=0.7,
        cost_c=5.0,
        delta_m=0.95,
        loops=12,
        runs=1000,
        seed=7,
        boundary_tolerance=0.01,
    )

    market = StageOneMarket(config)
    profiles = default_hospital_profiles(config)

    # Calibrate the boundary profile so beta_h * delta_h ~= gamma * (1 + kappa).
    boundary = profiles[-1]
    boundary.beta_h = (config.gamma * (1.0 + boundary.kappa)) / boundary.delta_h

    results = market.compare_profiles(profiles)
    print_results_table(results)

    report = build_report(results, config)
    output_path = Path("/Volumes/T7/Thesis/stage1_simulation_report.txt")
    output_path.write_text(report, encoding="utf-8")
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()
