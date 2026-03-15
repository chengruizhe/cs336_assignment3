import json
import numpy as np
from numpy.typing import NDArray
import scipy
from pathlib import Path

import matplotlib.pyplot as plt


def load_curve_points(data_path: Path) -> list[dict[str, int | float]]:
    with data_path.open() as f:
        curve_points = json.load(f)
    return curve_points


def plot_isoflops(
    curve_points: list[dict[str, int | float]],
    output_path: Path,
) -> Path:
    """Plot ISOFLOP curves from JSON data and save the resulting figure."""

    curves_by_budget: dict[float, list[tuple[float, float]]] = {}
    for point in curve_points:
        compute_budget = float(point["compute_budget"])
        curves_by_budget.setdefault(compute_budget, []).append(
            (float(point["parameters"]), float(point["final_loss"]))
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    for compute_budget in sorted(curves_by_budget):
        points = sorted(curves_by_budget[compute_budget])
        parameters, final_losses = zip(*points)
        ax.plot(
            parameters, final_losses, marker="o", label=f"{compute_budget:.0e} FLOPs"
        )

    ax.set_xscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Final loss")
    ax.set_title("ISOFLOP Curves")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(title="Compute budget")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def optimal_model_param(
    compute_budget: float | NDArray,
    a: float,
    b: float,
) -> float | NDArray:
    return a * (compute_budget**b)


def fit_optimals(
    curve_points: list[dict[str, int | float]],
    params_output_path: Path,
    tokens_output_path: Path,
) -> NDArray:
    # find minimal loss per ISOFlop curve
    min_loss_params: dict[float, dict[str, float]] = {}
    for p in curve_points:
        compute = float(p["compute_budget"])
        if (compute not in min_loss_params) or (
            p["final_loss"] < min_loss_params[compute]["final_loss"]
        ):
            min_loss_params[compute] = {
                "parameters": float(p["parameters"]),
                "final_loss": float(p["final_loss"]),
            }

    computes = np.array(sorted(min_loss_params), dtype=float)
    params = np.array(
        [min_loss_params[compute]["parameters"] for compute in computes],
        dtype=float,
    )
    tokens = computes / (6.0 * params)

    param_popt, _ = scipy.optimize.curve_fit(
        optimal_model_param,
        computes,
        params,
        p0=(1.0, 0.5),
        maxfev=10000,
    )

    compute_grid = np.logspace(np.log10(computes.min()), 24, 1000)
    params_grid = optimal_model_param(compute_grid, *param_popt)
    tokens_grid = [c / (6.0 * p) for c, p in zip(compute_grid, params_grid)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(computes, params, color="tab:blue", s=55, label="Optimal model size")
    ax.plot(
        compute_grid,
        params_grid,
        color="tab:orange",
        linewidth=2.5,
        label="curve_fit power law",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Parameters")
    ax.set_title("Optimal Parameters vs. Compute")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(params_output_path, dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(computes, tokens, color="tab:green", s=55, label="Optimal token count")
    ax.plot(
        compute_grid,
        tokens_grid,
        color="tab:red",
        linewidth=2.5,
        label="curve_fit power law",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Tokens")
    ax.set_title("Optimal Tokens vs. Compute")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(tokens_output_path, dpi=200)
    plt.close(fig)

    return param_popt


def main():
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent
    data_path = project_root / "data" / "isoflops_curves.json"
    output_path = module_dir / "isoflops.png"
    optimal_params_output_path = module_dir / "optimal_params_fit.png"
    optimal_tokens_output_path = module_dir / "optimal_tokens_fit.png"
    curve_points = load_curve_points(data_path)

    output_path = plot_isoflops(
        curve_points=curve_points,
        output_path=output_path,
    )
    param_popt = fit_optimals(
        curve_points=curve_points,
        params_output_path=optimal_params_output_path,
        tokens_output_path=optimal_tokens_output_path,
    )
    print(f"Saved ISOFLOP plot to {output_path}")
    print(f"Saved optimal-parameter fit plot to {optimal_params_output_path}")
    print(f"Saved optimal-token fit plot to {optimal_tokens_output_path}")
    print(f"Parameter fit coefficients: {param_popt}")

    flops = [1e23, 1e24]
    opt_params = optimal_model_param(flops, *param_popt)
    opt_tokens = [c / (6.0 * p) for c, p in zip(flops, opt_params)]
    for idx, f in enumerate(flops):
        p = opt_params[idx] / 1e9
        t = opt_tokens[idx] / 1e9
        print(
            f"At {f} FLOPs, the predicted optimal number of params is {p:.3f}B,"
            f"optimal number of tokens is {t:.3f}B."
        )


if __name__ == "__main__":
    main()
