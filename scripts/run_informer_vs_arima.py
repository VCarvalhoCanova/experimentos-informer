"""Headless entry point for the Informer vs. ARIMA demonstration.

Same experiment as ``notebooks_examples/informer_vs_arima.ipynb`` but without
Jupyter, which is handy for long runs and for CI-like checks.

Examples:
    python scripts/run_informer_vs_arima.py --list
    python scripts/run_informer_vs_arima.py --quick
    python scripts/run_informer_vs_arima.py --datasets ETTh1.csv sunspot.txt --epochs 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment import benchmark  # noqa: E402  (path setup above)


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=None, help="dataset names (default: all eight)")
    parser.add_argument("--list", action="store_true", help="list the registered datasets and exit")
    parser.add_argument("--test-size", type=float, default=0.2, help="test fraction (default: 0.2)")
    parser.add_argument("--epochs", type=int, default=6, help="Informer epochs (default: 6)")
    parser.add_argument("--quick", action="store_true", help="small network, 3 epochs")
    parser.add_argument("--plots", action="store_true", help="also render the comparison figures (Agg backend)")
    parser.add_argument(
        "--output",
        type=Path,
        default=benchmark.RESULT_PATH / "informer_vs_arima_metrics.csv",
        help="CSV path for the metric table",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison and persist the metric table."""
    args = build_parser().parse_args(argv)

    if args.list:
        for spec in benchmark.DATASETS.values():
            print(
                f"{spec.name:20s} group={spec.group:9s} freq={spec.freq:6s} "
                f"seasonality={spec.seasonality:3d} horizon={spec.horizon:3d} "
                f"max_points={spec.max_points}"
            )
        return 0

    if args.plots:
        import matplotlib

        matplotlib.use("Agg")

    df_metrics, results = benchmark.run_comparison(
        dataset_names=args.datasets,
        test_size=args.test_size,
        epochs=args.epochs,
        quick=args.quick,
        verbose=True,
    )

    if df_metrics.empty:
        print("no model produced a forecast; check the dependencies (torch, pmdarima, sktime)")
        return 1

    report(df_metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.reset_index().to_csv(args.output, index=False)
    print(f"metric table saved to {args.output}")

    if args.plots:
        benchmark.plot_forecasts(results, tail=150)
        benchmark.plot_metric_comparison(df_metrics, "RMSE")
        print("figures rendered (Agg backend, not displayed)")

    return 0


def report(df_metrics) -> None:
    """Print the per-dataset metric table plus the RMSE/MAE comparisons."""
    import pandas as pd

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print()
    print(df_metrics.round(4))
    for metric in ("RMSE", "MAE"):
        if metric in df_metrics.columns:
            print()
            print(f"{metric} (informer_gain_% > 0 means Informer ahead)")
            print(benchmark.comparison_table(df_metrics, metric).round(3))


if __name__ == "__main__":
    raise SystemExit(main())
