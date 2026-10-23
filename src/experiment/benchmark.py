"""Experiment helpers for the Informer vs. ARIMA demonstration.

The module contains the pieces that the demonstration notebook
(``notebooks_examples/informer_vs_arima.ipynb``) needs but that do not belong
to a model itself:

* :data:`DATASETS` -- registry of the eight raw series shipped in ``data/raw``
  (frequency, seasonal period, default forecast horizon, ``max_points`` used to
  keep the demo tractable on a CPU, and whether ARIMA should be estimated with
  a seasonal component).
* :func:`load_series` -- loads a series as a ``pandas.Series`` with a
  ``DatetimeIndex`` (the ETT files already carry a ``date`` column, the ``.txt``
  files are indexed with the frequency declared in the registry).
* :func:`split_series` -- train/test split.
* :func:`run_arima` / :func:`run_informer` -- fit, forecast and score one model.
* :func:`run_comparison` -- runs both models on the selected datasets and
  returns a metric table in the *same format* produced by
  ``metrics.open_fold_result`` (columns ``MSE, RMSE, MAPE, SMAPE, MAE, theil,
  ARV, IA, POCID, time_testing, time_training``).
* :func:`plot_forecasts` / :func:`plot_metric_comparison` -- quick figures for
  the notebook.

Evaluation protocol: both models forecast the test set in blocks of
``horizon`` steps from block origins spaced by ``horizon``. ARIMA predicts the
whole block and is then updated with the observed values; Informer performs a
single generative forward pass per block (the decoder of the paper writes the
whole horizon at once). That makes the comparison symmetric: both models are
scored on multi-step forecasts, never on a one-step-ahead shortcut.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

#: Project root resolved from this file (``src/experiment/benchmark.py``). Resolving it
#: locally keeps the helpers usable regardless of ``sys.path``/installation: the result
#: matches ``config.RAW_DATA_PATH`` and ``config.MODEL_DATA_PATH``.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
RESULT_PATH = PROJECT_ROOT / "data" / "result"


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    """Description of one raw series and of the demo settings used with it."""

    name: str
    target: str
    freq: str
    """Pandas frequency alias used to rebuild the index of the ``.txt`` files."""
    seasonality: int
    """Seasonal period (hours per day, months per year, ...)."""
    horizon: int
    """Default multi-step forecast horizon (in points)."""
    seq_len: int
    label_len: int
    arima_seasonal: bool
    """Whether ``pmdarima`` should estimate a seasonal component."""
    max_points: Optional[int] = None
    """Number of most recent points kept, ``None`` keeps the whole series."""
    group: str = ""

    @property
    def arima_periods(self) -> int:
        """Seasonal period passed to the ARIMA model (1 disables seasonality)."""
        return self.seasonality if self.arima_seasonal else 1


DATASETS: Dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        # ETT: oil temperature target of the paper, hourly.
        DatasetSpec("ETTh1.csv", "OT", "h", 24, 24, 96, 48, False, 1200, "ETT"),
        DatasetSpec("ETTh2.csv", "OT", "h", 24, 24, 96, 48, False, 1200, "ETT"),
        # ETT: 15 minutes granularity, the paper's target is still "OT".
        DatasetSpec("ETTm1.csv", "OT", "15min", 96, 48, 96, 48, False, 1200, "ETT"),
        DatasetSpec("ETTm2.csv", "OT", "15min", 96, 48, 96, 48, False, 1200, "ETT"),
        # Classical single series (no timestamps in the files).
        DatasetSpec("coloradoRiver.txt", "y", "MS", 12, 12, 36, 18, True, None, "Classical"),
        DatasetSpec("sunspot.txt", "y", "YE", 11, 11, 36, 18, True, None, "Classical"),
        DatasetSpec("taylor.txt", "y", "30min", 48, 48, 96, 48, False, 1200, "Classical"),
        DatasetSpec("temperature.txt", "y", "MS", 12, 12, 36, 18, True, None, "Classical"),
    ]
}


def available_datasets() -> List[str]:
    """Return the names of the eight registered datasets."""
    return list(DATASETS)


def load_series(name: str, max_points: Optional[int] = None) -> pd.Series:
    """Load one raw series with a ``DatetimeIndex``.

    Args:
        name (str): Dataset name as listed in :data:`DATASETS`.
        max_points (Optional[int]): Keep only the ``max_points`` most recent
            observations (``None`` uses the registry value).

    Returns:
        pd.Series: The target series, float dtype, named after the dataset.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; options: {available_datasets()}")

    spec = DATASETS[name]
    path = RAW_DATA_PATH / name
    frame = pd.read_csv(path)

    if "date" in frame.columns:
        index = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
        values = frame[spec.target].astype(float)
    else:
        # The .txt files are single column ("y") without time stamps: rebuild
        # the index from the frequency declared in the registry.
        index = pd.date_range("1900-01-01", periods=len(frame), freq=spec.freq)
        values = frame["y"].astype(float)

    tail = spec.max_points if max_points is None else max_points
    if tail is not None and tail < len(values):
        values = values.iloc[-tail:]
        index = index[-tail:]

    series = pd.Series(values.to_numpy(), index=index, name=name)
    return series


def split_series(
    series: pd.Series, test_size: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    """Split a series into train and test, preserving the time order."""
    n_test = max(1, int(round(len(series) * test_size)))
    return series.iloc[:-n_test], series.iloc[-n_test:]


def split_validation(
    train: pd.Series, val_size: float = 0.15
) -> Tuple[pd.Series, pd.Series]:
    """Split the training window into an inner train and a validation tail.

    Used by the hyperparameter search in
    ``notebooks_examples/informer_vs_arima_optimized.ipynb``: the search only
    ever sees ``(inner_train, validation)``, so the test window stays untouched
    until the final evaluation.

    Args:
        train (pd.Series): Training observations (already split from the test
            set with :func:`split_series`).
        val_size (float, optional): Fraction of ``train`` held out for
            validation. Defaults to 0.15.

    Returns:
        Tuple[pd.Series, pd.Series]: ``(inner_train, validation)``, in time
        order, with at least one point in each part.
    """
    n_val = int(max(1, min(len(train) - 2, round(len(train) * val_size))))
    return train.iloc[:-n_val], train.iloc[-n_val:]


def effective_horizon(spec: DatasetSpec, n_test: int) -> int:
    """Shrink the horizon when the test window is too short for the blocks."""
    return int(max(1, min(spec.horizon, n_test // 3)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Score a forecast with the project's metric module.

    The project's ``src/metrics/metrics.py`` is used on purpose: the demonstration
    then reports exactly the same nine metrics as
    ``notebooks_examples/calculate_metrics.ipynb``.

    Args:
        y_true (np.ndarray): Observed values.
        y_pred (np.ndarray): Forecast values.

    Returns:
        Dict[str, float]: ``MSE, RMSE, MAPE, SMAPE, MAE, theil, ARV, IA, POCID``.
    """
    from metrics import metrics  # local import: keeps the module importable alone

    return metrics.gerenerate_metric_results(y_true, y_pred)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def run_arima(
    train: pd.Series,
    test: pd.Series,
    horizon: int,
    seasonal_periods: int = 1,
    verbose: bool = True,
) -> Dict[str, object]:
    """Fit the project ARIMA wrapper and forecast the test set in blocks.

    Args:
        train (pd.Series): Training observations.
        test (pd.Series): Test observations.
        horizon (int): Number of steps written per block.
        seasonal_periods (int, optional): Seasonal period (1 = non seasonal).
        verbose (bool, optional): Let ``pmdarima``/``auto_arima`` print.

    Returns:
        Dict[str, object]: ``test_pred``, ``test_true``, metrics and timings.
    """
    from model.arima import Arima  # project model: pmdarima auto_arima wrapper

    model = Arima(seazonal_lag=seasonal_periods, horizon=horizon)

    start = time.time()
    model.fit(train.to_numpy(dtype=float))
    training_time = time.time() - start

    forecaster = model.forecaster
    predictions: List[np.ndarray] = []

    start = time.time()
    for origin in range(0, len(test) - horizon + 1, horizon):
        block = np.asarray(forecaster.predict(horizon), dtype=float)
        predictions.append(block)
        # Rolling origin: feed the observed block back to the model.
        for value in test.iloc[origin : origin + horizon].to_numpy(dtype=float):
            forecaster.update(value)
    testing_time = time.time() - start

    pred = np.concatenate(predictions) if predictions else np.empty(0)
    true = test.to_numpy(dtype=float)[: len(pred)]

    if verbose:
        try:
            order = forecaster.order
            seasonal_order = forecaster.seasonal_order
            print(f"    ARIMA{order}x{seasonal_order} fitted")
        except Exception:  # pragma: no cover - only informational
            pass

    return {
        "model": "arima",
        "ts": train.name,
        "test_predict": pred,
        "test_true": true,
        "metrics": compute_metrics(true, pred) if len(pred) else {},
        "time_training": training_time,
        "time_testing": testing_time,
    }


#: Hyperparameters of :class:`model.informer.InformerForecaster` that
#: :func:`run_informer` accepts through ``params`` (besides ``seq_len`` and
#: ``label_len``, which are clamped to the length of the training window and to
#: ``pred_len = horizon``).
INFORMER_PARAM_KEYS = (
    "d_model",
    "n_heads",
    "e_layers",
    "d_layers",
    "d_ff",
    "factor",
    "dropout",
    "distil",
    "n_encoder_stacks",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "patience",
    "val_ratio",
    "seed",
    "device",
)


def run_informer(
    train: pd.Series,
    test: pd.Series,
    horizon: int,
    spec: DatasetSpec,
    epochs: int = 6,
    batch_size: int = 64,
    quick: bool = False,
    verbose: bool = True,
    params: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Fit the Informer forecaster and forecast the test set in blocks.

    Args:
        train (pd.Series): Training observations.
        test (pd.Series): Test observations.
        horizon (int): Number of steps written per block.
        spec (DatasetSpec): Dataset settings (sequence lengths mainly).
        epochs (int, optional): Maximum number of epochs. Defaults to 6.
        batch_size (int, optional): Batch size. Defaults to 64.
        quick (bool, optional): Use a small network for a fast demonstration.
        verbose (bool, optional): Print the training progress.
        params (Optional[Dict[str, object]], optional): Overrides for the
            hyperparameters of :class:`InformerForecaster`. Accepted keys are
            :data:`INFORMER_PARAM_KEYS` plus ``seq_len`` and ``label_len``
            (both clamped to what the training window supports). ``pred_len``
            cannot be overridden: the generative decoder must write exactly
            ``horizon`` steps. ``None`` (default) keeps the original, small
            demonstration configuration.

    Returns:
        Dict[str, object]: ``test_pred``, ``test_true``, metrics and timings.
    """
    from model.informer import InformerForecaster

    config: Dict[str, object] = {
        "d_model": 32 if quick else 48,
        "n_heads": 2 if quick else 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 64 if quick else 96,
        "factor": 5,
        "dropout": 0.1,
        "distil": True,
        "epochs": 3 if quick else epochs,
        "batch_size": batch_size,
        "lr": 1e-3,
        "patience": 2,
        "verbose": verbose,
    }

    if params:
        unknown = set(params) - set(INFORMER_PARAM_KEYS) - {
            "seq_len",
            "label_len",
            "pred_len",
            "verbose",
        }
        if unknown:
            raise KeyError(
                f"unknown Informer hyperparameter(s) {sorted(unknown)}; "
                f"accepted: {sorted(set(INFORMER_PARAM_KEYS) | {'seq_len', 'label_len', 'pred_len'})}"
            )
        config.update({key: value for key, value in params.items() if key != "pred_len"})

    config.pop("pred_len", None)  # always the block size used by the protocol

    seq_len = int(min(config.pop("seq_len", spec.seq_len), len(train) - horizon - 1))
    seq_len = max(seq_len, horizon)
    label_len = int(min(config.pop("label_len", spec.label_len), seq_len // 2))
    label_len = max(label_len, 1)

    if verbose:
        print(
            f"    Informer: seq_len={seq_len}, label_len={label_len}, "
            f"pred_len={horizon}, epochs={config['epochs']}, "
            f"d_model={config['d_model']}, n_heads={config['n_heads']}, "
            f"e_layers={config['e_layers']}, d_layers={config['d_layers']}, "
            f"d_ff={config['d_ff']}, dropout={config['dropout']}, "
            f"lr={config['lr']}"
        )

    forecaster = InformerForecaster(
        seq_len=seq_len,
        label_len=label_len,
        pred_len=horizon,
        **config,
    )

    forecaster.fit(train.to_numpy(dtype=float), train.index)

    timestamps = train.index.append(test.index)
    predictions = forecaster.rolling_forecast(
        train.to_numpy(dtype=float),
        test.to_numpy(dtype=float),
        timestamps=timestamps,
    )

    pred = np.asarray(predictions, dtype=float)
    true = test.to_numpy(dtype=float)[: len(pred)]

    return {
        "model": "informer",
        "ts": train.name,
        "test_predict": pred,
        "test_true": true,
        "metrics": compute_metrics(true, pred) if len(pred) else {},
        "time_training": forecaster.time_exec["training"],
        "time_testing": forecaster.time_exec["testing"],
        "history": forecaster.history,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

METRIC_COLUMNS = ["MSE", "RMSE", "MAPE", "SMAPE", "MAE", "theil", "ARV", "IA", "POCID"]


def run_comparison(
    dataset_names: Optional[Iterable[str]] = None,
    test_size: float = 0.2,
    epochs: int = 6,
    quick: bool = False,
    verbose: bool = True,
    informer_params: Optional[Dict[str, Dict[str, object]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """Run ARIMA and Informer on the selected datasets.

    Args:
        dataset_names (Optional[Iterable[str]]): Subset of :data:`DATASETS`;
            defaults to all eight series.
        test_size (float, optional): Fraction of the series used as test.
        epochs (int, optional): Informer epochs. Defaults to 6.
        quick (bool, optional): Fast/small configuration for a smoke run.
        verbose (bool, optional): Print progress.
        informer_params (Optional[Dict[str, Dict[str, object]]], optional):
            Per-dataset overrides forwarded to :func:`run_informer`, e.g.
            ``{"ETTh1.csv": {"d_model": 64, "lr": 5e-4}}``. Datasets without
            an entry keep the default configuration; ``None`` (default)
            reproduces the original demonstration.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
        * metric table with the columns used by ``metrics.open_fold_result``;
        * the raw results per dataset (``{"ts": ..., "arima": ..., "informer": ...}``)
          so the notebook can plot the forecasts.
    """
    names = list(dataset_names) if dataset_names is not None else available_datasets()
    rows: List[Dict[str, object]] = []
    results: Dict[str, Dict[str, object]] = {}

    for name in names:
        spec = DATASETS[name]
        series = load_series(name)
        train, test = split_series(series, test_size)
        horizon = effective_horizon(spec, len(test))

        if verbose:
            print(
                f"{name}: {len(series)} points | train {len(train)} | test {len(test)} "
                f"| horizon {horizon} | seasonality {spec.seasonality}"
            )

        dataset_results: Dict[str, object] = {}

        if verbose:
            print("  fitting ARIMA")
        try:
            arima_result = run_arima(
                train, test, horizon, seasonal_periods=spec.arima_periods, verbose=verbose
            )
        except Exception as error:  # the demo should survive a failing fit
            print(f"  ARIMA failed on {name}: {error}")
            arima_result = None

        if verbose:
            print("  fitting Informer")
        try:
            informer_result = run_informer(
                train,
                test,
                horizon,
                spec,
                epochs=epochs,
                quick=quick,
                verbose=verbose,
                params=(informer_params or {}).get(name),
            )
        except Exception as error:
            print(f"  Informer failed on {name}: {error}")
            informer_result = None

        for result in (arima_result, informer_result):
            if result is None:
                continue
            row = {
                "ts": name,
                "model": result["model"],
                **{key: result["metrics"].get(key, np.nan) for key in METRIC_COLUMNS},
                "time_testing": result["time_testing"],
                "time_training": result["time_training"],
            }
            rows.append(row)
            dataset_results[result["model"]] = result

        dataset_results["spec"] = spec
        dataset_results["horizon"] = horizon
        dataset_results["train"] = train
        dataset_results["test"] = test
        results[name] = dataset_results

    df_metrics = pd.DataFrame(rows)
    if not df_metrics.empty:
        df_metrics = df_metrics.set_index(["ts", "model"])[METRIC_COLUMNS + ["time_training", "time_testing"]]
    return df_metrics, results


def comparison_table(df_metrics: pd.DataFrame, metric: str = "RMSE") -> pd.DataFrame:
    """Pivot the metric table into one row per dataset with both models.

    Args:
        df_metrics (pd.DataFrame): Output of :func:`run_comparison`.
        metric (str, optional): Metric used to decide the best model.

    Returns:
        pd.DataFrame: Columns ``arima``, ``informer``, ``best`` and
        ``informer_gain_%`` (positive means Informer ahead).
    """
    pivot = df_metrics[metric].unstack("model")
    table = pivot.copy()
    table["best"] = pivot.idxmin(axis=1)
    table[f"informer_gain_%"] = (pivot["arima"] - pivot["informer"]) / pivot["arima"] * 100
    return table


# ---------------------------------------------------------------------------
# Plotting helpers (matplotlib imported lazily)
# ---------------------------------------------------------------------------


def plot_series_overview(
    dataset_names: Optional[Iterable[str]] = None,
    max_points: int = 1500,
) -> None:
    """Plot the raw shape of the eight series in a 4x2 grid."""
    import matplotlib.pyplot as plt

    names = list(dataset_names) if dataset_names is not None else available_datasets()
    figure, axes = plt.subplots(4, 2, figsize=(14, 12))
    for axis, name in zip(axes.ravel(), names):
        series = load_series(name)
        shown = series.iloc[-max_points:]
        axis.plot(shown.index, shown.to_numpy(), linewidth=0.8)
        axis.set_title(f"{name} ({len(series)} points)")
        axis.tick_params(axis="x", labelrotation=20, labelsize=7)
    figure.suptitle("Series available in data/raw")
    figure.tight_layout()
    plt.show()


def plot_forecasts(
    results: Dict[str, Dict[str, object]],
    tail: int = 200,
    max_panels: int = 8,
) -> None:
    """Actual vs. ARIMA vs. Informer on the test window of each dataset."""
    import matplotlib.pyplot as plt

    names = [name for name in results if "test" in results[name]][:max_panels]
    if not names:
        print("no results to plot")
        return

    n_cols = 2
    n_rows = int(np.ceil(len(names) / n_cols))
    figure, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.2 * n_rows), squeeze=False)

    for axis, name in zip(axes.ravel(), names):
        data = results[name]
        test: pd.Series = data["test"]  # type: ignore[assignment]
        actual = test.to_numpy(dtype=float)[-tail:]
        axis.plot(test.index[-tail:], actual, label="actual", color="black", linewidth=1.2)

        for model_name, color in (("arima", "tab:orange"), ("informer", "tab:blue")):
            result = data.get(model_name)
            if not result:
                continue
            pred = np.asarray(result["test_predict"], dtype=float)
            # O ultimo bloco pode ser parcial: alinhe a previsao ao fim do teste.
            pred_index = test.index[len(test) - len(pred) :]
            axis.plot(pred_index[-tail:], pred[-tail:], label=model_name.upper(), color=color, linewidth=1.0)

        axis.set_title(f"{name} - test window (last {len(actual)} points)")
        axis.legend(fontsize=7)
        axis.tick_params(axis="x", labelrotation=20, labelsize=7)

    for axis in axes.ravel()[len(names) :]:
        axis.axis("off")

    figure.tight_layout()
    plt.show()


def plot_metric_comparison(df_metrics: pd.DataFrame, metric: str = "RMSE") -> None:
    """Grouped bar chart of one metric, with every dataset normalized by ARIMA."""
    import matplotlib.pyplot as plt

    pivot = df_metrics[metric].unstack("model")
    relative = pivot.div(pivot["arima"], axis=0)

    axis = relative.plot(kind="bar", figsize=(12, 4.5))
    axis.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_ylabel(f"{metric} relative to ARIMA (1.0 = tie)")
    axis.set_title(f"{metric} per dataset, normalized by the ARIMA value")
    axis.tick_params(axis="x", labelrotation=30, labelsize=8)
    axis.legend(title="model")
    plt.tight_layout()
    plt.show()
