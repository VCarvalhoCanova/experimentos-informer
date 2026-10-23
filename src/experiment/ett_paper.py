"""Paper-protocol ETT experiment (Informer, Table 2 of Zhou et al., 2021).

The helpers here implement the multivariate ETT setting of the article:

* all seven variables (six power-load features + ``OT``) are input **and**
  output (``c_in = c_out = 7``), and MSE/MAE are averaged over the variables;
* chronological split of **12 / 4 / 4 months** (Section 4, "Datasets");
* prediction windows of ``{24, 48, 168, 336, 720}`` for ETTh and
  ``{24, 48, 96, 288, 672}`` for ETTm (Table 1/2 and Section 4);
* rolling evaluation with **stride 1**: every test origin produces one
  ``L_y``-step window and the metrics are averaged over the windows
  ("Metrics" paragraph of Section 4);
* network/training setup from Section 4.2 and Appendix E/Table 7:
  ``d_model=512``, ``d_ff=2048``, 3-layer encoder stack + 1-layer replica with
  1/4 of the input, 2-layer decoder, dropout 0.1, Adam at ``1e-4`` halved every
  epoch, 8 epochs, batch size 32, early stopping on the validation months.

The module also fits the project's ARIMA wrapper per variable as an extra
baseline (the paper's multivariate table does not report ARIMA).

Everything is resumable: trained Informer weights are cached per
``(dataset, horizon)``, the fitted ARIMA models per dataset, and the metric rows
are appended to ``data/result/paper_ett/metrics.csv``.
"""

from __future__ import annotations

import copy
import io
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Project root resolved from this file (``src/experiment/ett_paper.py``).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
RESULT_PATH = PROJECT_ROOT / "data" / "result" / "paper_ett"

#: Feature order of the ETT csv files (paper's 6 load variables + target).
FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET = "OT"


@dataclass(frozen=True)
class ETTSpec:
    """Description of one ETT dataset and of the paper's protocol for it."""

    name: str
    freq: str
    horizons: Tuple[int, ...]
    arima_seasonal_period: int
    """Seasonal period used by the ARIMA baseline (1 disables seasonality)."""

    @property
    def points_per_month(self) -> int:
        """Number of observations in one 30-day month."""
        return 30 * 24 if self.freq == "h" else 30 * 24 * 4


DATASETS: Dict[str, ETTSpec] = {
    spec.name: spec
    for spec in [
        # Table 1/2 horizons for ETT hourly. The ARIMA baseline is non-seasonal
        # by default: a seasonal search with m=24 is the slowest part of the
        # pipeline and the paper's Table 2 does not report ARIMA. Pass
        # ``seasonal_period=24`` to run_arima/run_experiment to enable it.
        ETTSpec("ETTh1.csv", "h", (24, 48, 168, 336, 720), 1),
        ETTSpec("ETTh2.csv", "h", (24, 48, 168, 336, 720), 1),
        # Table 1/2 horizons for ETT 15-minute. ETTm2 is not in the paper tables
        # but is shipped with the dataset, so it uses the same protocol.
        ETTSpec("ETTm1.csv", "15min", (24, 48, 96, 288, 672), 1),
        ETTSpec("ETTm2.csv", "15min", (24, 48, 96, 288, 672), 1),
    ]
}

#: Network and optimisation setup of the paper (Section 4.2, Appendix E, Table 7).
PAPER_CONFIG: Dict[str, object] = {
    "seq_len": 96,
    "label_len": 48,
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 3,
    "d_layers": 2,
    "d_ff": 2048,
    "factor": 5,
    "dropout": 0.1,
    "distil": True,
    "n_encoder_stacks": 2,
    "replica_layers": 1,
    "replica_input_fraction": 0.25,
    "epochs": 8,
    "batch_size": 32,
    "lr": 1e-4,
    "patience": 3,
}

#: Informer MSE/MAE reported in Table 2 (multivariate), used as reference only.
PAPER_REFERENCE: Dict[str, Dict[int, Tuple[float, float]]] = {
    "ETTh1.csv": {
        24: (0.577, 0.549),
        48: (0.685, 0.625),
        168: (0.931, 0.752),
        336: (1.128, 0.873),
        720: (1.215, 0.896),
    },
    "ETTh2.csv": {
        24: (0.720, 0.665),
        48: (1.457, 1.001),
        168: (3.489, 1.515),
        336: (2.723, 1.340),
        720: (3.467, 1.473),
    },
    "ETTm1.csv": {
        24: (0.323, 0.369),
        48: (0.494, 0.503),
        96: (0.678, 0.614),
        288: (1.056, 0.786),
        672: (1.192, 0.926),
    },
    # ETTm2 was added to the dataset after the paper; no reference numbers.
}

METRIC_COLUMNS = [
    "dataset",
    "horizon",
    "model",
    "MSE",
    "MAE",
    "MSE_OT",
    "MAE_OT",
    "n_windows",
    "n_params",
    "time_training",
    "time_testing",
]


def available_datasets() -> List[str]:
    """Return the four ETT datasets covered by this protocol."""
    return list(DATASETS)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_frame(name: str) -> pd.DataFrame:
    """Load one ETT csv with a ``DatetimeIndex`` and the seven paper columns.

    Args:
        name (str): File name, e.g. ``"ETTh1.csv"``.

    Returns:
        pd.DataFrame: Values of :data:`FEATURES`, float, indexed by ``date``.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; options: {available_datasets()}")
    frame = pd.read_csv(RAW_DATA_PATH / name, parse_dates=["date"]).set_index("date")
    missing = [column for column in FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing the columns {missing}")
    return frame[FEATURES].astype(float)


def paper_split(frame: pd.DataFrame, freq: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the series into the paper's 12 / 4 / 4 months.

    Args:
        frame (pd.DataFrame): Output of :func:`load_frame`.
        freq (str): ``"h"`` or ``"15min"``.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation and
        test frames, in chronological order.
    """
    points = 30 * 24 if freq == "h" else 30 * 24 * 4
    train_end = 12 * points
    val_end = train_end + 4 * points
    test_end = val_end + 4 * points
    if test_end > len(frame):
        raise ValueError(
            f"series with {len(frame)} points is too short for the 12/4/4-month split "
            f"({test_end} points at freq={freq!r})"
        )
    return frame.iloc[:train_end], frame.iloc[train_end:val_end], frame.iloc[val_end:test_end]


def split_summary() -> pd.DataFrame:
    """Return one row per dataset with the split sizes and date ranges."""
    rows = []
    for name, spec in DATASETS.items():
        frame = load_frame(name)
        train, val, test = paper_split(frame, spec.freq)
        rows.append(
            {
                "dataset": name,
                "freq": spec.freq,
                "pontos_arquivo": len(frame),
                "treino": len(train),
                "val": len(val),
                "teste": len(test),
                "treino_inicio": train.index[0],
                "treino_fim": train.index[-1],
                "teste_inicio": test.index[0],
                "teste_fim": test.index[-1],
                "horizontes": spec.horizons,
                "pontos_nao_usados": len(frame) - len(train) - len(val) - len(test),
            }
        )
    return pd.DataFrame(rows)


def reference_table() -> pd.DataFrame:
    """Long table with the paper's Table 2 Informer numbers (where available)."""
    rows = []
    for name, horizons in PAPER_REFERENCE.items():
        for horizon, (mse, mae) in horizons.items():
            rows.append({"dataset": name, "horizon": horizon, "MSE_paper": mse, "MAE_paper": mae})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def window_scores(prediction: np.ndarray, truth: np.ndarray) -> Dict[str, object]:
    """Score rolling windows exactly as the paper does.

    The paper computes MSE/MAE inside every prediction window and averages both
    over the windows and (for the multivariate setting) over the variables.
    Because every window has the same size this is the mean over all elements of
    ``prediction - truth``; the per-variable values are kept for inspection.

    Args:
        prediction (np.ndarray): Forecasts ``(n_windows, horizon, n_features)``.
        truth (np.ndarray): Observed windows with the same shape.

    Returns:
        Dict[str, object]: ``MSE``, ``MAE`` (averaged), ``mse_by_var`` and
        ``mae_by_var``.
    """
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    squared = difference**2
    absolute = np.abs(difference)
    return {
        "MSE": float(squared.mean()),
        "MAE": float(absolute.mean()),
        "mse_by_var": squared.mean(axis=(0, 1)),
        "mae_by_var": absolute.mean(axis=(0, 1)),
    }


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _informer_checkpoint(name: str, horizon: int) -> Path:
    return RESULT_PATH / "checkpoints" / f"informer_{Path(name).stem}_{horizon}.pt"


def _arima_models_path(name: str, seasonal_period: int) -> Path:
    return RESULT_PATH / "arima" / f"{Path(name).stem}_m{seasonal_period}.joblib"


def metrics_path() -> Path:
    """Path of the resumable metric table."""
    return RESULT_PATH / "metrics.csv"


def load_metrics() -> pd.DataFrame:
    """Load the partial metric table (empty frame when nothing was run yet)."""
    path = metrics_path()
    if not path.exists():
        return pd.DataFrame(columns=METRIC_COLUMNS)
    return pd.read_csv(path)


def _cached_row(name: str, horizon: int, model: str) -> Optional[Dict[str, object]]:
    frame = load_metrics()
    if frame.empty:
        return None
    selected = frame[
        (frame["dataset"] == name) & (frame["horizon"] == horizon) & (frame["model"] == model)
    ]
    return selected.iloc[-1].to_dict() if len(selected) else None


def _append_row(row: Dict[str, object]) -> None:
    path = metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    scalar = {key: row.get(key) for key in METRIC_COLUMNS}
    frame = pd.DataFrame([scalar])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _save_samples(name: str, horizon: int, model: str, samples: Dict[str, np.ndarray]) -> None:
    directory = RESULT_PATH / "samples"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{Path(name).stem}_{horizon}_{model}.npz", **samples)


def load_samples(name: str, horizon: int, model: str) -> Optional[Dict[str, np.ndarray]]:
    """Load the sample windows saved for one (dataset, horizon, model)."""
    path = RESULT_PATH / "samples" / f"{Path(name).stem}_{horizon}_{model}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# Informer
# ---------------------------------------------------------------------------


def make_forecaster(config: Dict[str, object], horizon: int, device: str, verbose: bool):
    from model.informer import InformerMultivariateForecaster

    parameters = dict(PAPER_CONFIG)
    parameters.update(config or {})
    parameters.pop("pred_len", None)
    return InformerMultivariateForecaster(
        seq_len=int(parameters["seq_len"]),
        label_len=int(parameters["label_len"]),
        pred_len=int(horizon),
        d_model=int(parameters["d_model"]),
        n_heads=int(parameters["n_heads"]),
        e_layers=int(parameters["e_layers"]),
        d_layers=int(parameters["d_layers"]),
        d_ff=int(parameters["d_ff"]),
        factor=int(parameters["factor"]),
        dropout=float(parameters["dropout"]),
        distil=bool(parameters["distil"]),
        n_encoder_stacks=int(parameters["n_encoder_stacks"]),
        replica_layers=parameters["replica_layers"],
        replica_input_fraction=float(parameters["replica_input_fraction"]),
        epochs=int(parameters["epochs"]),
        batch_size=int(parameters["batch_size"]),
        lr=float(parameters["lr"]),
        patience=int(parameters["patience"]),
        seed=int(parameters.get("seed", 42)),
        device=device,
        verbose=verbose,
    )


def describe_model(
    horizon: int = 24, config: Optional[Dict[str, object]] = None
) -> Dict[str, object]:
    """Instantiate the paper's network without training and describe its shape.

    Args:
        horizon (int, optional): Forecast horizon used to build the decoder.
        config (Optional[Dict[str, object]]): Overrides for :data:`PAPER_CONFIG`.

    Returns:
        Dict[str, object]: Number of parameters, encoder/decoder layers and
        number of features, ready to display in the notebook.
    """
    forecaster = make_forecaster(config or {}, horizon, device="cpu", verbose=False)
    model = forecaster.build_model(len(FEATURES))
    return {
        "n_features": len(FEATURES),
        "n_params": int(sum(parameter.numel() for parameter in model.parameters())),
        "encoder_stacks": [len(encoder.attn_layers) for encoder in model.encoders],
        "encoder_layers_total": int(sum(len(encoder.attn_layers) for encoder in model.encoders)),
        "decoder_layers": len(model.decoder.layers),
        "encoder_input_fractions": [round(value, 3) for value in model.encoder_input_fractions],
    }


def run_informer(
    name: str,
    horizon: int,
    config: Optional[Dict[str, object]] = None,
    device: str = "cpu",
    force: bool = False,
    save_checkpoint: bool = True,
    batch_size: int = 64,
    max_origins: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Train (or reload) and evaluate the Informer on one ETT dataset/horizon.

    Args:
        name (str): Dataset file name.
        horizon (int): Forecast horizon ``L_y``.
        config (Optional[Dict[str, object]]): Overrides for :data:`PAPER_CONFIG`.
        device (str, optional): Torch device. Defaults to ``"cpu"``.
        force (bool, optional): Retrain even if a checkpoint exists.
        save_checkpoint (bool, optional): Persist the trained weights.
        batch_size (int, optional): Origins per forward pass during evaluation.
        max_origins (Optional[int], optional): Cap the number of test origins
            (only for smoke tests; ``None`` runs the full stride-1 evaluation).
        verbose (bool, optional): Print the training/evaluation progress.

    Returns:
        Dict[str, object]: Metric row plus ``samples`` and ``forecaster``.
    """
    spec = DATASETS[name]
    frame = load_frame(name)
    train, val, test = paper_split(frame, spec.freq)
    full = pd.concat([train, val, test])

    checkpoint = _informer_checkpoint(name, horizon)
    desired = make_forecaster(config or {}, horizon, device=device, verbose=False)
    forecaster = None
    if checkpoint.exists() and not force:
        from model.informer import InformerMultivariateForecaster

        cached = InformerMultivariateForecaster.load_checkpoint(checkpoint, device=device)
        if cached._constructor_kwargs() == desired._constructor_kwargs():
            forecaster = cached
            if verbose:
                print(f"    checkpoint carregado: {checkpoint.name}")
        elif verbose:
            print("    configuração mudou; retreinando o Informer")
    if forecaster is None:
        forecaster = make_forecaster(config or {}, horizon, device=device, verbose=verbose)
        if verbose:
            print(
                f"    treinando Informer: seq_len={forecaster.seq_len}, "
                f"label_len={forecaster.label_len}, pred_len={horizon}, "
                f"d_model={forecaster.d_model}, e_layers={forecaster.e_layers}"
                f"+{forecaster.replica_layers}, d_layers={forecaster.d_layers}, "
                f"epochs={forecaster.epochs}"
            )
        forecaster.fit(
            train.to_numpy(dtype=float),
            train.index,
            val.to_numpy(dtype=float),
            val.index,
        )
        if save_checkpoint:
            forecaster.save_checkpoint(checkpoint)

    values = full.to_numpy(dtype=float)
    test_start = len(train) + len(val)
    n_test = len(test)
    n_windows = n_test - horizon + 1
    if n_windows <= 0:
        raise ValueError(f"test window with {n_test} points is too short for horizon={horizon}")
    origins = np.arange(n_windows, dtype=np.int64) + test_start
    if max_origins is not None:
        origins = origins[: int(max_origins)]

    sample_positions = sorted({0, len(origins) // 2, len(origins) - 1})
    samples: Dict[str, np.ndarray] = {}
    total_squared = 0.0
    total_absolute = 0.0
    squared_by_var = np.zeros(len(FEATURES), dtype=np.float64)
    absolute_by_var = np.zeros(len(FEATURES), dtype=np.float64)
    n_elements = 0

    start = time.time()
    for begin in range(0, len(origins), batch_size):
        batch_origins = origins[begin : begin + batch_size]
        predictions = forecaster.predict_windows(
            values, full.index, batch_origins, batch_size=len(batch_origins)
        )
        targets = np.stack(
            [values[origin + forecaster.seq_len : origin + forecaster.seq_len + horizon] for origin in batch_origins]
        )
        difference = predictions.astype(np.float64) - targets.astype(np.float64)
        squared = difference**2
        absolute = np.abs(difference)
        total_squared += float(squared.sum())
        total_absolute += float(absolute.sum())
        squared_by_var += squared.sum(axis=(0, 1))
        absolute_by_var += absolute.sum(axis=(0, 1))
        n_elements += difference.size

        for position in sample_positions:
            if begin <= position < begin + len(batch_origins):
                index = position - begin
                samples.setdefault("origin", []).append(int(batch_origins[index]))
                samples.setdefault("pred", []).append(predictions[index].astype(np.float32))
                samples.setdefault("true", []).append(targets[index].astype(np.float32))

    testing_time = time.time() - start
    samples = {key: np.asarray(value) for key, value in samples.items()}
    _save_samples(name, horizon, "informer", samples)

    mse = total_squared / n_elements
    mae = total_absolute / n_elements
    target_index = FEATURES.index(TARGET)
    row: Dict[str, object] = {
        "dataset": name,
        "horizon": horizon,
        "model": "informer",
        "MSE": mse,
        "MAE": mae,
        "MSE_OT": float(squared_by_var[target_index] / (len(origins) * horizon)),
        "MAE_OT": float(absolute_by_var[target_index] / (len(origins) * horizon)),
        "n_windows": int(len(origins)),
        "n_params": int(sum(parameter.numel() for parameter in forecaster.model.parameters())),
        "time_training": forecaster.time_exec["training"],
        "time_testing": testing_time,
        "mse_by_var": dict(zip(FEATURES, squared_by_var / (len(origins) * horizon))),
        "mae_by_var": dict(zip(FEATURES, absolute_by_var / (len(origins) * horizon))),
        "history": forecaster.history,
        "samples": samples,
        "forecaster": forecaster,
    }
    _append_row(row)

    if verbose:
        print(
            f"    Informer avaliado: {len(origins)} janelas | step 1 | "
            f"MSE {mse:.4f} | MAE {mae:.4f} | teste {testing_time:.1f}s"
        )
    return row


# ---------------------------------------------------------------------------
# ARIMA (extra baseline, one univariate model per variable)
# ---------------------------------------------------------------------------


def fit_arima_models(
    name: str,
    seasonal_period: Optional[int] = None,
    force: bool = False,
    verbose: bool = True,
) -> List[object]:
    """Fit one project ARIMA per ETT variable and cache them with joblib.

    Args:
        name (str): Dataset file name.
        seasonal_period (Optional[int]): Seasonal period; defaults to the
            dataset registry value.
        force (bool, optional): Refit even if a cached file exists.
        verbose (bool, optional): Show the ``auto_arima`` search.

    Returns:
        List[object]: Seven fitted ``pmdarima`` models, in :data:`FEATURES` order.
    """
    import joblib

    from model.arima import Arima

    spec = DATASETS[name]
    period = spec.arima_seasonal_period if seasonal_period is None else int(seasonal_period)
    path = _arima_models_path(name, period)
    if path.exists() and not force:
        if verbose:
            print(f"    ARIMA carregado: {path.name}")
        return joblib.load(path)

    train, _, _ = paper_split(load_frame(name), spec.freq)
    models = []
    for column in FEATURES:
        model = Arima(seazonal_lag=period, horizon=1)
        # ``auto_arima`` is configured with trace=True in the project wrapper;
        # capture it so the notebook progress stays readable.
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            model.fit(train[column].to_numpy(dtype=float))
        models.append(model.forecaster)
        if verbose:
            fitted = model.forecaster
            print(f"    ARIMA({column}) pronto: {fitted.order} x {fitted.seasonal_order}")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path)
    return models


def run_arima(
    name: str,
    horizon: int,
    device: str = "cpu",
    force: bool = False,
    seasonal_period: Optional[int] = None,
    batch_size: int = 64,
    max_origins: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Evaluate the project's ARIMA on the paper's rolling protocol.

    One ARIMA is fitted per variable on the training months and updated with
    every observed test value, producing an ``horizon``-step forecast at every
    test origin (stride 1).

    Args:
        name (str): Dataset file name.
        horizon (int): Forecast horizon.
        device (str, optional): Kept for symmetry with :func:`run_informer`.
        force (bool, optional): Refit the ARIMA models.
        seasonal_period (Optional[int]): Override the seasonal period.
        batch_size (int, optional): Unused; kept for API symmetry.
        max_origins (Optional[int], optional): Cap the number of origins (smoke
            tests only; ``None`` runs the whole test window).
        verbose (bool, optional): Print progress.

    Returns:
        Dict[str, object]: Metric row plus ``samples``.
    """
    del device, batch_size  # the ARIMA baseline is always CPU/sequential

    spec = DATASETS[name]
    frame = load_frame(name)
    train, val, test = paper_split(frame, spec.freq)
    models = copy.deepcopy(fit_arima_models(name, seasonal_period, force=force, verbose=verbose))

    test_values = test.to_numpy(dtype=float)
    n_test = len(test_values)
    n_windows = n_test - horizon + 1
    if n_windows <= 0:
        raise ValueError(f"test window with {n_test} points is too short for horizon={horizon}")
    if max_origins is not None and int(max_origins) < 1:
        raise ValueError("max_origins must be positive when provided")
    wanted = np.zeros(n_windows, dtype=bool)
    wanted[: n_windows if max_origins is None else min(n_windows, int(max_origins))] = True
    selected = np.flatnonzero(wanted)
    sample_positions = set(int(position) for position in [selected[0], selected[len(selected) // 2], selected[-1]])

    samples: Dict[str, List[np.ndarray]] = {}
    total_squared = 0.0
    total_absolute = 0.0
    squared_by_var = np.zeros(len(FEATURES), dtype=np.float64)
    absolute_by_var = np.zeros(len(FEATURES), dtype=np.float64)
    n_elements = 0
    count = 0

    start = time.time()
    for position in range(n_test):
        if position < n_windows and wanted[position]:
            prediction = np.column_stack([model.predict(horizon) for model in models])
            truth = test_values[position : position + horizon]
            difference = prediction.astype(np.float64) - truth.astype(np.float64)
            squared = difference**2
            absolute = np.abs(difference)
            total_squared += float(squared.sum())
            total_absolute += float(absolute.sum())
            squared_by_var += squared.sum(axis=0)
            absolute_by_var += absolute.sum(axis=0)
            n_elements += difference.size
            count += 1
            if position in sample_positions:
                samples.setdefault("origin", []).append(int(position))
                samples.setdefault("pred", []).append(prediction.astype(np.float32))
                samples.setdefault("true", []).append(truth.astype(np.float32))
        for column, model in enumerate(models):
            model.update(test_values[position, column])
        if verbose and count and count % 500 == 0:
            print(f"      {count}/{int(wanted.sum())} janelas ARIMA")

    testing_time = time.time() - start
    samples_array = {key: np.asarray(value) for key, value in samples.items()}
    _save_samples(name, horizon, "arima", samples_array)

    mse = total_squared / n_elements
    mae = total_absolute / n_elements
    target_index = FEATURES.index(TARGET)
    row: Dict[str, object] = {
        "dataset": name,
        "horizon": horizon,
        "model": "arima",
        "MSE": mse,
        "MAE": mae,
        "MSE_OT": float(squared_by_var[target_index] / (count * horizon)),
        "MAE_OT": float(absolute_by_var[target_index] / (count * horizon)),
        "n_windows": int(count),
        "n_params": np.nan,
        "time_training": np.nan,
        "time_testing": testing_time,
        "mse_by_var": dict(zip(FEATURES, squared_by_var / (count * horizon))),
        "mae_by_var": dict(zip(FEATURES, absolute_by_var / (count * horizon))),
        "samples": samples_array,
    }
    _append_row(row)
    if verbose:
        print(
            f"    ARIMA avaliado: {count} janelas | step 1 | "
            f"MSE {mse:.4f} | MAE {mae:.4f} | teste {testing_time:.1f}s"
        )
    return row


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_experiment(
    name: str,
    horizon: int,
    models: Sequence[str] = ("informer", "arima"),
    config: Optional[Dict[str, object]] = None,
    device: str = "cpu",
    force: bool = False,
    max_origins: Optional[int] = None,
    arima_seasonal_period: Optional[int] = None,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """Run the selected models on one (dataset, horizon), reusing cached rows.

    Args:
        name (str): Dataset file name.
        horizon (int): Forecast horizon.
        models (Sequence[str], optional): ``"informer"`` and/or ``"arima"``.
        config (Optional[Dict[str, object]]): Informer overrides.
        device (str, optional): Torch device for the Informer.
        force (bool, optional): Ignore cached checkpoints and metric rows.
        max_origins (Optional[int], optional): Cap the test origins (smoke only).
        arima_seasonal_period (Optional[int]): Seasonal period of the ARIMA
            baseline; ``None`` uses the registry (non-seasonal by default).
        verbose (bool, optional): Print progress.

    Returns:
        List[Dict[str, object]]: One metric row per model.
    """
    rows: List[Dict[str, object]] = []
    for model in models:
        if model not in {"informer", "arima"}:
            raise ValueError(f"unknown model {model!r}")
        # A custom Informer config invalidates the metric cache as well (the
        # cache key is dataset/horizon/model, not the hyperparameters).
        custom_informer = model == "informer" and config is not None
        cached = (
            None
            if force or max_origins is not None or custom_informer
            else _cached_row(name, horizon, model)
        )
        if cached is not None:
            if verbose:
                print(f"    {model}: resultado em cache (MSE {cached['MSE']:.4f})")
            rows.append(cached)
            continue
        if verbose:
            print(f"  {name} | h={horizon} | {model}")
        if model == "informer":
            rows.append(
                run_informer(
                    name,
                    horizon,
                    config=config,
                    device=device,
                    force=force,
                    max_origins=max_origins,
                    verbose=verbose,
                )
            )
        else:
            rows.append(
                run_arima(
                    name,
                    horizon,
                    force=force,
                    seasonal_period=arima_seasonal_period,
                    max_origins=max_origins,
                    verbose=verbose,
                )
            )
    return rows


def comparison_table(metrics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Pivot the metric rows into one line per (dataset, horizon, model)."""
    frame = load_metrics() if metrics is None else metrics
    if frame.empty:
        return frame
    table = frame.pivot_table(
        index=["dataset", "horizon"], columns="model", values=["MSE", "MAE"], aggfunc="last"
    )
    table.columns = [f"{metric}_{model}" for metric, model in table.columns]
    return table.reset_index()


def clear_cache(name: Optional[str] = None, horizon: Optional[int] = None) -> None:
    """Delete cached checkpoints/metric rows (optionally for one dataset/horizon)."""
    frame = load_metrics()
    if not frame.empty:
        keep = pd.Series(True, index=frame.index)
        if name is not None:
            keep &= frame["dataset"] != name
        if horizon is not None:
            keep &= frame["horizon"] != horizon
        frame[keep].to_csv(metrics_path(), index=False)
    if name is not None:
        for path in (RESULT_PATH / "checkpoints").glob(f"informer_{Path(name).stem}_*.pt"):
            if horizon is None or path.stem.endswith(f"_{horizon}"):
                path.unlink()
        for path in (RESULT_PATH / "samples").glob(f"{Path(name).stem}_*.npz"):
            if horizon is None or path.stem.split("_")[-2] == str(horizon):
                path.unlink()
        if horizon is None:
            for path in (RESULT_PATH / "arima").glob(f"{Path(name).stem}_m*.joblib"):
                path.unlink()
