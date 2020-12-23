import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numereval.scores import *
from numereval.neutralize import *


def calculate_val_corrs(validation_data):

    validation_correlations = validation_data.groupby("era").apply(score)
    return validation_correlations


def calculate_sharpe(validation):

    validation_sharpe = validation_correlations.mean() / validation_correlations.std(
        ddof=0
    )
    return validation_sharpe


def plot_correlation(validation_correlations: pd.Series):
    colors = (
        ["r" for _ in range(121, 132 + 1)]
        + ["g" for _ in range(197, 206 + 1)]
        + ["b" for _ in range(207, 212 + 1)]
    )

    r_patch = mpatches.Patch(color="red", label="Val1")
    g_patch = mpatches.Patch(color="green", label="Val2")
    b_patch = mpatches.Patch(color="blue", label="Val3")

    fig, ax = plt.subplots()
    validation_correlations.plot(
        kind="bar", color=colors, legend=True, title="Validation correlations"
    )
    ax.legend(handles=[r_patch, g_patch, b_patch])
    plt.show()


def calculate_max_drawdown(validation_correlations: pd.Series):

    rolling_max = (
        (validation_correlations + 1).cumprod().rolling(window=100, min_periods=1).max()
    )
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -(rolling_max - daily_value).max()

    return max_drawdown


def calculate_feature_exposure(validation_data):

    feature_names = [f for f in validation_data.columns if f.startswith("feature")]
    feature_exposures = validation_data[feature_names].apply(
        lambda d: correlation(validation_data[PREDICTION_NAME], d), axis=0
    )

    feature_exposure = np.sqrt(np.mean(np.square(feature_exposures)))

    max_per_era = validation_data.groupby("era").apply(
        lambda d: d[feature_names].corrwith(d[PREDICTION_NAME]).abs().max()
    )
    max_feature_exposure = max_per_era.mean()

    return max_feature_exposure, feature_exposure


def evaluate(validation_data: pd.DataFrame, example_preds_loc: str = None):

    if example_preds_loc is None:
        example_preds_loc = "example_predictions.csv"

    feature_names = [f for f in validation_data.columns if f.startswith("feature")]

    metrics = {}

    validation_correlations = calculate_val_corrs(validation_data)
    plot_correlation(validation_correlations)

    metrics["validation mean"] = validation_correlations.mean()
    metrics["validation std"] = validation_correlations.std(ddof=0)
    validation_sharpe = metrics["validation mean"] / metrics["validation std"]
    metrics["validation Sharpe"] = validation_sharpe
    metrics["max drawdown"] = calculate_max_drawdown(validation_correlations)

    (
        metrics["max_feature_exp"],
        metrics["feature exposure"],
    ) = calculate_feature_exposure(validation_data)

    example_preds = pd.read_csv(example_preds_loc).set_index("id")["prediction"]
    validation_example_preds = example_preds.loc[validation_data.index]
    validation_data["ExamplePreds"] = validation_example_preds

    mmc_scores = []
    corr_scores = []
    for _, x in validation_data.groupby("era"):
        series = neutralize_series(
            pd.Series(unif(x[PREDICTION_NAME])), pd.Series(unif(x["ExamplePreds"]))
        )
        mmc_scores.append(np.cov(series, x[TARGET_NAME])[0, 1] / (0.29 ** 2))
        corr_scores.append(correlation(unif(x[PREDICTION_NAME]), x[TARGET_NAME]))

    val_mmc_mean = np.mean(mmc_scores)
    val_mmc_std = np.std(mmc_scores)
    val_mmc_sharpe = val_mmc_mean / val_mmc_std
    corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
    corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)
    corr_plus_mmc_mean = np.mean(corr_plus_mmcs)
    corr_plus_mmc_sharpe_diff = corr_plus_mmc_sharpe - validation_sharpe

    metrics["MMC Mean"] = val_mmc_mean
    metrics["Corr Plus MMC Sharpe"] = corr_plus_mmc_sharpe
    metrics["Corr Plus MMC Diff"] = corr_plus_mmc_sharpe_diff

    return pd.DataFrame.from_dict(metrics, orient="index")
