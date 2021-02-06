import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numereval.scores import *
from numereval.neutralize import *
from typing import Union


def calculate_val_corrs(validation_data):

    validation_correlations = validation_data.groupby("era").apply(score)
    return validation_correlations


def calculate_sharpe(validation):

    validation_sharpe = validation_correlations.mean() / validation_correlations.std(
        ddof=0
    )
    return validation_sharpe


def plot_correlation(validation_correlations: pd.Series, diagnostics=False):

    if diagnostics:

        colors = []
        era_num = validation_correlations.index.str.slice(start=3).astype(int) 
        for e in era_num:
            if e in range(121, 132+1):
                colors.append("#007691")
            elif e in range(197, 206+1):
                colors.append("#50bf84")
            elif e in range(207, 212+1):
                colors.append("#3ac4e1")

        r_patch = mpatches.Patch(color="#007691", label="Val1")
        g_patch = mpatches.Patch(color="#50bf84", label="Val2")
        b_patch = mpatches.Patch(color="#3ac4e1", label="Val3")

        fig, ax = plt.subplots()
        validation_correlations.plot(
            kind="bar", color=colors, legend=True, title="Validation correlations"
        )
        ax.legend(handles=[r_patch, g_patch, b_patch])
        plt.axhline(y=validation_correlations.mean(), 
                linewidth=1,
                color='b', linestyle='-')
        plt.show()
    else:
        fig, ax = plt.subplots()
        validation_correlations.plot(kind="bar", title="Correlations")
        plt.axhline(y=validation_correlations.mean(), 
                linewidth=1,
                color='r', linestyle='-')
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


def evaluate(validation_data: pd.DataFrame, plot=False, feature_exposure=False,
    eras= None
):

    feature_names = [f for f in validation_data.columns if f.startswith("feature")]

    if eras is not None:
        validation_data = validation_data[validation_data.era.isin(eras)]

    metrics = {}

    validation_correlations = calculate_val_corrs(validation_data)
    if plot:
        plot_correlation(validation_correlations)

    metrics["mean"] = validation_correlations.mean()
    metrics["std"] = validation_correlations.std(ddof=0)
    validation_sharpe = metrics["mean"] / metrics["std"]
    metrics["sharpe"] = validation_sharpe
    metrics["max_drawdown"] = calculate_max_drawdown(validation_correlations)

    if feature_exposure:
        (
            metrics["max_feature_exp"],
            metrics["feature_exposure"],
        ) = calculate_feature_exposure(validation_data)

    return pd.DataFrame.from_dict(metrics, orient="index",  columns=["metrics"]).round(4)


def diagnostics(
    validation_data: pd.DataFrame, plot=False, example_preds_loc: str = None,
    eras= None
):

    if example_preds_loc is None:
        example_preds_loc = "example_predictions.csv"

    feature_names = [f for f in validation_data.columns if f.startswith("feature")]

    if eras is not None:
        validation_data = validation_data[validation_data.era.isin(eras)]

    metrics = {}

    validation_correlations = calculate_val_corrs(validation_data)
    if plot:
        plot_correlation(validation_correlations, diagnostics=True)

    metrics["mean"] = validation_correlations.mean()
    metrics["std_dev"] = validation_correlations.std(ddof=0)
    validation_sharpe = metrics["mean"] / metrics["std_dev"]
    metrics["sharpe"] = validation_sharpe

    metrics["feat_neutral_mean"] = get_feature_neutral_mean(validation_data)

    metrics["max_drawdown"] = calculate_max_drawdown(validation_correlations)

    (
        metrics["max_feature_exp"],
        metrics["feature_exposure_mean"],
    ) = calculate_feature_exposure(validation_data)

    example_preds = pd.read_csv(example_preds_loc).set_index("id")["prediction"]
    validation_example_preds = example_preds.loc[validation_data.index]
    validation_data["ExamplePreds"] = validation_example_preds

    scores = validation_data.groupby("era").apply(
        lambda x: correlation(unif(x[PREDICTION_NAME]), 
                            x["ExamplePreds"])
        )

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

    metrics["mmc_mean"] = val_mmc_mean
    metrics["corr_plus_mmc_sharpe"] = corr_plus_mmc_sharpe
    #metrics["corr_plus_mmc_diff"] = corr_plus_mmc_sharpe_diff
    metrics["corr_example_preds"] = scores.mean()
    return pd.DataFrame.from_dict(metrics, orient="index", columns=["metrics"]).round(4)
