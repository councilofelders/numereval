
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numereval.scores import score_signals


def calculate_max_drawdown(validation_correlations: pd.Series):

    rolling_max = (
        (validation_correlations + 1).cumprod().rolling(window=100, min_periods=1).max()
    )
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value)).max()

    #TODO: Add Max drawdown weeks
    return max_drawdown

def run_analytics(era_scores, roll_mean=20, plot=False):

    '''Calculates some stats and plot cumulative scores.
        Taken from Jason Rosenfeld's notebook.
    '''

    metrics = {}

    metrics["weeks"] = len(era_scores)
    metrics["Mean correlation"] = era_scores.mean()
    metrics["Median correlation"] = era_scores.median()
    metrics["Std. Dev."] = era_scores.std()
    metrics["Mean Pseudo-Sharpe"] = era_scores.mean()/era_scores.std()
    metrics["Median Pseudo-Sharpe"] = era_scores.median()/era_scores.std()
    metrics["Hit Rate (% positive eras)"] = era_scores.apply(lambda x: np.sign(x)).value_counts()[1]/len(era_scores) * 100
    #metrics["Max Drawdown"] = calculate_max_drawdown(era_scores)

    if plot:

        era_scores.rolling(roll_mean).mean().plot(
            kind="line", title="Rolling Per Era Correlation Mean", figsize=(15, 4)
        )
        plt.axhline(y=0.0, color="black", linestyle="-", linewidth=1)
        plt.axhline(y=era_scores.mean(), color="g", linewidth=1, linestyle="--", label='Mean corr')
        plt.show()

        era_scores.cumsum().plot(title="Cumulative Sum of Era Scores", figsize=(15, 4))
        plt.axhline(y=0.0, color="r", linestyle="--")
        plt.show()
    
    return pd.Series(metrics).round(4)