# A small library to reproduce the scores on numer.ai diagnistics dashboard.

## Installation

`pip install numereval`

### Structure

![Structure](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/numereval_structure.png)

## Numerai main tournament evaluation metrics

### numereval.numereval.evaluate

A generic function to calculate basic per-era correlation stats with optional feature exposure and plotting.

Useful for evaluating custom validation split from training data without MMC metrics and correlation with example predictions.

```
from numereval.numereval import evaluate

evaluate(training_data, plot=True, feature_exposure=False)
```

Correlations plot      |  Returned metrics
:-------------------------:|:-------------------------:
![Training Correlations](https://github.com/parmarsuraj99/numereval/raw/master/images/training_eval.png)  |  ![Metrics](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/evaluate_metrics.png)

### numereval.numereval.diagnostics

To reproduce the scores on diagnostics dashboard locally with optional plotting of per-era correlations.

```python
from numereval.numereval import diagnostics

validation_data = tournament_data[tournament_data.data_type == "validation"]

diagnostics(
    validation_data,
    plot=True,
    example_preds_loc="numerai_dataset_244\example_predictions.csv",
)

```

Validation plot             |  Returned metrics
:-------------------------:|:-------------------------:
![all eras validation plot](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/nmr_eval.png)  |  ![all eras validation metrics](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/numertest.png)

#### Specific validation eras

specify a list of eras in the format `eras = ["era121", "era122", "era209"]`

```python
validation_data = tournament_data[tournament_data.data_type == "validation"]

eras = validation_data.era.unique()[11:-2]

numereval.numereval.diagnostics(
    validation_data,
    plot=True,
    example_preds_loc="numerai_dataset_244\example_predictions.csv",
    eras=eras,
)

```

Validation plot             |  Returned metrics
:-------------------------:|:-------------------------:
![all eras validation plot](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/nmr_eval_some_eras.png)  |  ![all eras validation metrics](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/numertest_specific_eras.png)


## Numerai Signals evaluation metrics

Note: Since predictions are neutralized to Numerai's internal features before scoring, results from `numereval.signalseval.run_analytics()` do not represent exact diagnostics scores.


```python
import numereval
from numereval.signalseval import run_analytics, score_signals

#after assigning predictions
train_era_scores = train_data.groupby(train_data.index).apply(score_signals)
test_era_scores = test_data.groupby(test_data.index).apply(score_signals)

train_scores = run_analytics(train_era_scores, plot=False)
test_scores = run_analytics(test_era_scores, plot=True)

```

![Test correlation plot](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/signals_test_corr.png)


![Test cumulative correlation plot](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/signals_test_cumulative.png)

train_scores            |  test_scores
:-------------------------:|:-------------------------:
![train_Scores](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/signals_train_scores.png)  |  ![test_Scores](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/signals_test_scores.png)


**Thanks to [Jason Rosenfeld](https://twitter.com/jrosenfeld13)** for allowing the `run_analytics()` to be integrated into the library.

Docs will be updated soon!
