# A small library to reproduce the scores on numer.ai diagnistics dashboard.

## Installation

`pip install numereval`

### Structure

![Structure](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/numereval_structure.png)

## Numerai main tournament evaluation metrics

### numereval.numereval.evaluate

A generic function to calculate basic per-era correlation stats with optional feature exposure and plotting.

Useful for evaluating custom validation split from training data.

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

Docs will be updated soon!
