# A small library to reproduce the scores on numer.ai diagnistics dashboard.

## Installation

`pip install numereval`

![Structure](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/numereval_structure.png)

### Numerai main tournament evaluation metrics

### Usage:

### numereval.numereval.evaluate:

A generic function to calculate basic per-era correlation stats with optional feature exposure and plotting.

Useful for evaluating custom validation split from training data.
```
evaluate(training_data, plot=True, feature_exposure=False)

---

mean            0.105676
std             0.027988
sharpe          3.775714
max_drawdown    -0.000000
```
![TRaining evaluation](https://github.com/parmarsuraj99/numereval/raw/master/images/training_eval.png)

### numereval.numereval.diagnostics:

To reproduce the scores on diagnostics dashboard locally with optional plotting of per-era correlations.

```
from numereval.numereval import evaluate

validation_data = tournament_data[tournament_data.data_type == "validation"]


evaluate(validation_data, example_preds_loc = "numerai_dataset_243\example_predictions.csv")
```

Validation plot

![Sample output](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/nmr_eval.png)

Returned metrics

![returned dataframe](https://raw.githubusercontent.com/parmarsuraj99/numereval/master/images/numertest.png)


Docs will be updated soon!
