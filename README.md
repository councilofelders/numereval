# A small library to reproduce the scores on numer.ai diagnistics dashboard.

## Numerai main tournament evaluation metrics

### Installation

`pip install numereval`

### Usage: 

```
from numereval.numereval import evaluate

validation_data = tournament_data[tournament_data.data_type == "validation"]

evaluate(validation_data, example_preds_loc = "path to example_predictions.csv")

---

evaluate(validation_data, example_preds_loc = "numerai_dataset_243\example_predictions.csv")
```

Validation plot

![Sample output](.\images/nmr_eval.png)

Returned metrics

![returned dataframe](.\images/numertest.png)


Docs will be updated soon!
