# CongoNames name-classification experiments

Run experiments that predict the `m` or `f` value recorded in the CongoNames source data
from a person's name. This is a study tool: the recorded value is not gender identity
and must not be used to profile people or make decisions about them.

## Before you start

Place the published dataset here:

```text
data/dataset/names.csv
```

Then install the project once:

```bash
uv sync
```

## Open the web app

```bash
uv run ners web
```

Use the app to start experiments and view their results. Stop it with `Ctrl+C` when you
are finished.

## Run all models

This command runs every available model in both settings:

- surname included: the complete three-part name
- native names only: the first two parts of the name

```bash
uv run ners experiments compare-views
```

The default uses a small one-percent sample. To run the full dataset, add
`--sample-fraction 1` to any experiment command.

## Run one model in both settings

Each command below runs the named model twice: once with the surname included and once
with native names only. The result shows the two scores and the difference between them.

#### Control
```bash
uv run ners experiments compare-views --name dummy
```

#### Full dataset training for every model
```bash
uv run ners experiments compare-views --sample-fraction=1
```

#### Full dataset training for specific models
```bash
uv run ners experiments compare-views --name logistic_regression --sample-fraction=1
uv run ners experiments compare-views --name position_logistic_regression --sample-fraction=1
uv run ners experiments compare-views --name naive_bayes --sample-fraction=1
uv run ners experiments compare-views --name random_forest --sample-fraction=1
uv run ners experiments compare-views --name lightgbm --sample-fraction=1
uv run ners experiments compare-views --name xgboost --sample-fraction=1
uv run ners experiments compare-views --name cnn --sample-fraction=1
uv run ners experiments compare-views --name lstm --sample-fraction=1
uv run ners experiments compare-views --name bigru --sample-fraction=1
uv run ners experiments compare-views --name transformer --sample-fraction=1
uv run ners experiments compare-views --name ensemble --sample-fraction=1
```

## Responsible use

The source dataset has coverage and historical biases. Do not use experiment results to
infer gender identity, identify people, or make eligibility, employment, credit, health,
surveillance, or other consequential decisions.
