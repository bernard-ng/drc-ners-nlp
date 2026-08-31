# Measuring Sex Predictability in Congolese Names: A Controlled Comparison of Native-Only and Full-Name Models

> work in progress

[![audit](https://github.com/bernard-ng/drc-names-classifier/actions/workflows/audit.yml/badge.svg)](https://github.com/bernard-ng/drc-names-classifier/actions/workflows/audit.yml)
[![quality](https://github.com/bernard-ng/drc-names-classifier/actions/workflows/quality.yml/badge.svg)](https://github.com/bernard-ng/drc-names-classifier/actions/workflows/quality.yml)

---

## Abstract

This study investigates whether Congolese names are reliable indicators of recorded sex. Using a large corpus of names labeled with administrative F/M markers, we compare classifiers under two controlled input settings: native Congolese name components only and the complete name with the third name component included. The performance difference quantifies how much predictive information is contributed by that additional component and provides empirical evidence about the relative sex neutrality of native Congolese names.

## How to cite this work

```bib
to be provided after publication
```

## Workflows

Clone the repository and sync dependencies:

```bash
git clone https://github.com/bernard-ng/drc-names-classifier.git
cd drc-names-classifier
uv sync
```

Place the published [dataset](https://doi.org/10.5281/zenodo.19809985) in `data/dataset/names.csv`

Launch the local experiment interface with:

```bash
uv run drc-names-classifier web
```

## Training

This command runs every available model in both settings:

- surname included: the complete three-part name
- native names only: the first two parts of the name

```bash
uv run drc-names-classifier experiments compare-views
```

The default uses a small one-percent sample. To run the full dataset, add
`--sample-fraction 1` to any experiment command.

## Run one model in both settings

Each command below runs the named model twice: once with the surname included and once
with native names only. The result shows the two scores and the difference between them.

```bash
uv run drc-names-classifier experiments compare-views --name dummy
```

```bash
uv run drc-names-classifier experiments compare-views --name logistic_regression --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name position_logistic_regression --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name naive_bayes --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name random_forest --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name lightgbm --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name xgboost --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name cnn --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name lstm --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name bigru --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name transformer --sample-fraction=1
uv run drc-names-classifier experiments compare-views --name ensemble --sample-fraction=1
```

## Responsible use

The source dataset has coverage and historical biases. Do not use experiment results to
infer gender identity, identify people, or make eligibility, employment, credit, health,
surveillance, or other consequential decisions.
