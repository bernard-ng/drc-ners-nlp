# A Culturally-Aware NLP System for Congolese Name Analysis and Gender Inference

Despite the growing success of gender inference models in Natural Language Processing (NLP), these tools often
underperform when applied to culturally diverse African contexts due to the lack of culturally-representative training
data.
This project introduces a comprehensive pipeline for Congolese name analysis with a large-scale dataset of over 5
million names from the Democratic Republic of Congo (DRC) annotated with gender and demographic metadata.

## Getting Started

### Installation & Setup

Instructions and command line snippets bellow are provided to help you set up the project environment quickly and
efficiently.
assuming you have Python 3.11 and Git installed and working on a Unix-like system (Linux, macOS, etc.).

**Using Makefile (Recommended)**

```bash
git clone https://github.com/bernard-ng/drc-ners-nlp.git
cd drc-ners-nlp

# Setup environment
make setup
make activate
```

**Manual Setup**

```bash
git clone https://github.com/bernard-ng/drc-ners-nlp.git
cd drc-ners-nlp

# Setup environment
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

pip install --upgrade pip
pip install -r requirements.txt
pip install jupyter notebook ipykernel pytest black flake8 mypy

source .venv/bin/activate
```

## Data Processing

This project includes a robust data processing pipeline designed to handle large datasets efficiently with batching,
checkpointing, and parallel processing capabilities.
step are defined in the `drc-ners-nlp/processing/steps` directory. and configuration to enable them is managed through
the `drc-ners-nlp/config/pipeline.yaml` file.

**Pipeline Configuration**

```yaml
stages:
  - "data_cleaning"
  - "feature_extraction"
  - "llm_annotation"
  - "data_splitting"
```

**Running the Pipeline**

```bash
python main.py --env development
```

## Experiments

This project provides a modular experiment (model training and evaluation) framework for systematic model comparison and
research iteration. models are defined in the `drc-ners-nlp/research/models` directory.
you can define model features, training parameters, and evaluation metrics in the `research_templates.yaml` file.

**Running Experiments**

```bash
python train.py --name="bigru" --type="baseline" --env="development"
python train.py --name="cnn" --type="baseline" --env="development"
python train.py --name="lightgbm" --type="baseline" --env="development"

python train.py --name="logistic_regression_fullname" --type="baseline" --env="development"
python train.py --name="logistic_regression_native" --type="baseline" --env="development"
python train.py --name="logistic_regression_surname" --type="baseline" --env="development"

python train.py --name="lstm" --type="baseline" --env="development"
python train.py --name="random_forest" --type="baseline" --env="development"
python train.py --name="svm" --type="baseline" --env="development"
python train.py --name="naive_bayes" --type="baseline" --env="development"
python train.py --name="transformer" --type="baseline" --env="development"
python train.py --name="xgboost" --type="baseline" --env="development"
```

## Web Interface

This project includes a user-friendly web interface built with Streamlit, allowing non-technical users to run
experiments and make predictions without needing to understand the underlying code.

### Running the Web Interface

```bash
streamlit run app.py
```

## Contributors

<a href="https://github.com/bernard-ng/drc-ners-nlp/graphs/contributors" title="show all contributors">
  <img src="https://contrib.rocks/image?repo=bernard-ng/drc-ners-nlp" alt="contributors"/>
</a>
