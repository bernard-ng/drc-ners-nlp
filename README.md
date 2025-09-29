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
  - "data_splitting"
```

**Running the Pipeline**

```bash
python main.py --env production
```

## NER Processing (Optional)

This project implements a custom named entity recognition (NER) pipeline tailored for Congolese names. 
Its main objective is to accurately identify and tag the different components of a Congolese name, 
specifically distinguishing between the native part and the surname.

```bash
python ner.py --env production
```

Once you've built and train the NER model you can use it to annotate **COMPOSE** name in the original dataset 

**Running the Pipeline with NER Annotation**
```yaml
stages:
  - "data_cleaning"
  - "feature_extraction"
  - "ner_annotation"
  - "data_splitting"
```

**Running the Pipeline with LLM Annotation**
```yaml
stages:
  - "data_cleaning"
  - "feature_extraction"
  - "llm_annotation"
  - "data_splitting"
```

## Experiments

This project provides a modular experiment (model training and evaluation) framework for systematic model comparison and
research iteration. models are defined in the `drc-ners-nlp/research/models` directory.
you can define model features, training parameters, and evaluation metrics in the `research_templates.yaml` file.

**Running Experiments**

```bash
# bigru
python train.py --name="bigru" --type="baseline" --env="production"
python train.py --name="bigru_native" --type="baseline" --env="production"
python train.py --name="bigru_surname" --type="baseline" --env="production"

# cnn
python train.py --name="cnn" --type="baseline" --env="production"
python train.py --name="cnn_native" --type="baseline" --env="production"
python train.py --name="cnn_surname" --type="baseline" --env="production"

# lightgbm
python train.py --name="lightgbm" --type="baseline" --env="production"
python train.py --name="lightgbm_native" --type="baseline" --env="production"
python train.py --name="lightgbm_surname" --type="baseline" --env="production"

# logistic regression
python train.py --name="logistic_regression" --type="baseline" --env="production"
python train.py --name="logistic_regression_native" --type="baseline" --env="production"
python train.py --name="logistic_regression_surname" --type="baseline" --env="production"

# lstm
python train.py --name="lstm" --type="baseline" --env="production"
python train.py --name="lstm_native" --type="baseline" --env="production"
python train.py --name="lstm_surname" --type="baseline" --env="production"

# random forest
python train.py --name="random_forest" --type="baseline" --env="production"
python train.py --name="random_forest_native" --type="baseline" --env="production"
python train.py --name="random_forest_surname" --type="baseline" --env="production"

# svm
python train.py --name="svm" --type="baseline" --env="production"
python train.py --name="svm_native" --type="baseline" --env="production"
python train.py --name="svm_surname" --type="baseline" --env="production"

# naive bayes
python train.py --name="naive_bayes" --type="baseline" --env="production"
python train.py --name="naive_bayes_native" --type="baseline" --env="production"
python train.py --name="naive_bayes_surname" --type="baseline" --env="production"

# transformer
python train.py --name="transformer" --type="baseline" --env="production"
python train.py --name="transformer_native" --type="baseline" --env="production"
python train.py --name="transformer_surname" --type="baseline" --env="production"

# xgboost
python train.py --name="xgboost" --type="baseline" --env="production"
python train.py --name="xgboost_native" --type="baseline" --env="production"
python train.py --name="xgboost_surname" --type="baseline" --env="production"
```

## Web Interface

This project includes a user-friendly web interface built with Streamlit, allowing non-technical users to run
experiments and make predictions without needing to understand the underlying code.

### Running the Web Interface

```bash
streamlit run web/app.py
```

## GPU Acceleration

This project can leverage GPUs for faster training when supported libraries and hardware are available.

- TensorFlow/Keras models (BiGRU, LSTM, CNN, Transformer)
  - Uses GPU automatically if a TensorFlow GPU build is installed.
  - The code enables safe GPU memory growth by default; optionally enable mixed precision for additional speed:
    - Add `mixed_precision: true` in the experiment `model_params` (e.g., in `config/research_templates.yaml`).
  - The final layer outputs are set to float32 for numerical stability under mixed precision.

- spaCy NER
  - Automatically prefers GPU if available; otherwise falls back to CPU.
  - Ensure a compatible CUDA-enabled spaCy/thinc stack is installed to use GPU.

- XGBoost
  - Enable GPU by adding to the experiment `model_params`:
    - `use_gpu: true` (sets `tree_method: gpu_hist` and `predictor: gpu_predictor`).

- LightGBM
  - Enable GPU by adding to the experiment `model_params`:
    - `use_gpu: true` (sets `device: gpu`). Optional: `gpu_platform_id`, `gpu_device_id`.

Example template snippet (GPU on):

```yaml
- name: "lstm_gpu"
  description: "LSTM with GPU + mixed precision"
  model_type: "lstm"
  features: ["full_name"]
  model_params:
    embedding_dim: 128
    lstm_units: 64
    epochs: 5
    batch_size: 128
    use_gpu: true
    mixed_precision: true
  tags: ["gpu", "mixed_precision"]

- name: "xgboost_gpu"
  description: "XGBoost with GPU"
  model_type: "xgboost"
  features: ["full_name"]
  model_params:
    n_estimators: 200
    use_gpu: true
```

Notes:
- Install CUDA‑enabled binaries for TensorFlow/spaCy/LightGBM/XGBoost to actually use GPU.
- If GPU is requested but not available, training will proceed on CPU with a warning.

## Contributors

<a href="https://github.com/bernard-ng/drc-ners-nlp/graphs/contributors" title="show all contributors">
  <img src="https://contrib.rocks/image?repo=bernard-ng/drc-ners-nlp" alt="contributors"/>
</a>

## Acknowledgements
- Map Visualization: [https://data.humdata.org/dataset/anciennes-provinces-rdc-old-provinces-drc](https://data.humdata.org/dataset/anciennes-provinces-rdc-old-provinces-drc)
