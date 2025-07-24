# NERS-NLP: A Culturally-Aware Natural Language Processing System with Named Entity Recognition and Gender Inference Models

Despite the growing success of Named Entity Recognition (NER) systems and gender inference models in Natural Language Processing (NLP), these tools often underperform when applied to culturally diverse African contexts due to the lack of culturally-representative training data. In this paper, we propose NERS-NLP, a culturally-aware NLP system with Named Entity Recognition and Gender Inference Models. This study introduces a large-scale dataset of over 7 million names of the population of the Democratic Republic of Congo (DRC) annotated with gender and demographic metadata, including geographical distribution. We explore the linguistic and sociocultural features embedded in these names and examine their impact on two key NLP tasks, namely, entity recognition and gender classification.
Our approach involves :

- (1) a statistical and feature analysis of Congolese name structures, 
- (2) the development of supervised gender prediction models leveraging name components and demographic patterns, 
- (3) the integration of the curated name lexicon into NER pipelines to improve recognition accuracy for Congolese entities. 


Experiments conducted on custom evaluation sets, including multilingual and code-switched Congolese texts, show that our culturally-aware methods significantly outperform state-of-the-art multilingual baselines.
This work demonstrates the importance of culturally grounded resources in reducing bias and improving performance in NLP systems applied to underrepresented regions. Our findings open new directions for inclusive language technologies in African contexts and contribute a valuable resource for future research in regional linguistics, onomastics, and identity-aware artificial intelligence.


## Installation
```bash
git clone https://github.com/bernard-ng/drc-ners-nlp.git
cd drc-ners-nlp

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```


## Dataset
### Preparation
| Name             | Description                                                        | Default |
|------------------|--------------------------------------------------------------------|---------|
| --split_eval     | Split into evaluation and featured datasets                        | True    |
| --no-split_eval  | Do not split into evaluation and featured datasets                 |         |
| --split_by_sex   | Split by sex into male/female datasets                            | True    |
| --no-split_by_sex| Do not split by sex into male/female datasets                     |         |

```bash
python -m processing.prepare --split_eval --split_by_sex
```

### Annotation
| Name        | Description                                         | Default        |
|-------------|-----------------------------------------------------|----------------|
| --llm_model | Ollama model name to use                            | llama3.2:3b    |

Example:

```bash
python -m processing.annotate --llm_model=mistral7b
``` 

## Experiments
### Training
| Name           | Description                                      | Default            |
|----------------|--------------------------------------------------|--------------------|
| --dataset      | Path to the dataset file                         | names_featured.csv |
| --size         | Number of samples to use (None for full dataset) | None               |
| --threshold    | Probability threshold for gender classification  | 0.5                |
| --cv           | Number of cross-validation folds                 | None               |
| --save         | Whether to save the trained model                | False              |
| --balanced     | Whether to balance the dataset                   | False              |
| --epochs       | Number of training epochs                        | 10                 |
| --test_size    | Proportion of data to use as test set            | 0.2                |
| --random_state | Random seed for reproducibility                  | 42                 |


Examples: 

```bash
python -m pipelilne.gender.models.lstm --size 1000000 --save
python -m pipelilne.gender.models.logreg --size 1000000 --save
python -m pipelilne.gender.models.transformer --size 1000000 --save
```

```bash
python -m pipelilne.gender.models.lstm --size 1000000 --balanced --save
python -m pipelilne.gender.models.logreg --size 1000000 --balanced --save
python -m pipelilne.gender.models.transformer --size 1000000 --balanced --save
```

### Evaluation
| Name       | Description                                   | Default              |
|------------|-----------------------------------------------|----------------------|
| --model    | Model type: logreg, lstm, or transformer      | (required)           |
| --dataset  | Path to the dataset CSV file                  | names_featured.csv   |
| --size     | Number of rows to load from the dataset       | None                 |
| --balanced | Load balanced dataset                         | False                |
| --threshold| Probability threshold for classification      | 0.5                  |

Examples:

```bash
python -m pipelilne.gender.eval --dataset names_evaluations.csv --model logreg
python -m pipelilne.gender.eval --dataset names_evaluations.csv --model lstm 
python -m pipelilne.gender.eval --dataset names_evaluations.csv --model transformer
```

### Inference
| Name        | Description                              | Default   |
|-------------|------------------------------------------|-----------|
| --model     | Model type: logreg, lstm, or transformer | (required)|
| --names     | One or more names                        | (required)|
| --threshold | Threshold for classification             | 0.5       |

Examples: 

```bash
python -m pipelilne.gender.predict --model logreg --names "Tshisekedi"
python -m pipelilne.gender.predict --model lstm --names "Ilunga Ngandu"
python -m pipelilne.gender.predict --model transformer --names "musenga wa musenga"
```
