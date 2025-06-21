# NERS-NLP: A Culturally-Aware Natural Language Processing System with Named Entity Recognition and Gender Inference Models

Despite the growing success of Named Entity Recognition (NER) systems and gender inference models in Natural Language Processing (NLP), these tools often underperform when applied to culturally diverse African contexts due to the lack of culturally-representative training data. In this paper, we propose NERS-NLP, a culturally-aware NLP system with Named Entity Recognition and Gender Inference Models. This study introduces a large-scale dataset of over 7 million names of the population of the Democratic Republic of Congo (DRC) annotated with gender and demographic metadata, including geographical distribution. We explore the linguistic and sociocultural features embedded in these names and examine their impact on two key NLP tasks, namely, entity recognition and gender classification.
Our approach involves (1) a statistical and feature analysis of Congolese name structures, (2) the development of supervised gender prediction models leveraging name components and demographic patterns, and (3) the integration of the curated name lexicon into NER pipelines to improve recognition accuracy for Congolese entities. Experiments conducted on custom evaluation sets, including multilingual and code-switched Congolese texts, show that our culturally-aware methods significantly outperform state-of-the-art multilingual baselines.
This work demonstrates the importance of culturally grounded resources in reducing bias and improving performance in NLP systems applied to underrepresented regions. Our findings open new directions for inclusive language technologies in African contexts and contribute a valuable resource for future research in regional linguistics, onomastics, and identity-aware artificial intelligence.


# Usage
```bash
git clone https://github.com/bernard-ng/drc-ners-nlp.git
cd drc-ners-nlp

python3 -m venv .venv
source .venv/bin/activate
cp .env .env.local
make download

pip install -r requirements.txt
```

## Gender Inference
### 1. Training

```bash
python -m ners.gender.models.lstm --dataset names.csv --size 1000000 --save
python -m ners.gender.models.logreg --dataset names.csv --size 1000000 --save
python -m ners.gender.models.transformer --dataset names.csv --size 1000000 --save
```

### 2. Evaluation
```bash
python -m ners.gender.eval --dataset eval.csv --model logreg --threshold 0.5 --size 20000
python -m ners.gender.eval --dataset eval.csv --model lstm 
python -m ners.gender.eval --dataset eval.csv --model transformer
```

### 3. Inference
```bash
python -m ners.gender.predict --model logreg --name "Tshisekedi"
python -m ners.gender.predict --model lstm --name "Ilunga" "Albert" "Ilunga Albert" --threshold 0.7
python -m ners.gender.predict --model transformer --name "musenga wa musenga"
```
