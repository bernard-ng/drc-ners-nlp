# Formal Model Specifications

This document formalises the statistical models implemented in
`src/ners/research/models`. Throughout, the training set is
$\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$ with labels
$y^{(i)} \in \{0,1\}$ for the binary gender classes. Feature vectors
$\mathbf{x}^{(i)}$ combine

* character $n$-gram count representations of name strings produced by
  `CountVectorizer` or `TfidfVectorizer`, and
* engineered scalar or categorical metadata (e.g., name length, province)
  that are either used directly or encoded by `LabelEncoder`.

For neural architectures, character or token sequences are converted into
integer index sequences using a `Tokenizer` before being padded to a
maximum length specified in the configuration. Predictions are returned as
class posterior probabilities via a softmax layer unless otherwise noted.

## Logistic Regression (`logistic_regression_model.py`)

**Feature map.** Character $n$-gram counts
$\phi(\mathbf{x}) \in \mathbb{R}^d$ obtained with
`CountVectorizer(analyzer="char", ngram_range=(2,4))` (default configuration).【F:src/ners/research/models/logistic_regression_model.py†L16-L46】

**Model.** The linear logit for class $1$ is
$z = \mathbf{w}^\top \phi(\mathbf{x}) + b$. The class posteriors are
$p(y=1\mid \mathbf{x}) = \sigma(z)$ and $p(y=0\mid \mathbf{x}) = 1 - \sigma(z)$
with $\sigma(u) = (1 + e^{-u})^{-1}$.

**Training objective.** Minimise the regularised negative log-likelihood

$$\mathcal{L}(\mathbf{w}, b) = -\sum_{i=1}^N \left[y^{(i)}\log p(y^{(i)}=1\mid \mathbf{x}^{(i)}) + (1-y^{(i)}) \log p(y^{(i)}=0\mid \mathbf{x}^{(i)})\right] + \lambda R(\mathbf{w}),$$

where $R$ is the penalty induced by the chosen solver (e.g., $\ell_2$ for
`liblinear`).

## Multinomial Naive Bayes (`naive_bayes_model.py`)

**Feature map.** Character $n$-gram counts
$\phi(\mathbf{x}) \in \mathbb{N}^d$ derived with
`CountVectorizer(analyzer="char", ngram_range=(2,4))` by default.【F:src/ners/research/models/naive_bayes_model.py†L15-L38】

**Generative model.** For each class $c \in \{0,1\}$, the class prior is
$\pi_c = \frac{N_c}{N}$. Conditional feature probabilities are estimated with
Laplace smoothing (parameter $\alpha$):

$$\theta_{cj} = \frac{N_{cj} + \alpha}{\sum_{k=1}^d (N_{ck} + \alpha)},$$

where $N_{cj}$ counts the total occurrences of feature $j$ among examples of
class $c$. The likelihood of an input with counts $\phi_j(\mathbf{x})$ is

$$p(\phi(\mathbf{x})\mid y=c) = \prod_{j=1}^d \theta_{cj}^{\phi_j(\mathbf{x})}.$$

**Inference.** Predict with the maximum a posteriori (MAP) decision
$\hat{y} = \arg\max_c \left\{ \log \pi_c + \sum_{j=1}^d \phi_j(\mathbf{x}) \log \theta_{cj} \right\}$.

## Random Forest (`random_forest_model.py`)

**Feature map.** Concatenation of engineered numerical features and label
encoded categorical attributes produced on demand in
`prepare_features`.【F:src/ners/research/models/random_forest_model.py†L28-L71】

**Model.** An ensemble of $T$ decision trees ${\{T_t\}}_{t=1}^T$, each trained on
a bootstrap sample of the data with random feature sub-sampling. Each tree
outputs a class prediction $T_t(\mathbf{x}) \in \{0,1\}$. The forest prediction
is the mode of individual votes:

$$\hat{y} = \operatorname{mode}\{T_t(\mathbf{x}) : t = 1,\dots,T\}.$$ 

**Class probability.** For soft outputs,
$p(y=1\mid \mathbf{x}) = \frac{1}{T} \sum_{t=1}^T p_t(y=1\mid \mathbf{x})$ where
$p_t$ is the class distribution estimated at the leaf reached by
$\mathbf{x}$ in tree $t$.

## LightGBM (`lightgbm_model.py`)

**Feature map.** Hybrid of numeric inputs, categorical label encodings, and
character $n$-gram counts expanded into dense columns and assembled into a
feature matrix persisted in `self.feature_columns`.【F:src/ners/research/models/lightgbm_model.py†L38-L118】

**Model.** Gradient boosted decision trees forming an additive function
$F_M(\mathbf{x}) = \sum_{m=0}^M \eta h_m(\mathbf{x})$, where $h_m$ denotes the
$m$-th tree and $\eta$ is the learning rate.

**Training objective.** LightGBM minimises

$$\mathcal{L}^{(m)} = \sum_{i=1}^N \ell\big(y^{(i)}, F_{m-1}(\mathbf{x}^{(i)}) + h_m(\mathbf{x}^{(i)})\big) + \Omega(h_m),$$

using second-order Taylor approximations of the loss $\ell$ (binary log-loss by
default) and regulariser $\Omega$ determined by tree complexity constraints.

## XGBoost (`xgboost_model.py`)

**Feature map.** Combination of numeric metadata, categorical label encodings,
and character $n$-gram counts as described in `prepare_features`.【F:src/ners/research/models/xgboost_model.py†L41-L113】

**Model.** Additive ensemble of regression trees
$F_M(\mathbf{x}) = \sum_{m=1}^M f_m(\mathbf{x})$ with $f_m \in \mathcal{F}$, the
space of trees with fixed structure.

**Training objective.** At boosting iteration $m$, minimise the regularised
objective

$$\mathcal{L}^{(m)} = \sum_{i=1}^N \ell\big(y^{(i)}, F_{m-1}(\mathbf{x}^{(i)}) + f_m(\mathbf{x}^{(i)})\big) + \Omega(f_m),$$

where $\Omega(f) = \gamma T_f + \tfrac{1}{2} \lambda \sum_{j=1}^{T_f} w_j^2$ penalises the
number of leaves $T_f$ and their scores $w_j$. The optimal leaf weights follow

$$w_j^{\star} = - \frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda},$$

with $g_i$ and $h_i$ denoting first- and second-order gradients of the loss for
sample $i$.

## Convolutional Neural Network (`cnn_model.py`)

**Input encoding.** Character-level token sequences padded to length
$L$ using `Tokenizer(char_level=True)` followed by `pad_sequences`.【F:src/ners/research/models/cnn_model.py†L23-L64】

**Architecture.** Embedding layer producing $X \in \mathbb{R}^{L \times d}$,
followed by two convolutional blocks:

1. $H^{(1)} = \operatorname{ReLU}(\operatorname{Conv1D}_{k_1}(X))$ with kernel size
   $k_1$ and $F$ filters, then temporal max-pooling.
2. $H^{(2)} = \operatorname{ReLU}(\operatorname{Conv1D}_{k_2}(H^{(1)}))$ with kernel
   size $k_2$ and $F$ filters.

Global max-pooling yields $h = \max_{t} H^{(2)}_{t,:}$, which passes through a
dense layer and dropout before the softmax layer producing
$p(y\mid \mathbf{x}) = \operatorname{softmax}(W h + b)$.

**Loss.** Cross-entropy between softmax output and the ground-truth label.

## Bidirectional GRU (`bigru_model.py`)

**Input encoding.** Word-level sequences padded to length $L$ with
`Tokenizer(char_level=False)` and `pad_sequences`.【F:src/ners/research/models/bigru_model.py†L47-L69】

**Recurrent dynamics.** A stacked bidirectional GRU computes forward and
backward hidden states according to

\[
\begin{aligned}
\mathbf{z}_t &= \sigma(W_z \mathbf{x}_t + U_z \mathbf{h}_{t-1} + \mathbf{b}_z),\\
\mathbf{r}_t &= \sigma(W_r \mathbf{x}_t + U_r \mathbf{h}_{t-1} + \mathbf{b}_r),\\
\tilde{\mathbf{h}}_t &= \tanh\big(W_h \mathbf{x}_t + U_h(\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h\big),\\
\mathbf{h}_t &= (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t.
\end{aligned}
\]

The final representation concatenates the last forward and backward states
before passing through dense layers and a softmax classifier.

## Bidirectional LSTM (`lstm_model.py`)

**Input encoding.** Word-level sequences padded to length $L$ using the same
pipeline as the BiGRU model.【F:src/ners/research/models/lstm_model.py†L45-L67】

**Recurrent dynamics.** At each timestep, the LSTM updates its memory cell via

\[
\begin{aligned}
\mathbf{i}_t &= \sigma(W_i \mathbf{x}_t + U_i \mathbf{h}_{t-1} + \mathbf{b}_i),\\
\mathbf{f}_t &= \sigma(W_f \mathbf{x}_t + U_f \mathbf{h}_{t-1} + \mathbf{b}_f),\\
\mathbf{o}_t &= \sigma(W_o \mathbf{x}_t + U_o \mathbf{h}_{t-1} + \mathbf{b}_o),\\
\tilde{\mathbf{c}}_t &= \tanh(W_c \mathbf{x}_t + U_c \mathbf{h}_{t-1} + \mathbf{b}_c),\\
\mathbf{c}_t &= \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t,\\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{c}_t).
\end{aligned}
\]

Bidirectional aggregation concatenates terminal forward/backward hidden vectors
before the dense-softmax head.

## Transformer Encoder (`transformer_model.py`)

**Input encoding.** Token sequences padded to a fixed length with positional
indices $\{0, \ldots, L-1\}$ added through a learned positional embedding.
`Tokenizer` initialises the vocabulary; padding uses `pad_sequences`.【F:src/ners/research/models/transformer_model.py†L25-L77】

**Architecture.** For hidden dimension $d$, the encoder block computes

\[
\begin{aligned}
Z^{(0)} &= X + P,\\
Z^{(1)} &= \operatorname{LayerNorm}\big(Z^{(0)} + \operatorname{Dropout}(\operatorname{MHAttn}(Z^{(0)}))\big),\\
Z^{(2)} &= \operatorname{LayerNorm}\big(Z^{(1)} + \operatorname{Dropout}(\phi(Z^{(1)} W_1 + b_1) W_2 + b_2)\big),
\end{aligned}
\]

where $\operatorname{MHAttn}$ is multi-head self-attention with
$H$ heads. Global average pooling produces a fixed-length vector for the dense
and dropout layers before the final softmax classifier.

## Ensemble Voting (`ensemble_model.py`)

**Base learners.** A configurable set of pipelines that include character
$n$-gram vectorisers and classical classifiers (logistic regression,
random forest, naive Bayes).【F:src/ners/research/models/ensemble_model.py†L29-L96】

**Aggregation.** Given model posteriors $\mathbf{p}_j(\mathbf{x})$ and non-negative
weights $w_j$, the soft-voting ensemble predicts

$$p(y=c\mid \mathbf{x}) = \frac{1}{\sum_j w_j} \sum_j w_j \, p_j(y=c\mid \mathbf{x}).$$

Hard voting instead returns
$\hat{y} = \operatorname{mode}\{\arg\max_c p_j(y=c\mid \mathbf{x})\}$.
