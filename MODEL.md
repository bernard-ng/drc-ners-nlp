# Comparative model notation

Let the published observations be

\[
\mathcal{D}=\{(N_i,y_i)\}_{i=1}^{n}, \qquad y_i \in \{\mathrm{f},\mathrm{m}\},
\]

where \(N_i\) is a complete name string and \(y_i\) is sex. Here, sex means the `f` or `m`
marker in the source records, not gender identity.

## Controlled surname ablation contract

The primary comparison uses normalized records with exactly three tokens. Write

\[
N_i=(a_{i1},a_{i2},s_i), \qquad A_i=(a_{i1},a_{i2}),
\]

where \(A_i\) is the operational native-name view and \(s_i\) is the presumed surname.
The two model inputs are \(N_i\) (surname included) and \(A_i\) (native only).

A namespaced BLAKE2 hash \(h(A)\) maps every distinct native name to one of 10,000
buckets. It is independent of dataframe-library hash implementations. For a test
fraction \(q\), observation \(i\) is held out when

\[
h(A_i) \bmod 10000 < \operatorname{round}(10000q).
\]

Grouping on \(A_i\), rather than the full name, is essential: different surnames attached
to the same native name cannot cross the train-test boundary after surname removal. An
independent hash of \(A_i\) selects the study sample. The full and native-only variants
therefore contain identical source rows and held-out groups. This is stricter than grouping
only exact full-name duplicates and avoids representation-induced leakage.

The study records accuracy, balanced accuracy, weighted precision/recall/F1,
macro-F1, Matthews correlation coefficient (MCC), and the confusion matrix ordered as
\((\mathrm{f},\mathrm{m})\). Cross-validation, when enabled, uses stratified native-name
groups and refits the vectorizer inside every fold.

### Corpus audit motivating the contract

The current local corpus has 7,500,097 usable labeled rows. Exactly 5,901,580 rows (78.7%)
have three tokens; this dominant cohort contains 2,367,836 F and 3,533,744 M records. It is
large enough to test the stated token-order assumptions without inventing parsing rules for
the long tail of one-, two-, and four-plus-token names.

Within the three-token cohort, 115,934 first-two-token groups have both labels, covering
1,025,305 rows. By contrast, only 2,290 exact full-name groups have both labels, covering
9,115 rows. The native-only sequence is therefore genuinely much more label-ambiguous.
Those observations are retained and assigned to one split as a group; choosing a label
using global counts would leak held-out information and hide the task's irreducible
ambiguity.

## Character feature models

For a character n-gram \(g\), let \(c_g(N)\) be its count in full name \(N\).

### Logistic regression

`MaxAbsScaler` divides each n-gram column by its largest absolute training value. Let
\(\widetilde{\phi}(N)\) denote the resulting vector. Logistic regression maps it to

\[
p(y=1\mid N)=\sigma(w^\top\widetilde{\phi}(N)+b),
\]

The estimator minimizes class-weighted, L2-regularized binary log loss with SAGA. The
scaler preserves zero entries and gives SAGA comparable feature ranges without converting
the sparse matrix to a dense matrix.

### Position-aware logistic regression

For the controlled three-token comparison, a feature union concatenates character TF-IDF
vectors for the whole name and for each token position:

\[
\Phi(N)=
\left[
\phi_{\mathrm{sequence}}(N);
\phi_1(t_1);
\phi_2(t_2);
\phi_3(t_3)
\right].
\]

A class-weighted logistic regression classifier receives \(\Phi(N)\). In the native-only
view, the input contains the first two tokens, so the third-position channel receives a
constant missing value. This keeps the estimator structure fixed while the paired
experiment measures the information added by the third token.

### Multinomial Naive Bayes

For class \(c\), smoothed n-gram probabilities are

\[
\theta_{cg}=\frac{C_{cg}+\alpha}{\sum_j(C_{cj}+\alpha)}.
\]

Prediction maximizes the class prior plus the summed n-gram log likelihood.

### Random forest, LightGBM, and XGBoost

The templates cap these models at 4,096 sparse TF-IDF features:

\[
\operatorname{tfidf}_g(N)=\operatorname{tf}_g(N)\operatorname{idf}_g.
\]

The random forest aggregates bootstrap-tree votes. LightGBM and XGBoost learn additive
tree ensembles by minimizing regularized binary loss. The shared sparse representation
avoids treating each unique full name as a categorical integer and makes unseen-name
evaluation meaningful.

## Neural sequence models

Let \((c_1,\ldots,c_L)\) be a padded character sequence and let an embedding layer map
characters to \(X\in\mathbb{R}^{L\times d}\). All neural architectures are character-level.
Whole-word tokenization is intentionally avoided because rare and unseen name tokens would
create a very large embedding vocabulary and poor out-of-vocabulary behavior.

### Character CNN

The CNN applies stacked one-dimensional convolutions and pooling:

\[
H^{(k)}=\operatorname{ReLU}(\operatorname{Conv1D}_k(X)),
\]

followed by global max pooling and a two-class softmax head. It tests whether local
character motifs are sufficient for the target.

### Bidirectional LSTM and BiGRU

Forward and backward recurrent states summarize character order in both directions. The model
concatenates their terminal representations and passes them through a regularized dense
softmax head. The two gated architectures test different recurrent memory mechanisms.

### Transformer encoder

Character and positional embeddings enter multi-head self-attention:

\[
\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

Residual connections, layer normalization, a feed-forward block, global pooling, and a
two-class softmax head complete the model.

Neural training reserves validation groups by a stable hash of the native name, applies
inverse-frequency class weights, stops early on validation loss, and restores the best
epoch. This prevents native-name variants from leaking into early-stopping validation.

## Initial controlled result

The deterministic one-percent experiment on the current local corpus produced 46,193
training rows and 12,207 held-out rows after selecting the three-token cohort. These are
development results, not final full-corpus estimates.

| Model | Full macro-F1 | Native macro-F1 | Surname gain |
| --- | ---: | ---: | ---: |
| Majority-class dummy | 0.3712 | 0.3712 | 0.0000 |
| Random forest | 0.8587 | 0.5937 | 0.2650 |
| Multinomial Naive Bayes | 0.8721 | 0.6293 | 0.2428 |
| XGBoost | 0.8874 | 0.5642 | 0.3232 |
| Soft-voting ensemble | 0.8959 | 0.6095 | 0.2864 |
| Character logistic regression | 0.9088 | 0.6236 | 0.2853 |
| LightGBM | 0.9093 | 0.6137 | 0.2956 |
| Position-aware logistic regression | **0.9205** | **0.6400** | 0.2804 |

The position-aware full model has 0.9231 accuracy and 0.8410 MCC; its native-only
counterpart has 0.6479 accuracy and 0.2815 MCC. XGBoost illustrates why accuracy alone is
not suitable here: its native-only accuracy is 0.6454, but its 0.5642 macro-F1 reveals a
substantial class imbalance in its errors.

The full-minus-native macro-F1 gain is 0.2853 for the standard linear model and 0.2804
for the position-aware model. The largest signed coefficients in the full position-aware
model are overwhelmingly third-token features. Both observations support the surname
influence hypothesis on this sample. The native-only result remains well above chance but
is much weaker, consistent with comparatively gender-neutral native tokens.

The sparse position-aware linear model is the recommended primary architecture. It is more
accurate, smaller, faster, and more interpretable than the tree ensembles on this sample.
Keep plain character logistic regression as the architecture-neutral sanity check and
Naive Bayes as the lightweight probabilistic baseline. The voting ensemble is not retained
as a preferred model because its added complexity does not improve held-out performance.

## Voting ensemble

For base-model posteriors \(p_j(y=c\mid N)\), soft voting estimates

\[
p(y=c\mid N)=\frac{1}{J}\sum_{j=1}^{J}p_j(y=c\mid N).
\]

The ensemble combines logistic regression, random forest, and Naive Bayes to test
whether their complementary inductive biases improve held-out performance.
