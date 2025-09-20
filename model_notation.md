# Model Notation Reference

This document summarises the mathematical formulation and notation behind the models available in `research/models`. In all cases, the input example is represented by a feature vector $\mathbf{x}$ (after any feature-extraction or vectorisation steps) and the target label belongs to a finite set of classes $\mathcal{Y}$.

## Logistic Regression
- Decision function: $z = \mathbf{w}^\top \mathbf{x} + b$.
- Binary posterior: $p(y=1\mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$ and $p(y=0\mid \mathbf{x}) = 1 - \sigma(z)$.
- Multi-class (one-vs-rest or softmax): $p(y=c\mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^\top \mathbf{x} + b_c)}{\sum_{k \in \mathcal{Y}} \exp(\mathbf{w}_k^\top \mathbf{x} + b_k)}$.
- Loss: negative log-likelihood $\mathcal{L} = -\sum_i \log p(y_i\mid \mathbf{x}_i)$ plus regularisation when configured.
- Gender prediction rationale: linear decision boundaries over character n-gram counts provide a strong, interpretable baseline for name-based gender attribution.
- Implementation notes: uses character n-grams via `CountVectorizer`; `solver='liblinear'` with optional `class_weight` and `n_jobs` to speed up sparse optimization.

## Multinomial Naive Bayes
- Class prior: $p(y=c) = \frac{N_c}{N}$ where $N_c$ counts training instances in class $c$.
- Conditional likelihood (bag-of-ngrams): $p(\mathbf{x}\mid y=c) = \prod_{j=1}^{d} p(x_j\mid y=c)^{x_j}$ with categorical parameters estimated via Laplace smoothing.
- Posterior up to normalisation: $\log p(y=c\mid \mathbf{x}) \propto \log p(y=c) + \sum_{j=1}^{d} x_j \log p(x_j\mid y=c)$.
- Gender prediction rationale: captures the relative frequency of character patterns associated with each gender, giving a fast and robust probabilistic baseline for sparse n-gram features.
- Implementation notes: character n-gram counts with Laplace smoothing; extremely fast to train and deploy.

## Support Vector Machine (RBF Kernel)
- Dual-form decision function: $f(\mathbf{x}) = \operatorname{sign}\Big( \sum_{i=1}^{M} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \Big)$.
- RBF kernel: $K(\mathbf{x}_i, \mathbf{x}) = \exp\big(-\gamma \lVert \mathbf{x}_i - \mathbf{x} \rVert_2^2\big)$.
- Soft-margin optimisation: $\min_{\mathbf{w}, \xi} \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 + C \sum_i \xi_i$ s.t. $y_i(\mathbf{w}^\top \phi(\mathbf{x}_i) + b) \geq 1 - \xi_i$, $\xi_i \geq 0$.
- Gender prediction rationale: non-linear kernels model subtle character-pattern interactions beyond linear baselines, improving separability when male and female names share prefixes but diverge in internal structure.
- Implementation notes: TF–IDF character features; increased `cache_size` and optional `class_weight` for stability on imbalanced data.

## Random Forest
- Ensemble of $T$ decision trees: $\hat{y} = \operatorname{mode}\{ T_t(\mathbf{x}) : t=1, \dots, T \}$ for classification.
- Each tree draws a bootstrap sample of the training set and a random subset of features at each split.
- Feature importance (used in implementation): mean decrease in impurity aggregated over splits per feature.
- Gender prediction rationale: handles heterogeneous engineered features (length, province, endings) without heavy preprocessing, while delivering interpretable feature-importance signals.
- Implementation notes: enables `n_jobs=-1` for parallel trees; persistent label encoders ensure stable categorical mappings.

## LightGBM (Gradient Boosted Trees)
- Additive model: $F_0(\mathbf{x}) = \hat{p}$ (initial prediction), $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta h_m(\mathbf{x})$.
- Each weak learner $h_m$ is a decision tree grown with leaf-wise strategy and depth constraint.
- Optimises differentiable loss (default: logistic) using first- and second-order gradients over data in each boosting iteration.
- Gender prediction rationale: excels with sparse categorical encodings and numerous engineered features, offering strong accuracy with manageable inference cost.
- Implementation notes: `objective='binary'`, `n_jobs=-1` for throughput; works well with compact character-gram features plus metadata.

## XGBoost
- Objective: $\mathcal{L}^{(t)} = \sum_{i} \ell(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$ with regulariser $\Omega(f) = \gamma T + \frac{1}{2} \lambda \sum_j w_j^2$.
- Tree score expansion via second-order Taylor approximation; optimal leaf weight $w_j = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$ where $g_i$ and $h_i$ are gradient and Hessian statistics.
- Final prediction: $\hat{y}(\mathbf{x}) = \sum_{t=1}^{M} \eta f_t(\mathbf{x})$.
- Gender prediction rationale: strong regularisation and gradient-informed splits capture interactions between textual and metadata features; suited to high-stakes deployment when tuned carefully.
- Implementation notes: `tree_method='hist'`, `n_jobs=-1` for efficient CPU training; integrates engineered categorical encodings.

## Convolutional Neural Network (1D)
- Token/character embeddings produce $X \in \mathbb{R}^{L \times d}$.
- Convolution layer: $H^{(k)} = \operatorname{ReLU}(X * W^{(k)} + b^{(k)})$ where $*$ denotes 1D convolution with filter $W^{(k)}$.
- Pooling summarises temporal dimension (max or global max); dense layers map pooled vector to logits $\mathbf{z}$.
- Output probabilities: $p(y=c\mid \mathbf{x}) = \operatorname{softmax}_c(\mathbf{z})$; loss via cross-entropy.
- Gender prediction rationale: convolutional filters learn discriminative prefixes, suffixes, and intra-name motifs directly from characters, accommodating mixed-language inputs.
- Implementation notes: adds `SpatialDropout1D` on embeddings and `padding='same'` in conv layers for stability and length-invariance.

## Bidirectional GRU
- Forward GRU recursion: $\begin{aligned}
&\mathbf{z}_t = \sigma(W_z \mathbf{x}_t + U_z \mathbf{h}_{t-1} + \mathbf{b}_z),\\
&\mathbf{r}_t = \sigma(W_r \mathbf{x}_t + U_r \mathbf{h}_{t-1} + \mathbf{b}_r),\\
&\tilde{\mathbf{h}}_t = \tanh(W_h \mathbf{x}_t + U_h(\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_h),\\
&\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t.
\end{aligned}$
- Backward GRU mirrors the recurrence from $t=L$ to $1$; final representation concatenates $[\mathbf{h}_L^{\rightarrow}; \mathbf{h}_1^{\leftarrow}]$ before dense layers and softmax output.
- Gender prediction rationale: bidirectional context processes character sequences in both directions, learning gender-specific morphemes appearing at any position within the name.
- Implementation notes: `Embedding(mask_zero=True)` propagates masks to GRUs, ignoring padding; optional `recurrent_dropout` reduces overfitting.

## LSTM
- Gates per timestep: $\begin{aligned}
&\mathbf{i}_t = \sigma(W_i \mathbf{x}_t + U_i \mathbf{h}_{t-1} + \mathbf{b}_i),\\
&\mathbf{f}_t = \sigma(W_f \mathbf{x}_t + U_f \mathbf{h}_{t-1} + \mathbf{b}_f),\\
&\mathbf{o}_t = \sigma(W_o \mathbf{x}_t + U_o \mathbf{h}_{t-1} + \mathbf{b}_o),\\
&\tilde{\mathbf{c}}_t = \tanh(W_c \mathbf{x}_t + U_c \mathbf{h}_{t-1} + \mathbf{b}_c),\\
&\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t,\\
&\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t).
\end{aligned}$
- Bidirectional stacking concatenates final hidden vectors before classification via softmax.
- Gender prediction rationale: long short-term memory cells model long-range dependencies within names, capturing compound structures common in multilingual gendered naming conventions.
- Implementation notes: `Embedding(mask_zero=True)` and `recurrent_dropout` regularise sequence modeling across padded batches.

## Transformer Encoder (Single Block)
- Input embeddings $X \in \mathbb{R}^{L \times d}$ plus positional embeddings $P$ produce $Z^{(0)} = X + P$.
- Multi-head self-attention: $\operatorname{MHAttn}(Z) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_H) W^O$ where $\text{head}_h = \operatorname{softmax}\big(\frac{Q_h K_h^\top}{\sqrt{d_k}}\big) V_h$ and $(Q_h, K_h, V_h) = (Z W_h^Q, Z W_h^K, Z W_h^V)$.
- Add & norm: $Z^{(1)} = \operatorname{LayerNorm}(Z^{(0)} + \operatorname{Dropout}(\operatorname{MHAttn}(Z^{(0)})))$.
- Position-wise feed-forward: $Z^{(2)} = \operatorname{LayerNorm}(Z^{(1)} + \operatorname{Dropout}(\phi(Z^{(1)} W_1 + b_1) W_2 + b_2))$, with activation $\phi(\cdot)$ (ReLU).
- Sequence pooling (global average) feeds dense layers and softmax classifier.
- Gender prediction rationale: self-attention captures global dependencies and shared subword units across names, outperforming recurrent models when sufficient labelled data is available; otherwise risk of overfitting should be monitored.
- Implementation notes: `Embedding(mask_zero=True)` with learned positional embeddings; attention dropout (`attn_dropout`) and classifier dropout improve generalisation.

## Ensemble (Soft Voting)
- Base learners indexed by $j$ output probability vectors $\mathbf{p}_j(\mathbf{x})$.
- Aggregated prediction with weights $w_j$: $p(y=c\mid \mathbf{x}) = \frac{1}{\sum_j w_j} \sum_j w_j \, p_j(y=c\mid \mathbf{x})$.
- Hard voting variant predicts $\hat{y} = \operatorname{mode}\{ \hat{y}_j \}$, where $\hat{y}_j = \arg\max_c p_j(y=c\mid \mathbf{x})$.
- Gender prediction rationale: blends complementary inductive biases (linear, tree-based, neural) to reduce variance on ambiguous names; remains suitable provided individual members are well-calibrated.
