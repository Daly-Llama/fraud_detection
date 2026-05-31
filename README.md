# Credit Card Fraud Detection

A decision tree classifier that predicts fraudulent credit card transactions, built with interpretability and regulatory transparency as explicit design requirements alongside predictive performance.

---

## The Business Problem

Credit card fraud cost American cardholders an estimated $6.2 billion in 2024. Banks and financial institutions bear most of that cost, creating a strong incentive to detect fraud before transactions are processed. At the same time, two constraints shape what a fraud detection model can look like in practice:

- **Speed** — transactions must be evaluated almost instantly without disrupting the customer experience
- **Transparency** — banks operate in heavily regulated environments; a model whose decisions can't be explained is a liability, not an asset

These constraints drove the choice of a Decision Tree as the modeling approach: it is fast, efficient, and produces decisions that can be traced, audited, and explained to regulators step by step.

---

## Data

The project uses the [Credit Card Fraud dataset](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud) from Kaggle, containing 1 million historical credit card transactions with 7 features and a binary fraud target.

**Features:**
- `distance_from_home` — distance from the customer's home address to the transaction location
- `distance_from_last_transaction` — distance between the current and previous transaction
- `ratio_to_median_purchase_price` — transaction amount relative to the customer's median purchase
- `repeat_retailer` — whether the customer has transacted with this retailer before
- `used_chip` — whether the transaction used a chip reader
- `used_pin_number` — whether a PIN was entered
- `online_order` — whether the transaction was online

**One data quality note:** The dataset shows an 8.7% fraud rate, which is significantly higher than the Federal Reserve's reported rate of under 1%. This suggests the dataset used oversampling to produce a higher proportion of fraud cases for modeling purposes. Results should be interpreted with that context in mind, and additional testing against live transaction data would be needed before deployment.

---

## Approach

**Preprocessing** was minimal — the dataset had no missing values or duplicates, and no features required transformation or removal based on the correlation matrix. This is likely because the data had already been cleaned before publication.

**Model:** Decision Tree Classifier, with hyperparameters set to limit overfitting:
- `max_depth = 5`
- `min_samples_split = 200`
- `min_samples_leaf = 50`
- `class_weight = 'balanced'`

**Train/test split:** 700,000 training observations, 300,000 held out for testing.

**Cross-validation:** 5-fold cross-validation was run to check for overfitting, given the near-perfect test results.

---

## Results

| Metric | Not Fraud | Fraud |
|--------|-----------|-------|
| Precision | 1.00 | 0.96 |
| Recall | 1.00 | 1.00 |
| F1-Score | 1.00 | 0.98 |

**Overall accuracy: 100%**

The model missed only 2 fraudulent transactions out of 26,131 in the test set. Recall — the most important metric for this problem, since missing real fraud is more costly than a false positive — was effectively perfect.

---

## Decision Tree Interpretability

One of the strengths of this model is that its decisions can be traced explicitly. Two example fraud paths from the tree:

**Path 1 — Online fraud pattern:**
Transaction amount not abnormally large → located more than 100 miles from home → online purchase → no chip → no PIN → **flagged as fraud**

**Path 2 — In-person fraud pattern:**
Transaction amount more than 4x the customer's median → not an online purchase → located less than 2 miles from home → no chip → no PIN → **flagged as fraud**

The full decision tree visualization is included in the repository as `decision_tree.png`.

---

## Deployment Recommendation

Given the strong results, the model is a candidate for deployment — but with two important caveats:

1. **Additional testing on live data is needed** before treating the results as definitive, given the likely oversampling in the training dataset.

2. **The model should flag, not block.** Automatically blocking suspected fraud risks blocking valid transactions, harming customer experience, and inviting regulatory scrutiny. The recommended workflow is to surface flagged transactions to a fraud analyst who can apply judgment. Any automatic holds should include a fast, painless verification path for customers.

---

## Stack

- Python, Pandas
- Scikit-learn (DecisionTreeClassifier, train_test_split, cross_val_score)
- Matplotlib, Seaborn

---

## Data Source

[Credit Card Fraud](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud) — Kaggle (Narayanan, 2022)
