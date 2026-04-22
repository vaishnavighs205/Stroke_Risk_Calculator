# Stroke_Risk_Calculator
Use supervised machine learning to determine the weight of multiple symptoms and demographics towards the risk of stroke and predict the likelihood of stroke.

## Dataset
**Source:** [Stroke Risk Prediction Dataset v2](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset-v2) (Kaggle — downloaded automatically at runtime via `kagglehub`)

> ⚠️ This is a *synthetic* dataset created for educational purposes. Results should not be used for real clinical decision-making.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com//stroke-risk-analysis.git
cd stroke-risk-analysis
```

### 2. Install dependencies
```bash
pip install kagglehub numpy pandas matplotlib scikit-learn
```

### 3. Kaggle API credentials (required for dataset download)
1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Place the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
3. Run `chmod 600 ~/.kaggle/kaggle.json` (macOS/Linux only)

## How to Run

Open `stroke_risk_analysis.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook stroke_risk_analysis.ipynb
```

The notebook will automatically download the dataset on first run.

## What the Notebook Does

1. **Data Loading** — Downloads the stroke risk dataset from Kaggle
2. **EDA & Visualization** — Age, gender, and target class distributions
3. **Preprocessing** — Train/test split (80/20), median imputation, standard scaling, one-hot encoding
4. **Model Training** — Trains five classifiers:
   - Decision Tree
   - K-Nearest Neighbors (KNN)
   - Support Vector Machine (SVM)
   - Naïve Bayes
   - Logistic Regression
5. **Evaluation** — Hold-out test set metrics + 5-fold cross-validation
6. **Symptom Importance** — Measures how toggling each risk factor affects the best model's predictions


## Results

| Rank | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | **SVM** ✅ | 0.9879 | 0.9826 | 0.9845 | 0.9835 | 0.9995 |
| 2 | Logistic Regression | 0.9796 | 0.9688 | 0.9759 | 0.9724 | 0.9984 |
| 3 | KNN | 0.9263 | 0.9332 | 0.8615 | 0.8959 | 0.9850 |
| 4 | Naive Bayes | 0.8869 | 0.7972 | 0.9290 | 0.8581 | 0.9645 |
| 5 | Decision Tree | 0.8947 | 0.9163 | 0.7858 | 0.8460 | 0.9547 |

## Key Findings

- **SVM was the best model** — the RBF kernel effectively captured the nonlinear relationship between features and stroke risk
- **Cross-validation scores matched holdout scores**, confirming the model generalizes well and is not overfit
- **High blood pressure** was the strongest single risk factor, raising the model's predicted stroke probability by ~29 percentage points when present (relative importance ≈ 17.5%)
