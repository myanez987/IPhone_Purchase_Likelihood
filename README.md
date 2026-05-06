[README_IPhone_Predictor.md](https://github.com/user-attachments/files/27426322/README_IPhone_Predictor.md)
# 📱 iPhone Purchase Likelihood Predictor

A K-Nearest Neighbors (KNN) classification model that predicts whether a customer is likely to purchase an iPhone based on their age, salary, and gender — and segments likely buyers into premium vs. financing profiles for targeted business insights.

---

## Overview

This project applies supervised machine learning to a customer purchase dataset to answer the question: **given a person's demographics, will they buy an iPhone?**

Beyond a binary yes/no prediction, the model adds a business intelligence layer — if the customer is predicted to buy, it further classifies them as a **premium buyer** (salary > $70k) or a **likely financing customer** ($10k–$70k salary), enabling more targeted sales and marketing strategies.

---

## Model Performance

| Metric | Score |
|---|---|
| Precision | 0.83 |
| Recall | 0.82 |
| F1 Score | 0.81 |

Metrics computed using weighted averaging across classes on a 30% held-out test set.

---

## How It Works

### 1. Data Preprocessing
- **Label Encoding** — converts the `Gender` column (`Male`/`Female`) from a string to a numeric integer (`1`/`0`) so the KNN algorithm can process it
- The original `Gender` column is dropped after encoding

### 2. Train/Test Split
- 70% training / 30% test
- `random_state=42` for reproducibility

### 3. Hyperparameter Tuning
- K values from 1 to 20 are evaluated using Euclidean distance
- The optimal K is selected based on the highest accuracy score on the test set

### 4. Final Model
- `KNeighborsClassifier(n_neighbors=5)` trained on the full training set
- Predictions evaluated with precision, recall, and F1 score

### 5. Interactive Prediction
The notebook prompts for user inputs at runtime (age, salary, gender) and returns one of three business-oriented outputs:

| Condition | Output |
|---|---|
| Predicted buyer + salary > $70,000 | *"Likely to purchase iPhone. Target as a premium customer."* |
| Predicted buyer + salary $10k–$70k | *"Will most likely finance iPhone."* |
| Not predicted to buy | *"Will not purchase iPhone."* |

---

## Dataset

The model trains on `iphone_purchase_records.csv`, which contains:

| Column | Type | Description |
|---|---|---|
| `Gender` | Categorical | Male / Female |
| `Age` | Numeric | Customer age |
| `Salary` | Numeric | Annual salary |
| `Purchase Iphone` | Binary | Target variable (1 = purchased, 0 = did not) |

> **Note:** Bring your own dataset. The CSV is not included in this repo. Any dataset with the same four columns will work.

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/your-username/iphone-purchase-predictor.git
cd iphone-purchase-predictor
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn
```

**3. Add your dataset**

Place `iphone_purchase_records.csv` in the root directory with columns: `Gender`, `Age`, `Salary`, `Purchase Iphone`.

**4. Run the notebook**
```bash
jupyter notebook IPhone_Purchases.ipynb
```

Run all cells. The final cell will prompt you to enter customer data and return a prediction.

---

## Example Prediction

```
Enter value for Age: 28
Enter value for Salary: 85000
Enter value for Gender_Encoded (Male = 1, Female = 0): 1

→ Prediction: Likely to purchase iPhone. Target as a premium customer.
```

---

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | KNN model, label encoding, train/test split, metrics |

---

## Project Structure

```
iphone-purchase-predictor/
├── IPhone_Purchases.ipynb        # Full notebook
├── iphone_purchase_records.csv   # Dataset (not included — bring your own)
└── README.md
```

---

## Potential Extensions

- **Feature scaling** — add `StandardScaler` before fitting; KNN is distance-based so unscaled salary values can dominate age in Euclidean distance calculations
- **Additional features** — incorporate credit score, device brand history, or region for richer predictions
- **Alternative classifiers** — compare KNN against Logistic Regression, Random Forest, or SVM for this binary classification task
- **Web interface** — wrap the prediction input/output in a Streamlit or Gradio UI to make it accessible without running a notebook

---

## License

MIT License. See `LICENSE` for details.
