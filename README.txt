# Student Academic Performance Prediction Using Machine Learning

A complete supervised machine learning project that predicts whether a student will **Pass or Fail** based on academic, demographic, social, and behavioral features.

**Authors:** Yazan Hijazi · Hasan Jitan  
**Dataset:** [Student Performance Data — Kaggle (Devansodariya)](https://www.kaggle.com/datasets/devansodariya/student-performance-data)

---

## Overview

The final grade column `G3` (0–20 scale) is converted into a binary classification target:

- **Pass (1)** → G3 ≥ 10  
- **Fail (0)** → G3 < 10

Four classifiers are trained and compared with hyperparameter tuning via GridSearchCV. A second experiment is run with `G1` and `G2` removed to show how much previous grades drive the prediction.

---

## Results Summary

### Main Experiment (with G1 & G2)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | **0.8734** | **0.9574** | **0.8491** | **0.9000** |
| **SVM** ⭐ | **0.8734** | **0.9574** | **0.8491** | **0.9000** |
| Decision Tree | 0.8608 | 0.9773 | 0.8113 | 0.8866 |
| Random Forest | 0.8481 | 0.9184 | 0.8491 | 0.8824 |

### Without G1 & G2 (ablation experiment)

| Model | Accuracy | F1-score |
|---|---|---|
| Random Forest | 0.6709 | 0.7969 |
| Logistic Regression | 0.6582 | 0.7611 |
| SVM | 0.6582 | 0.7611 |
| Decision Tree | 0.6456 | 0.7586 |

Removing previous grades drops F1 by roughly **0.13 points**, confirming that G1 and G2 are the strongest predictors.

---

## Project Structure

```
├── Student_Performance_ML_Project.ipynb   # Full implementation notebook
├── student_data.csv                       # Dataset (395 students, 33 features)
├── Student_Performance_Final_Report.pdf   # Written report
└── outputs/
    ├── model_comparison_results.csv       # Main experiment metrics
    ├── model_comparison_without_G1_G2.csv # Ablation experiment metrics
    ├── feature_importance.csv             # Permutation importance scores
    ├── confusion_matrices.json            # Confusion matrix data for all models
    ├── model_performance_comparison.png   # Bar chart: all model metrics
    ├── best_confusion_matrix.png          # Confusion matrix for best model
    ├── feature_importance.png             # Permutation importance chart
    ├── target_distribution.png            # Pass vs Fail class distribution
    └── top_correlations.png               # Top numeric correlations with G3
```

---

## Dataset

- **395** student records, **33** original features
- **265 Pass** / **130 Fail** after target conversion
- No missing values
- Mix of numerical features (age, studytime, failures, absences, G1, G2, …) and categorical features (school, sex, address, parent jobs, family support, internet access, …)

---

## Methodology

1. **Target creation** — convert G3 into binary Pass/Fail
2. **Preprocessing** — OneHotEncoding for categorical features, StandardScaler for numerical features, all inside a `sklearn.Pipeline`
3. **Train/test split** — 80/20 with stratified sampling
4. **Models** — Logistic Regression, Decision Tree, Random Forest, SVM
5. **Tuning** — GridSearchCV with 3-fold StratifiedKFold, optimizing F1-score
6. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrix, permutation feature importance
7. **Ablation** — repeat without G1 and G2 to isolate their contribution

---

## Key Findings

- **G2 is by far the most important feature** (permutation importance: 0.1452), followed by G1 (0.0331). Previous academic performance dominates the prediction.
- **Linear models win** — Logistic Regression and SVM match or outperform tree-based models, suggesting the decision boundary is largely linear when G1/G2 are included.
- **Without G1/G2**, best F1 drops from 0.90 to ~0.80, and accuracy from 0.87 to 0.67. The models still learn, but prediction is meaningfully harder.
- Secondary predictors include age, family relationship quality (`famrel`), going out frequency (`goout`), and parent cohabitation status (`Pstatus`).

---

## How to Run

1. Clone the repository and place `student_data.csv` in the same folder as the notebook.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
3. Open and run the notebook:
   ```bash
   jupyter notebook Student_Performance_ML_Project.ipynb
   ```
   Or open directly in **Google Colab** or **VS Code**.

---

## Dependencies

- Python 3.x
- numpy, pandas, matplotlib
- scikit-learn

---

## References

1. Devansodariya. *Student Performance Data.* Kaggle. https://www.kaggle.com/datasets/devansodariya/student-performance-data  
2. Scikit-learn Documentation. https://scikit-learn.org/  
3. Pandas Documentation. https://pandas.pydata.org/  
4. Matplotlib Documentation. https://matplotlib.org/
