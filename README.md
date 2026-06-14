# Credit Card Fraud Detection

This project builds a machine learning pipeline to detect fraudulent credit card transactions. The primary challenge addressed is **extreme class imbalance**, which is a common real-world problem in fraud detection systems.

## Dataset
The project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). 
* **Total Transactions:** 284,807
* **Normal Transactions:** 284,315
* **Fraudulent Transactions:** 492
* **Class Imbalance:** Fraud accounts for only **0.173%** of all transactions.

## Pipeline Components
1. **Data Preprocessing:** Standardized `Time` and `Amount` features using `StandardScaler`. Applied stratified train-test splitting to maintain the exact fraud ratio in both sets.
2. **Feature Engineering:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class, perfectly balancing the training data (50/50).
3. **Model Training:** Trained three models:
   * Logistic Regression
   * Random Forest Classifier
   * XGBoost Classifier
4. **Evaluation:** Evaluated models using Precision, Recall, F1-Score, and AUC-ROC, plotting Confusion Matrices and ROC curves.

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1-Score (Fraud) | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.06 | 0.92 | 0.11 | 0.9698 |
| **Random Forest** | 0.82 | 0.82 | 0.82 | 0.9688 |
| **XGBoost (Best)** | **0.73** | **0.89** | **0.80** | **0.9792** |

## Key Insights
* **XGBoost** performed the best overall. It achieved the highest AUC-ROC (0.9792) and maintained an excellent balance between precision (0.73) and recall (0.89). Catching 89% of fraud cases while keeping false positives manageable makes it the most viable model for a production system.
* **Logistic Regression** suffered from an overwhelmingly high number of false positives (precision of 0.06), despite having a strong recall (0.92). 
* **Handling Class Imbalance is critical:** Without SMOTE, the models would likely have defaulted to predicting all transactions as "Normal," achieving 99.8% accuracy but failing to catch any fraud.

## How to Run

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the `creditcard.csv` dataset from Kaggle and place it in the same directory.
4. Run the Jupyter Notebook:
   ```bash
   jupyter notebook "Fraud Detection.ipynb"
   ```
