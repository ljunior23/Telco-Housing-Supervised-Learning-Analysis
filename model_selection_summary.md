# Model Selection Summary Report

## Telco Customer Churn (`mini_project_telco.py`)

**Models Compared:**
- Logistic Regression
- K-Nearest Neighbors (KNN)

**Evaluation Metrics:**
- Classification report (precision, recall, f1-score, support)
- Confusion matrix
- Accuracy score

**How to Choose:**
- Compare the accuracy scores printed for both models.
- Review precision/recall/f1-score for the "Churn" class in the classification report.
- Logistic Regression also provides feature importance (coefficients).

**Recommendation:**
- Choose the model with the highest accuracy and best balance of precision/recall for the churn class.
- Logistic Regression is often preferred for interpretability and feature importance.

---

## California Housing (`mini_supervised_project.py`)

**Model Used:**
- Linear Regression

**Evaluation Metrics:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared Score (R²)

**How to Choose:**
- R² score indicates how much variance is explained by the model (closer to 1 is better).
- Lower MSE, RMSE, and MAE indicate better predictive performance.
- Feature coefficients show the impact of each feature.

**Recommendation:**
- Linear Regression is suitable if R² is high and errors are low.
---

## Summary Table

| Project                | Models Compared         | Best Metric(s)         | Recommended Model      |
|------------------------|------------------------|------------------------|-----------------------|
| Telco Churn            | Logistic Regression, KNN| Highest accuracy, f1   | Logistic Regression   |
| California Housing     | Linear Regression      | Highest R², lowest error| Linear Regression     |

---

Let me know if you want the actual metric values or a more detailed comparison!
