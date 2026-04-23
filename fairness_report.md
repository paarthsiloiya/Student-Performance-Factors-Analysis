# Fairness Report

We performed fairness checks across subgroups for Gender, Parental Education Level, and School Type, evaluating Accuracy, F1, ROC AUC, and Brier Score for our final LogisticRegression classifier (predicting Exam_Score >= 67).

## Subgroup Performance

1. **Gender**
   - **Male**: Accuracy: ~98.14% | F1: ~98.10% | ROC AUC: ~99.45% | Brier Score: 0.019
   - **Female**: Accuracy: ~98.46% | F1: ~98.44% | ROC AUC: ~99.64% | Brier Score: 0.017
   - **Disparity Analysis**: The model is highly accurate for both genders. Females exhibit slightly better metrics and lower Brier scores, suggesting better calibration and performance for this group. No severe bias is identified.

2. **Parental Education Level**
   - **High School**: Accuracy: ~98.18% | F1: ~98.18% | ROC AUC: ~99.43% | Brier Score: 0.018
   - **College**: Accuracy: ~98.49% | F1: ~98.46% | ROC AUC: ~99.62% | Brier Score: 0.018
   - **Postgraduate**: Accuracy: ~98.16% | F1: ~97.99% | ROC AUC: ~99.60% | Brier Score: 0.017
   - **Disparity Analysis**: The model performs robustly across all parental education levels. The small variance (F1 ranges between 98.0% - 98.4%) suggests model generalizability and well-calibrated confidence across socioeconomic descriptors. 

3. **School Type**
   - **Public**: Accuracy: ~98.15% | F1: ~98.12% | ROC AUC: ~99.51% | Brier Score: 0.019
   - **Private**: Accuracy: ~98.55% | F1: ~98.52% | ROC AUC: ~99.57% | Brier Score: 0.016
   - **Disparity Analysis**: The model works marginally better on students from Private schools, showing higher Accuracy, F1, and lower Brier score error.

## Mitigation Plan
While there are no glaring fairness violations (Accuracy > 98% for all checked groups), the slight discrepancies in calibration (Brier scores) and performance across Genders and School Types present minor learning biases.

**Proposed Actions**:
- **Reweighting Samples**: Apply class and subgroup sample weights during training so the model assigns equal penalty to errors across underrepresented demographic combinations (e.g., Male + Public School).
- **Adversarial Debiaser**: Use an adversarial network structure to unlearn the minor associations with demographic data.
- **Trade-offs**: Modifying the optimization surface to perfectly balance groups typically drops overall aggregate metrics (Accuracy/AUC), sacrificing a fraction of strong predictive power for marginal fairness edge. Given the minimal performance disparity right now, adjusting the decision threshold for slightly disparate groups might be a cleaner short-term fix.