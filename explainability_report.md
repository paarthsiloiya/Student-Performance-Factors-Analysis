# Explainability Report

Our analysis evaluated the drivers behind student performance (classification: Exam_Score >= 67). We employed both parameter-based methods (Logistic Regression coefficients) and model-agnostic methods (Permutation Feature Importance) to glean insights.

## Global Feature Importance Summaries
Based on our best performing Logistic Regression model (C=10.0), here are the top driving features:

**1. Attendance**
- *Model-Based (Coefficient Magnitude)*: Highest positive coefficient.
- *Permutation Importance*: Largest drop in AUC when shuffled.
- *Interpretation*: Regular class attendance is the strongest predictor of achieving a passing/high score.
  
**2. Hours Studied**
- *Model-Based (Coefficient Magnitude)*: Second highest positive coefficient.
- *Permutation Importance*: High importance.
- *Interpretation*: Increased weekly study time consistently translates to a higher likelihood of passing.
  
**3. Previous Scores**
- *Model-Based (Coefficient Magnitude)*: Positive coefficient.
- *Permutation Importance*: Significant importance.
- *Interpretation*: Academic history positively correlates with current achievement.

*(Visual artifacts for permutation importance can be found in `artifacts/model_evaluation/feature_importance.png`)*

## Local Explanations (Counterfactuals)
We examined representative misclassified cases to understand the model's blind spots.

- **False Positive (Predicted Pass, Actual Fail):** A student with 98% attendance and 24 hours studied was predicted to score >= 67, but actually scored lower. 
  - *Explanation/Counterfactual*: The model heavily favors attendance and raw hours studied. However, qualitative factors like `Motivation_Level = Low` or unmeasured external stressors likely mitigated the effort. If the model had properly weighted interactions, or if the student's motivation were "High", the model's confidence would align with reality.

- **False Negative (Predicted Fail, Actual Pass):** A student missing several classes (60% Attendance) but possessing `Parental_Education_Level = Postgraduate` and `Access_to_Resources = High` actually scored > 70.
  - *Explanation/Counterfactual*: The model penalized the low attendance severely. Counterfactually, if the attendance had been 85%, the model would have predicted a pass. This indicates the model may under-reward strong external support networks when attendance is poor.
