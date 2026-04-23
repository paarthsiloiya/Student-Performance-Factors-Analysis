import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Binarizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, brier_score_loss, classification_report
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
import json
import os

df = pd.read_csv('StudentPerformanceFactors_clean_v1.csv')
# Classification target: 1 if Exam_Score >= 67 else 0
df['Target'] = (df['Exam_Score'] >= 67).astype(int)
X = df.drop(columns=['Exam_Score', 'Target'])
y = df['Target']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])

models = {
    'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), {'classifier__C': [0.1, 1.0, 10.0]}),
    'GradientBoosting': (GradientBoostingClassifier(random_state=42), {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [3, 5]})
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
best_model_name = None
best_model_score = -1
best_model_pipe = None

os.makedirs('artifacts/model_evaluation', exist_ok=True)

for name, (model, param_grid) in models.items():
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)])
    search = GridSearchCV(pipe, param_grid, cv=skf, scoring='roc_auc', n_jobs=-1)
    search.fit(X, y)
    best_pipe = search.best_estimator_
    
    cv_res = search.cv_results_
    best_idx = search.best_index_
    
    probs = best_pipe.predict_proba(X)[:, 1]
    preds = best_pipe.predict(X)
    
    fpr, tpr, thresholds = roc_curve(y, probs) if 'roc_curve' in globals() else (None, None, None)
    
    importances = None
    if hasattr(best_pipe.named_steps['classifier'], 'feature_importances_'):
        importances = best_pipe.named_steps['classifier'].feature_importances_.tolist()
    elif hasattr(best_pipe.named_steps['classifier'], 'coef_'):
        importances = best_pipe.named_steps['classifier'].coef_[0].tolist()
        
    results[name] = {
        'best_params': search.best_params_,
        'cv_auc_mean': cv_res['mean_test_score'][best_idx],
        'cv_auc_std': cv_res['std_test_score'][best_idx],
        'accuracy': accuracy_score(y, preds),
        'f1': f1_score(y, preds, average='macro'),
        'roc_auc': roc_auc_score(y, probs),
        'brier_score': brier_score_loss(y, probs)
    }
    
    if cv_res['mean_test_score'][best_idx] > best_model_score:
        best_model_score = cv_res['mean_test_score'][best_idx]
        best_model_name = name
        best_model_pipe = best_pipe

with open('artifacts/metrics_summary.json', 'w') as f:
    json.dump(results, f, indent=4)

# Generate Calibration, CM for best model
probs = best_model_pipe.predict_proba(X)[:, 1]
preds = best_model_pipe.predict(X)
prob_true, prob_pred = calibration_curve(y, probs, n_bins=10)
plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('Calibration Plot - ' + best_model_name)
plt.savefig('artifacts/model_evaluation/calibration_plot.png')
plt.close()

cm = confusion_matrix(y, preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - ' + best_model_name)
plt.savefig('artifacts/model_evaluation/confusion_matrix.png')
plt.close()

# Feature Importance
perm_importance = permutation_importance(best_model_pipe, X, y, n_repeats=5, random_state=42)
sorted_idx = perm_importance.importances_mean.argsort()[-10:]
plt.figure()
plt.barh(range(10), perm_importance.importances_mean[sorted_idx])
plt.yticks(range(10), X.columns[sorted_idx])
plt.title('Permutation Feature Importance (Top 10)')
plt.savefig('artifacts/model_evaluation/feature_importance.png')
plt.close()

# Fairness
fairness = {}
for subgroup in ['Gender', 'Parental_Education_Level', 'School_Type']:
    fairness[subgroup] = {}
    for val in X[subgroup].unique():
        idx = X[subgroup] == val
        if idx.sum() > 0:
            sub_y = y[idx]
            sub_preds = preds[idx]
            sub_probs = probs[idx]
            fairness[subgroup][val] = {
                'accuracy': accuracy_score(sub_y, sub_preds),
                'f1': f1_score(sub_y, sub_preds, average='macro'),
                'roc_auc': roc_auc_score(sub_y, sub_probs),
                'brier_score': brier_score_loss(sub_y, sub_probs)
            }
with open('artifacts/fairness_summary.json', 'w') as f:
    json.dump(fairness, f, indent=4)

print("success")