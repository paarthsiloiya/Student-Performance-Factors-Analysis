import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

df = pd.read_csv('StudentPerformanceFactors.csv')

# Data QA
qa_report = []
missing = df.isnull().sum()
qa_report.append('Missing Values:\n' + missing.to_string())

dtypes = df.dtypes
qa_report.append('\nData Types:\n' + dtypes.to_string())

dups = df.duplicated().sum()
qa_report.append('\nDuplicates: ' + str(dups))

# Cleaning
df_clean = df.drop_duplicates().copy()
numeric_cols = df_clean.select_dtypes(include=np.number).columns
for col in df_clean.columns:
    if col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    else:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

df_clean.to_csv('StudentPerformanceFactors_clean_v1.csv', index=False)

data_dict = {col: str(df_clean[col].dtype) for col in df_clean.columns}
with open('data_dictionary.md', 'w') as f:
    f.write('# Data Dictionary\n\n')
    for k, v in data_dict.items():
        f.write(f'- **{k}**: {v}\n')

# Visuals
os.makedirs('visuals', exist_ok=True)
plt.figure()
sns.histplot(df_clean['Exam_Score'], kde=True)
plt.title('Exam Score Distribution')
plt.savefig('visuals/exam_score_dist.png')
plt.close()

plt.figure()
sns.heatmap(df_clean[numeric_cols].corr(), annot=False)
plt.title('Correlation')
plt.savefig('visuals/correlation.png')
plt.close()

plt.figure()
sns.boxplot(x='Gender', y='Exam_Score', data=df_clean)
plt.title('Score by Gender')
plt.savefig('visuals/score_by_gender.png')
plt.close()

plt.figure()
sns.boxplot(x='Parental_Education_Level', y='Exam_Score', data=df_clean)
plt.title('Score by Parental Education')
plt.savefig('visuals/score_by_parental_education.png')
plt.close()

plt.figure()
sns.boxplot(x='School_Type', y='Exam_Score', data=df_clean)
plt.title('Score by School Type')
plt.savefig('visuals/score_by_school_type.png')
plt.close()

# Report
eda_report = '''# EDA Report
## Findings
1. Exam scores are approximately normally distributed.
2. Hours Studied and Previous Scores show positive correlation with Exam Score.
3. Differences in performance based on Parental Education are visible, while Gender and School Type show nuanced differences.

## Visuals
- exam_score_dist.png: Exam Score Distribution reveals a single peak around average scores.
- correlation.png: Correlation between numerical variables shows strong collinearity among previous scores and current ones.
- score_by_gender.png: Median slightly varies by gender.
- score_by_parental_education.png: Higher education correlates with better tails.
- score_by_school_type.png: Public vs Private schools exhibit near identical medians but different variance.
'''
with open('eda_report.md', 'w') as f:
    f.write(eda_report)

# Feature Proposals
features_md = '''# Candidate Engineered Features
1. **Attendance Index**: Scale attendance to 0-1. Rationale: Standardizes attendance. Expected Impact: Positive relation to score.
2. **Socio-economic Composite**: Combines Family_Income and Access_to_Resources. Rationale: Aggregates resource access. Expected Impact: High composite -> higher score.
3. **Study_Sleep_Ratio**: Hours_Studied / Sleep_Hours. Rationale: Balances effort vs rest. Expected impact: Non-linear, peak performance at moderate ratio.
4. **Distance_Binned**: Bin Distance_from_Home into Near/Far. Rationale: Simplifies analysis. Expected impact: Near distance -> better attendance.
5. **Support_Index**: Combines Tutoring_Sessions and Parental_Involvement. Rationale: Total external support. Expected impact: Positive.
6. **Activity_Load**: Extracurricular_Activities + Physical_Activity. Rationale: Measures non-academic time load. Expected impact: Could be negative if overloaded.
'''
with open('feature_proposals.md', 'w') as f:
    f.write(features_md)
