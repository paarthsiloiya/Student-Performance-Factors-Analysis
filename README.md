# Student Performance Factors Analysis

An end-to-end data science project investigating the socio-economic, environmental, and academic variables that influence student outcomes. This repository features exploratory data analysis (EDA), statistical modeling, and machine learning techniques to identify key predictors of academic success.

## About Dataset
### Description
This dataset provides a comprehensive overview of various factors affecting student performance in exams. It includes information on study habits, attendance, parental involvement, and other aspects influencing academic success.

### Column Descriptions

| Attribute | Description |
| :--- | :--- |
| **Hours_Studied** | Number of hours spent studying per week. |
| **Attendance** | Percentage of classes attended. |
| **Parental_Involvement** | Level of parental involvement in the student's education (Low, Medium, High). |
| **Access_to_Resources** | Availability of educational resources (Low, Medium, High). |
| **Extracurricular_Activities** | Participation in extracurricular activities (Yes, No). |
| **Sleep_Hours** | Average number of hours of sleep per night. |
| **Previous_Scores** | Scores from previous exams. |
| **Motivation_Level** | Student's level of motivation (Low, Medium, High). |
| **Internet_Access** | Availability of internet access (Yes, No). |
| **Tutoring_Sessions** | Number of tutoring sessions attended per month. |
| **Family_Income** | Family income level (Low, Medium, High). |
| **Teacher_Quality** | Quality of the teachers (Low, Medium, High). |
| **School_Type** | Type of school attended (Public, Private). |
| **Peer_Influence** | Influence of peers on academic performance (Positive, Neutral, Negative). |
| **Physical_Activity** | Average number of hours of physical activity per week. |
| **Learning_Disabilities** | Presence of learning disabilities (Yes, No). |
| **Parental_Education_Level** | Highest education level of parents (High School, College, Postgraduate). |
| **Distance_from_Home** | Distance from home to school (Near, Moderate, Far). |
| **Gender** | Gender of the student (Male, Female). |
| **Exam_Score** | Final exam score. |

### Link
[https://www.kaggle.com/datasets/lainguyn123/student-performance-factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)

## Project Overview
This project analyzes various factors affecting student performance, including:
- Study habits (Hours Studied, Attendance)
- Parental involvement and education
- Access to resources
- Extracurricular activities

## Key Analysis Steps
1. **Data Cleaning**: Handling missing values and duplicates.
2. **EDA**: Univariate, Bivariate, and Multivariate analysis using Violin plots, Histograms, and Pairplots.
3. **Correlation Analysis**: Identifying key drivers of exam scores.
4. **Machine Learning**: Training a Random Forest Regressor to predict exam scores.
5. **Model Evaluation**: Assessing performance using R², RMSE, and MAE.

## Model Performance
The Random Forest Regressor achieved the following results:
- **R² Score**: 0.62 (Explains 62% of the variance in exam scores)
- **MAE**: 1.18
- **RMSE**: 2.43

## Key Insights
- **Attendance** and **Hours Studied** are the strongest predictors of academic success.
- **Previous Scores** also show a significant positive correlation.
- Socio-economic factors like **Parental Education** and **Distance from Home** showed minimal direct impact in this model compared to student engagement metrics.

## Technologies Used
- Python
- Pandas
- Seaborn & Matplotlib
- Scikit-learn
