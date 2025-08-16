📊 HR Analytics – Employee Promotion Prediction
📝 Executive Summary

Employee promotion is one of the most crucial aspects of talent management. This project leverages Machine Learning (ML) techniques to build a predictive system that helps HR departments identify deserving employees for promotion.

By analyzing historical employee data and performance metrics, this solution demonstrates how data-driven insights can enhance fairness, efficiency, and transparency in promotion decisions.

🎯 Business Problem

Promotions directly impact employee motivation, retention, and organizational growth.

Manual decision-making is prone to biases and inefficiencies.

Objective: Predict whether an employee should be promoted using data-driven models.

📂 Repository Structure
HR-Analytics/
│── data/employee_promotion.csv     # Dataset
│── src/
│   ├── data_preprocessing.py       # Handling missing values, encoding, scaling
│   ├── exploratory_analysis.py     # EDA with plots
│   ├── feature_selection.py        # RFE & correlation-based selection
│   ├── model_training.py           # Logistic Regression, Random Forest, XGBoost
│   ├── evaluation.py               # Model comparison, ROC curves
│── figures/                        # Visualizations saved here
│── main.py                         # End-to-end pipeline
│── requirements.txt                # Dependencies
│── README.md                       # Project documentation

📊 Dataset Description
Feature	Description
employee_id	Unique identifier
department	Department of employee
education	Highest qualification
length_of_service	Years in organization
previous_year_rating	Past performance rating
avg_training_score	Training exam score
KPIs_met	KPI achievement (Yes/No)
awards_won	Award recognition (Yes/No)
is_promoted	Target variable (1 = promoted, 0 = not promoted)
🔎 Methodology
1️⃣ Data Preprocessing

Missing Values:

previous_year_rating → filled with 0 for new employees

education → imputed with most frequent category

Categorical Encoding: Label Encoding for department, education, etc.

Feature Scaling: Standardization applied to numeric features.

2️⃣ Exploratory Data Analysis (EDA)

Department-wise promotion distribution

Age demographics and service years

Relationship between training scores & promotions

Correlation heatmap to identify key drivers

3️⃣ Feature Selection

Correlation Analysis

Recursive Feature Elimination (RFE)

Final selected features:

KPIs_met, awards_won, avg_training_score, previous_year_rating, length_of_service

4️⃣ Model Training

Logistic Regression (baseline)

Random Forest (ensemble)

XGBoost (gradient boosting, state-of-the-art)

5️⃣ Evaluation Metrics

Accuracy, Precision, Recall, F1-Score

ROC-AUC Curves

📈 Key Insights from EDA

Departmental Trends:
Sales & Marketing had the largest workforce but fewer promotions proportionally.

Age Distribution:
Majority of employees are in the 25–35 age group.

Training vs Service:
Employees with <10 years of service had more consistent high training scores.

Correlation Analysis:
Promotions strongly correlated with training scores, KPIs met, and awards won.

🤖 Model Performance
Model	Accuracy	AUC Score
Logistic Regression	88%	0.87
Random Forest	93%	0.87
XGBoost	94%	0.91 ✅

📌 XGBoost selected as final model due to superior performance.

📊 Visual Results
🔹 ROC-AUC Curve

🔹 Feature Importance (XGBoost)

🔹 Age Distribution

✅ Conclusion & Future Work
Key Takeaways

High training scores, KPI achievement, and awards are strong promotion indicators.

XGBoost provided the most reliable predictions (AUC 0.91).

Future Enhancements

Hyperparameter tuning with GridSearchCV

Deploying as an interactive HR dashboard (Flask / Streamlit)

Integrating with live HR systems for real-time analytics
HOW TO RUN
# Clone the repo
git clone https://github.com/Karishma2324/HR-Analytics.git
cd HR-Analytics

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py
👩‍💻 Author

Mohammad Karishma

linkedin.com/in/karishma-mohammad2324

📧 karishmamohammad506@gmail.com
