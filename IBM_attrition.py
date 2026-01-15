import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

df = pd.read_csv('/Users/vijayramanan/Downloads/archive/IBM HR Employee Attrition Data.csv')
#print(df.head())
df.drop(columns=['EmployeeCount','EmployeeNumber','Over18','StandardHours'],inplace=True)
#print(df.info())

categorical_cols = ["BusinessTravel","Department","EducationField","Gender","JobRole","MaritalStatus","OverTime"]
ordinal_cols = ["Education","JobLevel","JobSatisfaction","WorkLifeBalance","RelationshipSatisfaction","PerformanceRating","StockOptionLevel","EnvironmentSatisfaction","JobInvolvement"]
numerical_cols = ["Age","DailyRate","DistanceFromHome","MonthlyIncome","TotalWorkingYears","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager","HourlyRate","MonthlyRate","PercentSalaryHike","TrainingTimesLastYear","NumCompaniesWorked"]

df[ordinal_cols] = df[ordinal_cols].apply(pd.to_numeric)

print("Class distribution:",Counter(df["Attrition"]))

numerical_transformer = Pipeline(steps=[("imputer",SimpleImputer(strategy='median')),("scaling",StandardScaler())])
categorical_transformer = Pipeline(steps=[("imputer",SimpleImputer(strategy='most_frequent')),("encoding",OneHotEncoder(handle_unknown="ignore",sparse_output=False))])

# i tested ordinal encoder without parameter but it throws ValueError when value get from frontend so use handle_unknown="use_encoded_value",unknown_value=-1 to avoid this issue
ordinal_transformer = Pipeline(steps=[("imputer",SimpleImputer(strategy='most_frequent')),("ordinal",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))])

preprocessing = ColumnTransformer(transformers=[("numeric",numerical_transformer,numerical_cols),("categorical",categorical_transformer,categorical_cols),("ordinal",ordinal_transformer,ordinal_cols)],remainder="drop")

param = [{
    "model": [LogisticRegression(max_iter=1000,solver='lbfgs',class_weight='balanced')],
    "model__C": [0.1, 1.0, 10.0]
},
{
    "model": [RandomForestClassifier(class_weight='balanced')],
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20]
},
{
    "model": [SVC(class_weight='balanced')],
    "model__C": [0.1, 1.0, 10.0],
    "model__kernel": ["linear", "rbf"]
},
{
    "model":[XGBClassifier(scale_pos_weight=1)],
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20]
}
]

model = Pipeline(steps=[("preprocessor",preprocessing),("model",LogisticRegression())])

multiple_models = RandomizedSearchCV(estimator=model,param_distributions=param,n_iter=10,cv=5,scoring='roc_auc',random_state=42,n_jobs=-1)

X = df.drop(columns=['Attrition'])
y = df["Attrition"].map({'Yes':1,'No':0})

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

multiple_models.fit(X_train,y_train)

# run this to see the best model
print(multiple_models.best_estimator_.named_steps['model'])

# print(multiple_models.best_params_)
# train_result = pd.DataFrame(multiple_models.cv_results_)
# print(train_result[['mean_test_score','std_test_score','params','rank_test_score']].sort_values(by='rank_test_score').head())

# i already tested the ROC AUC score if you want you can check it

# y_proba = multiple_models.predict_proba(X_test)[:, 1]
# test_roc_auc = roc_auc_score(y_test, y_proba)
# print(test_roc_auc)

# joblib.dump(multiple_models.best_estimator_, "attrition_model.pkl")
# best_model = multiple_models.best_estimator_

# test_result = pd.DataFrame(best_model.predict(X_test))
# print(accuracy_score(y_test,test_result))