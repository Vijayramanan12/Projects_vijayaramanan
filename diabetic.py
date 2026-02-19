import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn import svm
import joblib

data = pd.read_csv('/Users/vijayramanan/VS code/diabetes.csv')
df = pd.DataFrame(data)
# print(df)

# replace 0 with NaN
df = df.replace({'Glucose': 0,
                 'Insulin' :0,
                 'BloodPressure':0,
                 'SkinThickness': 0,
                 'BMI': 0}, np.nan)

cleaned_df = pd.DataFrame(df,columns=df.columns)
X = cleaned_df[['Glucose','BloodPressure','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = cleaned_df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# data manipulation
mean_col = ['Glucose','BloodPressure','BMI']
median_col = ['Insulin']
other_col = ['DiabetesPedigreeFunction','Age']

mean_pipe = Pipeline([('imputer',SimpleImputer(strategy='mean')),('scaler',StandardScaler())])
median_pipe = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
other_pipe = Pipeline([('scaler',StandardScaler())])

preprocessor = ColumnTransformer([('mean',mean_pipe,mean_col),('median',median_pipe,median_col),('other',other_pipe,other_col)])

lr_model = Pipeline([('preprocessor',preprocessor),('classifier',LogisticRegression(max_iter=1000))])

rf_model = Pipeline([('preprocessor',preprocessor),('classifier',svm.SVC())])

lr_model.fit(X_train,y_train)
rf_model.fit(X_train,y_train)

lr_predicted = lr_model.predict(X_test)
rf_predicted = rf_model.predict(X_test)

#joblib.dump(lr_model, 'trained_model.pkl')
accuracy1 = accuracy_score(y_test,lr_predicted)
accuracy2 = accuracy_score(y_test,rf_predicted)

lr_score = cross_val_score(lr_model, X,y, cv=5)
lr_mean = lr_score.mean()*100
rf_score = cross_val_score(rf_model,X,y, cv=5)
rf_mean = rf_score.mean()*100
print(f"logistic Accuracy in %: {accuracy1*100:.2f}")
print(f"random forest Accuracy in %: {accuracy2*100:.2f}")

print(f"CV accuracy of logistic model: {lr_mean:.2f}")
print(f"CV accuracy of random forest: {rf_mean:.2f}")


# user input
# user_value = []
# for col in X.columns:
#     user_value.append(float(input(f"Enter {col} value: ")))
    
# user_df = pd.DataFrame([user_value],columns=X.columns)
# user_result = lr_model.predict(user_df)
# if int(user_result):
#     print("You have diabetic")
# else:
#     print("You are Non-Diabetic")