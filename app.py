from flask import Flask,request,render_template
import pandas as pd
import joblib

app = Flask(__name__)

model_diabetes = joblib.load('trained_model.pkl')
model_attrition = joblib.load('attrition_model.pkl')

@app.route('/',methods=['GET','POST'])
def diabetes_prediction():
    if request.method == 'POST':
        data = {
            'Glucose': float(request.form['glucose']),
            'BloodPressure': float(request.form['bloodpressure']),
            'Insulin': float(request.form['insulin']),
            'BMI': float(request.form['bmi']),
            'DiabetesPedigreeFunction': float(request.form['dpf']),
            'Age': float(request.form['age'])
        }
        
        df = pd.DataFrame([data])
        result = model_diabetes.predict(df)
        return render_template('diabetes.html',result="😱 you have diabetic,\nPlease concern 👨🏻‍⚕️" if result[0]==1 else "🤗 you are non-diabetic,\nHealth is wealth💪🏻")

    return render_template('diabetes.html')

@app.route('/attrition', methods=['GET', 'POST'])
def attrition_prediction():
    if request.method == 'POST':
        data = {
            'BusinessTravel': request.form['BusinessTravel'],
            'Department': request.form['Department'],
            'DailyRate': float(request.form['DailyRate']),
            'DistanceFromHome': float(request.form['DistanceFromHome']),
            'Education': request.form['Education'],
            'EducationField': request.form['EducationField'],
            'EnvironmentSatisfaction': request.form['EnvironmentSatisfaction'],
            'Gender': request.form['gender'],
            'JobRole': request.form['job_role'],
            'MaritalStatus': request.form['MaritalStatus'],
            'OverTime': request.form['OverTime'],
            'JobLevel': request.form['job_level'],
            'JobSatisfaction': request.form['jobSatisfaction'],
            'WorkLifeBalance': request.form['WorkLifeBalance'],
            'RelationshipSatisfaction': request.form['RelationshipSatisfaction'],
            'PerformanceRating': request.form['performance_rating'],
            'StockOptionLevel': request.form['stock_option'],
            'JobInvolvement': request.form['job_involvement'],
            'MonthlyIncome': int(request.form['monthly_income']),
            'Age': int(request.form['age']),
            'TotalWorkingYears': int(request.form['total_working_years']),
            'YearsAtCompany': int(request.form['years_at_company']),
            'YearsSinceLastPromotion': int(request.form['years_since_last_promotion']),
            'YearsInCurrentRole': int(request.form['years_in_current_role']),
            'YearsWithCurrManager': int(request.form['years_with_current_manager']),
            'HourlyRate': int(request.form['hourly_rate']),
            'MonthlyRate': int(request.form['monthly_rate']),
            'PercentSalaryHike': int(request.form['percent_salary_hike']),
            'TrainingTimesLastYear': int(request.form['training_times_last_year']),
            'NumCompaniesWorked': int(request.form['number_companies_worked'])
        }

        df = pd.DataFrame([data])
        result = model_attrition.predict(df)
        return render_template('attrition.html', result="High Risk of Attrition, take action!" if result[0] == 1 else "Low Risk of Attrition, our Employee 😎")

    return render_template('attrition.html')

if __name__=='__main__':
    app.run(debug=True)