
# Machine Learning Flask App

This project is a Flask-based web application for predicting employee attrition and diabetes using pre-trained machine learning models. It includes a custom virtual environment and all required dependencies.

## Features
- Employee attrition prediction (IBM dataset)
- Diabetes prediction
- Flask web interface with HTML templates
- Pre-trained models (`attrition_model.pkl`, `trained_model.pkl`)

## Technologies Used
- Python 3.11
- Flask
- scikit-learn
- pandas, numpy, matplotlib

## Project Structure
```
app.py                  # Main Flask app
IBM_attrition.py        # IBM attrition model logic
attrition_model.pkl     # Pre-trained attrition model
trained_model.pkl       # Pre-trained diabetes model
diabetic.py             # Diabetes model logic
L_machine_L/            # Python virtual environment
  attributes.py         # Additional attributes logic
  ...
templates/              # HTML templates
```

## Setup Instructions
1. **Clone the repository:**
	```sh
	git clone https://github.com/Vijayramanan12/Projects_vijayaramanan.git
	cd Projects_vijayaramanan
	```
2. **Activate the virtual environment:**
	```sh
	source L_machine_L/bin/activate
	```
	If you need to recreate the environment:
	```sh
	python3 -m venv L_machine_L
	source L_machine_L/bin/activate
	pip install -r requirements.txt  # If requirements.txt is available
	```
3. **Run the Flask app:**
	```sh
	python app.py
	```
	The app will be available at http://127.0.0.1:5000

## Usage
- Open your browser and go to `http://127.0.0.1:5000`
- Use the web interface to make predictions for attrition or diabetes

## Contact
Maintained by [Vijay Ramanan](https://github.com/Vijayramanan12)

---
Feel free to open issues or submit pull requests!
