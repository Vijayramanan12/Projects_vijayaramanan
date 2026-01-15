
templates/              # HTML templates

# Machine Learning Flask App

> Predict employee attrition and diabetes risk using a Flask web interface and pre-trained ML models.

---

## 🚀 Features
- Employee attrition prediction (IBM dataset)
- Diabetes risk prediction
- User-friendly Flask web interface
- Pre-trained models: `attrition_model.pkl`, `trained_model.pkl`

## 🛠 Technologies
- Python 3.11
- Flask
- scikit-learn
- pandas, numpy, matplotlib

## 📁 Project Structure
```
app.py                  # Main Flask app
IBM_attrition.py        # IBM attrition model logic
attrition_model.pkl     # Pre-trained attrition model
trained_model.pkl       # Pre-trained diabetes model
diabetic.py             # Diabetes model logic
L_machine_L/            # Python virtual environment (excluded from Git)
  attributes.py         # Additional attributes logic
templates/              # HTML templates
requirements.txt        # Python dependencies
```

## ⚡️ Quickstart
1. **Clone the repository:**
	```sh
	git clone https://github.com/Vijayramanan12/Projects_vijayaramanan.git
	cd Projects_vijayaramanan
	```
2. **Create and activate a virtual environment:**
	```sh
	python3 -m venv L_machine_L
	source L_machine_L/bin/activate
	```
3. **Install dependencies:**
	```sh
	pip install -r requirements.txt
	```
4. **Run the Flask app:**
	```sh
	python app.py
	```
	The app will be available at [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 💡 Usage
- Open your browser and go to `http://127.0.0.1:5000`
- Use the web interface to make predictions for attrition or diabetes

## 🤝 Contributing
Pull requests and issues are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
This project is open source and available under the MIT License.

## 📬 Contact
Maintained by [Vijay Ramanan](https://github.com/Vijayramanan12)

---
Feel free to open issues or submit pull requests!
