# Diabetes Prediction with Flask & Machine Learning

## 📌 Project Overview

This project is a diabetes prediction system that uses Machine Learning (ML) algorithms to classify whether a patient is diabetic or not based on medical parameters. The trained model is deployed using Flask, allowing users to enter their medical data and receive predictions via a web interface.

## 🚀 Features

- **Data Preprocessing**: Handling missing values, scaling, and outlier removal.
- **Machine Learning Model Training**: Decision Tree Classifier with hyperparameter tuning.
- **Nested Cross-Validation**: For better accuracy estimation.
- **Model Deployment using Flask API**.
- **Interactive Web Interface**: For user input & prediction display.

## 🛠 Tech Stack

- **Programming Language**: Python 🐍
- **Machine Learning**: Scikit-learn
- **Web Framework**: Flask
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib

## 🎯 Usage

1. Open project in your browser.
2. Enter the required medical parameters in the form fields (such as age, BMI, blood pressure, etc.).
3. Click on the **"Predict"** button to submit the data.
4. The system will classify whether the patient is **Diabetic** or **Not Diabetic** based on the trained model.

### Example Form Input

| Field            | Description                                    |
|------------------|------------------------------------------------|
| Age              | Age of the patient                             |
| BMI              | Body Mass Index of the patient                 |
| Blood Pressure   | Blood pressure measurement (mm Hg)             |
| Glucose Level    | Glucose concentration in the blood (mg/dL)     |
| Insulin Level    | Insulin concentration (µU/mL)                  |
| etc.             | Other relevant medical data                   |

After entering the required information and clicking **"Predict"**, the web interface will show the prediction: **Diabetic** or **Not Diabetic**.
Example Screenshots from Website:
![image](https://github.com/user-attachments/assets/269e4e2c-4372-4670-b827-cc6381b4453e)
![image](https://github.com/user-attachments/assets/8e394c8f-06f1-47a1-881c-69f1fcd77212)

## ⚙️ Model Training

The **Decision Tree Classifier** is trained with the **Pima Indians Diabetes Dataset**, using the following steps:

### Data Cleaning & Preprocessing:
- Handling missing values using **SimpleImputer** (median strategy).
- Scaling using **RobustScaler**.
- Outlier detection & removal using **Interquartile Range (IQR)**.

### Model Selection & Hyperparameter Tuning:
- Used **GridSearchCV** to optimize model parameters.
- Nested **Stratified K-Fold Cross-Validation** for accuracy.

### Final Model Export:
- The best performing model is saved as `model.pkl` using **pickle**.

## 📈 Performance

- **Accuracy**: ~80-85%
- **Evaluation Metrics**:
  - Accuracy Score
  - Confusion Matrix
  - Cross-validation
  - ROC-AUC Score (optional)

## 🛠 Future Improvements

- Test with **Random Forest**, **SVM**, or **XGBoost** for better performance.
- Implement **REST API** for seamless integration.
- Add **user authentication** for personalized tracking.
- Improve **UI/UX** with better frontend design.

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## 📜 License

This project is MIT Licensed. Feel free to use and modify it.
