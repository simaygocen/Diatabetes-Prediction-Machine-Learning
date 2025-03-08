
# 1. Initialization

# import all necessary libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# algorithms
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree


# nested cross validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay


from statistics import mean, stdev

# for evaluation metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

from numpy import mean
from numpy import std

from sklearn.impute import SimpleImputer


"""# 2. Data Preprocessing

## 2.1 Exploratory Data Analysis
"""

diabetes_data = pd.read_csv("./diabetesDataset.csv")
column_names = diabetes_data.columns.to_numpy()
attributes = column_names[0:8]


"""## 2.2 Replace zero values with NaN"""

diabetes_data.isnull().sum()

# replace 0 for columns where zero value does not make sense with NaN
diabetes_data_nan = diabetes_data
diabetes_data_nan[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = diabetes_data_nan[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)

diabetes_data_nan.isnull().sum()

percentages = []
for i in diabetes_data_nan.isnull().sum().values:
    percentages.append((i / 768) * 100)
for i in range(len(percentages)):
    print(column_names[i], ": \t", round(percentages[i], 2), " %")

"""## 2.3 Outlier Handling"""


# count outliers
Q1 = diabetes_data_nan.quantile(0.25)
Q3 = diabetes_data_nan.quantile(0.75)
IQR = Q3 - Q1

((diabetes_data_nan < (Q1 - 1.5 * IQR)) | (diabetes_data_nan > (Q3 + 1.5 * IQR))).sum()

# remove outliers
diabetes_data_new = diabetes_data_nan[~((diabetes_data_nan < (Q1 - 1.5 * IQR)) | (diabetes_data_nan > (Q3 + 1.5 * IQR))).any(axis=1)]
diabetes_data_new.shape



"""## 2.4 Missing Values Handling"""

# mean values of Outcome 1 and Outcome 0
print("Mean values for Outcome = 1")
print(diabetes_data_new.groupby('Outcome').mean().loc[1])
print("\n")
print("Mean values for Outcome = 0")
print(diabetes_data_new.groupby('Outcome').mean().loc[0])

# replace the null values by mean

imputer = SimpleImputer(missing_values=np.NaN, strategy='median')
diabetes_data_filter = pd.DataFrame(imputer.fit_transform(diabetes_data_new), columns = column_names)

diabetes_data_filter.isnull().sum()



"""## 2.5 Scaling Data"""

# scaling the dataset
X = pd.DataFrame(diabetes_data_filter).to_numpy()[:, 0:8]
y = pd.DataFrame(diabetes_data_filter).to_numpy()[:, 8]
scaler=RobustScaler()
X_scaled = scaler.fit_transform(X, y)
X = pd.DataFrame(X_scaled).to_numpy()[:, 0:8]

"""# Algorithms"""

random_state = 2023
splits = 10

# Import the necessary libraries


# Define the data (assuming X and y are defined elsewhere)
# X = ...
# y = ...

# Define the algorithm and its parameters
algorithm = tree.DecisionTreeClassifier(random_state=random_state)
params = {
    'criterion': ['gini'],
    'max_depth': [None],
    'splitter': ['random'],
    'min_samples_split': [2]
}

# Perform nested cross-validation for hyperparameter optimization
cv_outer = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
accuracy = []

for train, test in cv_outer.split(X, y):
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

    model = algorithm
    gs = GridSearchCV(model,
                    param_grid=params,
                    scoring='accuracy',
                    cv=3,  # Using 3-fold cross-validation for inner loop
                    refit=True)
    result = gs.fit(X_train, y_train)
    best_model = gs.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy.append(acc)

   
print('Accuracy=%.3f, Params=%s' % (acc, result.best_params_))


best_model.predict(X_test)
pickle.dump(best_model, open('model.pkl', 'wb'))


from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
app= Flask(__name__)

model=pickle.load(open('model.pkl','rb'))



@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    #int_features=[0, 300.499654	, 309.900000, 1396.994807, 45.901385,210]

    form_values = list(request.form.values())
    int_features=[float(form_values[0]),float(form_values[1]),float(form_values[2]),float(form_values[3]),float(form_values[4]),float(form_values[5]),float(form_values[6]),int(form_values[7])]
    final=np.array([int_features])

    final_scaled = scaler.transform(final)
    print(final_scaled)

    
    prediction=model.predict(final_scaled)
    print(prediction)
    output=prediction
    
    if output[0]==1:
        pred_text="Diabetic"
    else:
        pred_text="Not Diabetic"

    print(output[0])

    return render_template('index.html', pred='<h4 align="center" style="color:red;"><b>{}</b></h4>'.format(pred_text), probability=int(output[0]))


app.run(debug=True)
