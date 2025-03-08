from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np
app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[0,	84	,82	,31	,125,	38.2	,0.233	,23	]
    final=np.array([int_features])
    print(int_features)
    print(final)
    prediction=model.predict(final)
    output=prediction
    print(output)
  


 



if __name__ == '__main__':
    app.run(debug=True)