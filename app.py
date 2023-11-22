from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('dtclassifier.pickle', "rb"))
scaler = pickle.load(open('scaler.pickle', "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    pre_final_features = [np.array(int_features)]
    final_features = scaler.transform(pre_final_features)
    prediction = model.predict(final_features)
    if(prediction[0] == 1):
        output = 'buy'
    elif(prediction[0] == 0):
        output = 'not buy'
    else:
        output = 'not Sure'
    return render_template('index.html', prediction_text = 'This user will {} from social network ad'.format(output))

if __name__ == '__main__':
    app.run(debug=True)