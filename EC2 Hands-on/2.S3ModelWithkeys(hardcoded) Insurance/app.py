from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import boto3
from sagemaker import Session
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

#app = Flask(__name__)
app = Flask(__name__, template_folder='templates')
# Load the pre-trained model
#filename = 'finalized_model_ckd.sav'
#model = pickle.load(open(filename, 'rb'))

@app.route('/')
def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form
        age_input = float(request.form['age'])
        bmi_input = float(request.form['bmi'])
        children_input = float(request.form['children'])
        sex_male_input= float(request.form['sex_male'])
        smoker_yes_input= float(request.form['smoker_yes'])

        # Format the input data into a numpy array
        #input_data = np.array([[age, blood_pressure, specific_gravity, albumin, sugar]])

        # Make prediction using the loaded model
        #prediction = model.predict(input_data)
        # ----------- YOUR AWS CREDENTIALS ----------
        aws_access_key_id = ""
        aws_secret_access_key = "" 
        region_name = "ap-south-1"
        endpoint_name = "insurance-charge-endpoint4"
        # -------------------------------------------

        # Create a boto3 session using credentials
        boto_session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        # Create a SageMaker session from the boto3 session
        sagemaker_session = Session(boto_session=boto_session)

        # Create Predictor object
        predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session
        )
        #print(prediction)
        result = predict_insurance_charge(
            predictor,
            age=age_input,
            bmi=bmi_input,
            children=children_input,
            sex_male=sex_male_input,
            smoker_yes=smoker_yes_input
        )
        # Map prediction to result
        #result = 'Insurance' if prediction[0] == 1 else 'No Chronic Kidney Disease'

        #result=prediction
        print(result)

        return render_template('output.html', result=result)
    except Exception as e:
        return jsonify({'error': str(e)})
    
def predict_insurance_charge(predictor, age, bmi, children, sex_male, smoker_yes):
    """
    Sends a prediction request to the SageMaker endpoint.
    Returns: Predicted insurance charge
    """
    predictor.serializer = JSONSerializer()
    predictor.deserializer = JSONDeserializer()

    input_data = {
        "age": [age],
        "bmi": [bmi],
        "children": [children],
        "sex_male": [sex_male],
        "smoker_yes": [smoker_yes]
    }

    prediction = predictor.predict(input_data)
    return prediction

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
