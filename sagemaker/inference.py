# inference.py
import os
import joblib
import pandas as pd
import json
import numpy as np
import io

# 1. Load trained model
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    return joblib.load(model_path)

# 2. Parse input request
def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data_dict = json.loads(request_body)
        return pd.DataFrame(data_dict)
    
    elif content_type == 'application/x-npy':
        array = np.load(io.BytesIO(request_body))
        return pd.DataFrame(array)

    raise ValueError(f"Unsupported content type: {content_type}")


# 3. Predict
def predict_fn(input_data, model):
    return model.predict(input_data)

# 4. Format output
def output_fn(prediction, content_type):
    return json.dumps(prediction.tolist())
