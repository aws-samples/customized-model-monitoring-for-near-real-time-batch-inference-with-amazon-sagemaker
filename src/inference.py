# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle as pkl

from sagemaker_containers.beta.framework import (
    encoders,
    env,
    server,
    transformer,
    worker,
)

def model_fn(model_dir):
    """Load a XGBoost Model
    Args:
        model_dir: a directory where model is saved.
    Returns: A XGBoost model.
    """
    
    with open(os.path.join(model_dir, "xgboost-model"), "rb") as f:
        booster = pkl.load(f)
    
    return booster


def input_fn(input_data, content_type):
    """Take request data and de-serializes the data into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:
            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.
    Returns:
        (obj): data ready for prediction. For XGBoost, this defaults to DMatrix.
    """
    
    if content_type == "application/json":
        request_json = json.loads(input_data)
        prediction_df = pd.DataFrame.from_dict(request_json)
        return xgb.DMatrix(prediction_df)
    else:
        raise ValueError


def predict_fn(input_data, model):
    """A predict_fn for XGBooost Framework. Calls a model on data deserialized in input_fn.
    Args:
        input_data: input data (DMatrix) for prediction deserialized by input_fn
        model: XGBoost model loaded in memory by model_fn
    Returns: a prediction
    """
    output = model.predict(input_data, validate_features=True)
    return output


def output_fn(prediction, accept):
    """Function responsible to serialize the prediction for the response.
    Args:
        prediction (obj): prediction returned by predict_fn .
        accept (str): accept content-type expected by the client.
    Returns: JSON output
    """
    
    if accept == "application/json":
        prediction_labels = np.argmax(prediction, axis=1)
        prediction_scores = np.max(prediction, axis=1)
        output_returns = [
            {
                "payload_index": int(index), 
                "label": int(label), 
                "score": float(score)} for label, score, index in zip(
                prediction_labels, prediction_scores, range(len(prediction_labels))
            )
        ]
        return worker.Response(encoders.encode(output_returns, accept), mimetype=accept)
    
    else:
        raise ValueError
