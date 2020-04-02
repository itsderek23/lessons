import torch
import os
import logging
import sys
import json
from text_classification import predict
from sagemaker_inference import decoder, content_types, encoder

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger("model")

def model_fn(model_dir):
    log.info("In model_fn(). DEPLOY_ENV=",os.environ.get("DEPLOY_ENV"))

# From docs:
# Default json deserialization requires request_body contain a single json list.
# https://github.com/aws/sagemaker-pytorch-serving-container/blob/master/src/sagemaker_pytorch_serving_container/default_inference_handler.py#L49
def input_fn(request_body, request_content_type):
    data = json.loads(request_body)
    model_input = [{"text": features[0]} for features in data]
    return model_input

def predict_fn(input_data, model):
    prediction = predict.predict(
        experiment_id='latest', inputs=input_data)
    return prediction

def output_fn(prediction, content_type):
    res = [{"probabilities": result["probabilities"], "top_n_grams": result["top_n_grams"]} for result in prediction]
    return encoder.encode(res, content_type)
