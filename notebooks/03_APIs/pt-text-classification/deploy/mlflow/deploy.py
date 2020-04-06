import sys
sys.path.append(".")
import os
import mlflow
from mlflow import pyfunc as ml_pyfunc

from text_classification import config
from text_classification import predict
import torch
from deploy_env import DeployEnv

# Sagemaker
from mlflow import sagemaker as mfs

# Azure
import mlflow.azureml
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice

BUILD_DIR = "build/model"

env = DeployEnv()

class TextClassificationModel(mlflow.pyfunc.PythonModel):
    def __init__(self):
        pass

    def predict(self,context,input):
        model_input = [{"text": text} for text in input['text'].values]
        prediction = predict.predict(
            experiment_id='latest', inputs=model_input)
        return prediction

def build_azure_image(ws):
    # Build an Azure ML container image for deployment
    azure_image, azure_model = mlflow.azureml.build_image(model_uri="build/mlflow",
                                                          workspace=ws,
                                                          description=env.setting("model_name"),
                                                          synchronous=True)
    return azure_image

def deploy_azure(ws, azure_image):
    ws = Workspace.from_config(env.setting("workspace_config_path"))
    azure_image = build_azure_image(ws)

    webservice_deployment_config = AciWebservice.deploy_configuration()
    webservice = Webservice.deploy_from_image(
                    image=azure_image, workspace=ws, name=env.setting("model_name"))
    webservice.wait_for_deployment()
    print("Scoring URI is: %s", webservice.scoring_uri)

    return webservice

def deploy_sagemaker():
    # Deploying to Sagemaker results in the following error:
    # ERROR: Could not install packages due to an EnvironmentError: [Errno 28] No space left on device
    return mfs.deploy(app_name=env.setting("model_name")+"-mlflow",
           model_uri="build/mlflow",
           region_name=env.setting("region"),
           mode="create", # this should change to replace if the endpoint already exists
           execution_role_arn=env.setting("aws_role"),
           image_url=env.setting("image_url"),
           instance_type=env.setting("instance_type"))

def build_model_data_file():
    return os.system("tar -czf build/model.tar.gz experiments text_classification logging.json")

def extract_model_data_file():
    os.system("rm -Rf %s" % (BUILD_DIR))
    os.system("mkdir %s" % (BUILD_DIR))
    return os.system("tar -xzf build/model.tar.gz -C %s" % (BUILD_DIR))

def deploy():
    mlflow_conda_env = {
     'name': 'mlflow-env',
     'channels': ['defaults'],
     'dependencies': ['python=3.7.5', {'pip': ['sklearn==0.0','mlflow==1.7.2','torch==1.4.0']}]
    }

    build_model_data_file()
    extract_model_data_file()

    os.system("rm -Rf build/mlflow")

    # Need the inner contents of the `model` folder. Specifying just BUILD_DIR grabs the parent directory.
    code_path = [ BUILD_DIR+"/"+name  for name in os.listdir(BUILD_DIR)]
    print("adding code path=",code_path)
    ml_pyfunc.save_model("build/mlflow",
    code_path=code_path,
    conda_env=mlflow_conda_env,
    python_model=TextClassificationModel())

    if env.isLocal():
        os.system("mlflow models serve -m build/mlflow/")
    else:
        webservice = deploy_azure(ws, azure_image)

if __name__ == '__main__':
    deploy()
