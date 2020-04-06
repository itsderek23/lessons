import yaml
import os
import sagemaker
import boto3
import botocore

class DeployEnv(object):
    def __init__(self):
        self._client = None
        self._runtime_client = None
        self._set_config_filename()
        self._load_yaml()

    def current_env(self):
        return os.environ.get("DEPLOY_ENV","local")

    def setting(self,name):
        return self.data["environments"][self.current_env()][name]

    def isDeployed(self):
        """
        Checks if the model is deployed.
        IMPORTANT: always returns `False` for local endpoints as LocalSagemakerClient.describe_endpoint()
        seems to always throw:
        botocore.exceptions.ClientError: An error occurred (ValidationException) when calling the describe_endpoint operation: Could not find local endpoint
        """
        _isDeployed = False
        try:
            self.client().describe_endpoint(EndpointName = self.setting("model_name"))
            _isDeployed = True
        except (botocore.exceptions.ClientError) as e:
            pass

        return _isDeployed


    def runtime_client(self):
        if self._runtime_client:
            return self._runtime_client

        if self.isLocal():
            self._runtime_client = sagemaker.local.LocalSagemakerRuntimeClient()
        else:
            self._runtime_client = boto3.client('sagemaker-runtime')

        return self._runtime_client

    def client(self):
        if self._client:
            return self._client

        if self.isLocal():
            self._client = sagemaker.local.LocalSagemakerClient()
        else:
            self._client = boto3.client('sagemaker')

        return self._client

    def isLocal(self):
        return self.current_env() == 'local'

    def isProduction(self):
        return self.current_env() == 'production'

    def _set_config_filename(self):
        config_dirname = os.path.dirname(__file__)
        config_filename = os.path.join(config_dirname, 'config.yml')
        self.config_filename = os.path.join(config_dirname, 'config.yml')

    def _load_yaml(self):
        file_handle = open(self.config_filename,'r')
        self.data = yaml.load(file_handle.read(), Loader=yaml.CLoader)
        file_handle.close();
        return self.data
