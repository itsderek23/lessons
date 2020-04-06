import boto3
import json
import sagemaker
from deploy_env import DeployEnv

env = DeployEnv()

print("Attempting to invoke model_name=%s / env=%s..." % (env.setting('model_name'), env.current_env()))

payload = {
	"columns": ["text"],
	"data": [["The Wimbledon tennis tournament starts next week!"],["The Canadian President signed in the new federal law."]]
}

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker-runtime.html#SageMakerRuntime.Client.invoke_endpoint
response = env.runtime_client().invoke_endpoint(
    EndpointName=env.setting("model_name"),
    ContentType="application/json",
    Accept="application/json",
    Body=json.dumps(payload)
)

print("Response=",response)
response_body = json.loads(response['Body'].read())
print(json.dumps(response_body, indent=4))
