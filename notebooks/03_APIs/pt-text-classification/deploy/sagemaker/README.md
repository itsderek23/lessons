# Deploy Text classification w/ <img src="https://raw.githubusercontent.com/madewithml/images/master/images/pytorch.png" width="25rem"> PyTorch to AWS SageMaker

These scripts create a local+production AWS Sagemaker development environment for the text classification PyTorch model. The local environment lets us iterate on the ML app significantly faster than waiting for an AWS deploy to complete.

__See the blog post [_How to setup a local AWS SageMaker environment for PyTorch_](https://booklet.ai/blog/aws-sagemaker-pytorch-local-dev-flow/) for a full walk-thru.__

## Usage

All examples assume the current working directory is:

```
[REPO_DIRECTORY]/notebooks/03_APIs/pt-text-classification
```

* __Environment-Specific settings__ - Place these in `deploy/sagemaker/config.yml`
* __Deploy__: `python deploy/sagemaker/deploy.py`
* __Test__: `python deploy/sagemaker/test.py`
* __Loading and serving the model__: See the functions defined in `deploy/sagemaker/serve.py`.

By default, all scripts use the `local` environment. To use `production`, set the `DEPLOY_ENV=production` environment variable. For example, to deploy to production:

```
DEPLOY_ENV=production python deploy/sagemaker/deploy.py
```
