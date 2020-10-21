==========================================================================
 DevOps
==========================================================================

Airflow
==========================================================================

Links: 

Server installation with database: https://www.statworx.com/de/blog/a-framework-to-automate-your-work-how-to-set-up-airflow/
First dag: https://medium.com/better-programming/how-to-write-your-first-pipeline-in-airflow-a51141c3f4dd


Docker
==========================================================================

Good links:

The 3 following resources form a series very simple to follow and reproduce, and very useful to grab basic concepts:

- https://mlinproduction.com/docker-for-ml-part-1/ : this basically shows how to run jupyter notebook from a docker image. 

- https://mlinproduction.com/docker-for-ml-part-2/ : this shows how to build a docker image.

- https://mlinproduction.com/docker-for-ml-part-3/ : this shows an example of ML sklearn model packed in a docker image. 

Additional interesting resources:

- https://medium.com/better-programming/how-to-get-docker-to-play-nicely-with-your-python-data-science-packages-81d16f1080d2 

- https://medium.com/@itembe2a/docker-nvidia-conda-h204gpu-make-an-ml-docker-image-47451c5ced51 

- https://towardsdatascience.com/docker-for-data-science-9c0ce73e8263

High-level principle of Docker images:

.. figure:: Images/Docker_principle.png
   :scale: 100 %
   :alt: Docker images principle
   
Activate a conda environment in Docker: https://pythonspeed.com/articles/activate-conda-dockerfile/   

Spark on Docker: https://www.datamechanics.co/blog-post/spark-and-docker-your-spark-development-cycle-just-got-ten-times-faster

Kubernetes
==========================================================================

Good links:

Again, a very interesting series to address the basics concepts:

- https://mlinproduction.com/k8s-pods/ : part 1 of the series, on Kubernetes Pods 

- https://mlinproduction.com/k8s-jobs/ : part 2 of the series, on jobs

- https://mlinproduction.com/k8s-cronjobs/ : part 3 of the series, on CronJobs

- https://mlinproduction.com/k8s-deployments/ : part 4 of the series, on deployments

- https://mlinproduction.com/k8s-services/ : part 5 of the series

Spark on Kubernetes
--------------------------------------------------------------------------

Good links:

- https://levelup.gitconnected.com/spark-on-kubernetes-3d822969f85b



CICD developement
==========================================================================

what is it exactly?

- https://kumul.us/understanding-cicd-continuous-integration-deployment-delivery/

Tests
--------------------------------------------------------------------------

- regression testing

- performance testing

- coverage testing: tps://www.guru99.com/test-coverage-in-software-testing.html 

Git Flow
--------------------------------------------------------------------------

Intro here: https://medium.com/@patrickporto/4-branching-workflows-for-git-30d0aaee7bf

==========================================================================
MLOps - Machine learning life cycle
==========================================================================

MLFlow
==========================================================================

Advantages:

MLFlow is a Data Science platform built with machine learning model development, versioning and deployment in mind.

Developed by Databricks, open-sourced, and donated to Linux foundation. As such, heavily documented. Became de-facto standard in last 2 years

For development, ability to log parameters (see tracking API)

For deployment, ability to version-control model, and tag model: none-staging-production-archived (see model registry API)

the open source version exists as a server-client application, accessible through:

- a user-friendly (data scientist-friendly) UI

- through python APIs

- through the MLFlow CLI

See the components of MLFlow here: https://www.mlflow.org/docs/latest/concepts.html#mlflow-components 

Good links:

- Main concepts of MLFlow: https://www.mlflow.org/docs/latest/concepts.html

- https://towardsdatascience.com/setup-mlflow-in-production-d72aecde7fef

- https://pedro-munoz.tech/how-to-setup-mlflow-in-production/ ; https://medium.com/datatau/how-to-setup-mlflow-in-production-a6f70511ebdc

- https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039

- https://blog.noodle.ai/introduction-to-mlflow-for-mlops-part-1-anaconda-environment/

MLFLow Tracking: https://www.mlflow.org/docs/latest/tracking.html
--------------------------------------------------------------------------

Log scikit-learn models in MLFlow: https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.log_model

Log pyspark models in MLFlow: https://mlflow.org/docs/latest/python_api/mlflow.spark.html#mlflow.spark.log_model

Log tensorflow models in MLFlow: https://www.mlflow.org/docs/latest/python_api/mlflow.tensorflow.html#mlflow.tensorflow.log_model

MLFLow Projects: https://www.mlflow.org/docs/latest/projects.html
--------------------------------------------------------------------------

MLFlow Models: https://www.mlflow.org/docs/latest/models.html
--------------------------------------------------------------------------

MLFlow Model registry: https://www.mlflow.org/docs/latest/model-registry.html
--------------------------------------------------------------------------

Model registry example: https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/model-registry-example 

* How to register a model using the model registry UI: https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/model-registry-example#register-and-manage-the-model-using-the-mlflow-ui

* How to register a model in the model registry in the MLFlow API: https://docs.microsoft.com/en-us/azure/databricks/applications/mlflow/model-registry-example#register-and-manage-the-model-using-the--mlflow-api 

MLFlow server
--------------------------------------------------------------------------



