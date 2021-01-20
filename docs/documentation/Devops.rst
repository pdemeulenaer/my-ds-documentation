==========================================================================
 DevOps
==========================================================================

Airflow
==========================================================================

Links: 

Server installation with database: https://www.statworx.com/de/blog/a-framework-to-automate-your-work-how-to-set-up-airflow/
First dag: https://medium.com/better-programming/how-to-write-your-first-pipeline-in-airflow-a51141c3f4dd
How to unit-test Airflow: https://godatadriven.com/blog/testing-and-debugging-apache-airflow/ 


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

- Intro to docker & kubernetes video: https://www.youtube.com/watch?v=bhBSlnQcq2k&ab_channel=Amigoscode

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

- How to start, official site: https://kubernetes.io/docs/home/

- 50 days to Kubernetes, Azure (AKS): https://azure.microsoft.com/mediahandler/files/resourcefiles/kubernetes-learning-path/Kubernetes%20Learning%20Path%20version%201.0.pdf

Intro to docker & kubernetes video: https://www.youtube.com/watch?v=bhBSlnQcq2k&ab_channel=Amigoscode

Spark on Kubernetes
--------------------------------------------------------------------------

Good links:

- https://levelup.gitconnected.com/spark-on-kubernetes-3d822969f85b

Openshift
==========================================================================

Learning: https://learn.openshift.com/

CICD developement
==========================================================================

what is it exactly?

- https://kumul.us/understanding-cicd-continuous-integration-deployment-delivery/

Interesting series of Videos on CI/CD of a ML model (NLP detection of positive/negative comments) using GitLab, Jenkins and Flask (and Docker):

- video 1: installation of Docker, GitLab, Jenkins: https://www.youtube.com/watch?v=SUyHDYb1aBM&ab_channel=ThinkwithRiz
- video 2: model building using Flask: https://www.youtube.com/watch?v=XT2TFQexYrg&ab_channel=ThinkwithRiz
- video 3: deploy NLP Machine Learning Model Flask App to Docker: https://www.youtube.com/watch?v=rb_DkKAZzyA&t=5s&ab_channel=ThinkwithRiz
- video 4: Flask Application End To End CI-CD Pipeline using Jenkins & GitLab: https://www.youtube.com/watch?v=sg1S7A532gM&ab_channel=ThinkwithRiz

GitLab
--------------------------------------------------------------------------

Jenkins
--------------------------------------------------------------------------

Azure DevOps
--------------------------------------------------------------------------

Azure DevOps with azure container registry (for Docker container images): https://docs.microsoft.com/en-us/azure/devops/pipelines/ecosystems/containers/acr-template?view=azure-devops

Tests
--------------------------------------------------------------------------

- regression testing

- performance testing

- coverage testing: tps://www.guru99.com/test-coverage-in-software-testing.html 

Git Flow
--------------------------------------------------------------------------

Intro here: https://medium.com/@patrickporto/4-branching-workflows-for-git-30d0aaee7bf

Git flow extended demo: https://datasift.github.io/gitflow/IntroducingGitFlow.html

The one of Azure DevOps: https://docs.microsoft.com/en-us/azure/devops/repos/tfvc/effective-tfvc-branching-strategies-for-devops?view=azure-devops

==========================================================================
MLOps - Machine learning life cycle
==========================================================================

ML platforms: https://medium.com/better-programming/5-great-mlops-tools-to-launch-your-next-machine-learning-model-3e403d0c97d3



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

- through the MLFlow CLI: https://www.mlflow.org/docs/latest/cli.html

See the components of MLFlow here: https://www.mlflow.org/docs/latest/concepts.html#mlflow-components 

Good links:

- Main concepts of MLFlow: https://www.mlflow.org/docs/latest/concepts.html

- https://blog.noodle.ai/introduction-to-mlflow-for-mlops-part-1-anaconda-environment/

- How to track MLFlow using the Databricks MLFlow server (accessed from local environment): https://docs.databricks.com/applications/mlflow/access-hosted-tracking-server.html

- tutorial: https://www.adaltas.com/en/2020/03/23/mlflow-open-source-ml-platform-tutorial

- Setup MLflow in Production: https://towardsdatascience.com/setup-mlflow-in-production-d72aecde7fef, https://pedro-munoz.tech/how-to-setup-mlflow-in-production/, https://medium.com/datatau/how-to-setup-mlflow-in-production-a6f70511ebdc

- Deploy MLflow with Docker-compose: https://towardsdatascience.com/deploy-mlflow-with-docker-compose-8059f16b6039

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

SonarQube (or SonarCloud, SonarLint): static code analysis
--------------------------------------------------------------------------

How to use SonarQube with Azure DevOps: https://docs.sonarqube.org/latest/analysis/azuredevops-integration/

- SonarQube: on prem version

- SonarCloud: cloud available version

- SonarLint: version available in IDEs, like VSCode. 

How to use the cloud version of SonarQube, SonarCloud, with Azure DevOps: 

- https://azuredevopslabs.com/labs/vstsextend/sonarcloud/

- https://www.codewrecks.com/blog/index.php/2019/01/05/sonar-analysis-of-python-with-azure-devops-pipeline/ : with concrete YAML file example

What does SonarQube/SonarCloud "better" than classical linting tools like flake8 or pylint? 

- https://blog.sonarsource.com/sonarcloud-finds-bugs-in-high-quality-python-projects

- https://community.sonarsource.com/t/is-that-possible-to-use-pylint-with-sonarqube/24874 (also introduces how to mix pylint & Sonar)

Rule explorer: here is the list of 174 rules used in SonarQube/SonarCloud/SonarLint: https://rules.sonarsource.com/python (for Python, but exists also for different languages)







