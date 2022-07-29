==========================================================================
 Azure Cloud, Databricks
==========================================================================
 
Azure Cloud
==========================================================================

Azure Key Vault
--------------------------------------------------------------------------

`This tutorial <https://docs.microsoft.com/en-us/azure/databricks/scenarios/store-secrets-azure-key-vault>`_ explains how to save the keys of a blob storage in Azure Key Vault (and use that connection in Azure Databricks)

`This tutorial <https://microsoft-bitools.blogspot.com/2020/02/use-azure-key-vault-for-azure-databricks.html>` explains how to store secrets in Azure Key Vaults

Note that within Azure Databricks, if the workspace was created with Standard plan, during the creation of the secret scope (by adding #secrets/createScope to the workspace URL, so that url is like https://<\location>.azuredatabricks.net/?o=<\orgID>#secrets/createScope), in the Manage Principal option, we need to select the "users", otherwise it will not work. 

Azure Databricks
==========================================================================

Sasha Dittmann's lectures "Databricks MLOps - Deploy Machine Learning Model On Azure". Extremely useful to link Azure Databricks and Azure DevOps: 

https://www.youtube.com/watch?v=NLXis7FlnMM 

https://www.youtube.com/watch?v=HL36Q-eU5wU&t=198s

https://www.youtube.com/watch?v=fv3p3r3ByfY&t=1016s

Here is the full chain with Databricks' MLflow and azure ML: https://databricks.com/blog/2020/10/13/using-mlops-with-mlflow-and-azure.html Very good tutorial.

`This tutorial <https://docs.microsoft.com/en-us/azure/databricks/scenarios/store-secrets-azure-key-vault>`_ explains how to connect an Azure Blob storage to Azure Databricks within Databricks notebooks (having the keys of the blob storage in Azure Key Vault).

Deploying an MLflow model as a container on ACI and AKS: https://docs.azuredatabricks.net/_static/notebooks/mlflow/mlflow-quick-start-deployment-azure.html

Databricks workflow with databricks-connect: https://menziess.github.io/howto/enhance/your-databricks-workflow/

DBX - Databricks Extended CLI (supersedes Databricks-connect)
--------------------------------------------------------------------------

Main documentation:

- https://docs.databricks.com/dev-tools/dbx.html

- https://dbx.readthedocs.io/en/latest/properties_propagation.html

Some great examples:

- Create a DBX project: https://docs.microsoft.com/en-us/azure/databricks/dev-tools/dbx#--create-a-dbx-project

- Create a minimal template for CI/CD: https://docs.microsoft.com/en-us/azure/databricks/dev-tools/dbx#--create-a-dbx-templated-project-for-python-with-cicd-support

- Some next steps: https://docs.microsoft.com/en-us/azure/databricks/dev-tools/dbx#next-steps

What are the necessary steps needed in order to deploy and launch a custom code?

For a project which is meant to be deployed on the databricks platform as a package, use:

.. sourcecode:: python

  dbx deploy --jobs=validation --deployment-file=./conf/deployment-validation.json 
  dbx launch --job=validation --trace

For a code which is just deployed and run (without installing the project as a package on databricks), use:

.. sourcecode:: python

  dbx deploy --jobs=validation --no-rebuild --no-package --deployment-file=./conf/deployment-validation.json
  dbx launch --job=validation --trace

Databricks-connect
--------------------------------------------------------------------------

Intro blog from databricks: https://databricks.com/blog/2019/06/14/databricks-connect-bringing-the-capabilities-of-hosted-apache-spark-to-applications-and-microservices.html

Also, more recent one (2021): https://docs.databricks.com/dev-tools/databricks-connect.html

To install it: https://menziess.github.io/howto/install/databricks-connect/ (short intro)

See also https://docs.microsoft.com/en-us/azure/databricks/dev-tools/databricks-connect for a in-depth documentation

Good example of use of databricks-connect with pyspark UDF: https://medium.com/swlh/productionizing-a-spark-job-with-databricks-notebook-dd950a242c7d

How is databricks-connect maintained: https://pypi.org/project/databricks-connect/#history

Before installing databricks-connect, we need to uninstall pyspark, since they might conflict:

$ pip uninstall pyspark

Then install databricks-connect with the right version, the one that matches the databricks cluster version:

$ pip install -U databricks-connect==7.3.* 

Then we can configure the connection: https://docs.databricks.com/dev-tools/databricks-connect.html#step-2-configure-connection-properties 

Example here: https://menziess.github.io/howto/install/databricks-connect/

$ databricks-connect configure
The current configuration is:
* Databricks Host: https://westeurope.azuredatabricks.net/
* Databricks Token: dapi5c376de3a2a54a2b03016c8c3b123456 (build it yourself from settings)
* Cluster ID: 0214-195926-aptin821
* Org ID: 3892784943666666  (get it from /?o= argument in URL of cluster)
* Port: 8787

Run databricks-connect test to test your installation. You’ll hopefully see something along the lines of:

$ databricks-connect test

Tricks to set-up and optimize databricks-connect: https://dev.to/aloneguid/tips-and-tricks-for-using-python-with-databricks-connect-593k

**Fixing Out-of-Memory Issues**. Often when using Databricks Connect you might encounter an error like Java Heap Space etc. This simply means your local spark node (driver) is running out of memory, which by default is 2Gb. If you need more memory, it's easy to increase it. First, find out where PySpark's home directory is:

$ databricks-connect get-spark-home
/home/philippe/Documents/Github/Time-series-prediction/.venv/lib/python3.7/site-packages/pyspark

This should have a subfolder conf (create it if it doesn't exist). And a file spark-defaults.conf (again, create if doesn't exist). Full file path would be /home/philippe/Documents/Github/Time-series-prediction/.venv/lib/python3.7/site-packages/pyspark/conf/spark-defaults.conf. Add a line:

spark.driver.memory 8g (or 4g)

List of limitations of databricks-connect: https://datathirst.net/blog/2019/3/7/databricks-connect-limitations

How to connect data (azure storage for example) and also explore dbutils?

.. sourcecode:: python

  from pyspark.sql import SparkSession
  
  spark = SparkSession.builder.getOrCreate()
  
  setting = spark.conf.get("spark.master")
  if "local" in setting:
      from pyspark.dbutils import DBUtils
      dbutils = DBUtils(spark)  # HERE spark!  (in some places I saw spark.sparkContext, but wrong/outdated)
  else:
      print("Do nothing - dbutils should be available already")
  
  print(setting)
  # local[*] #when running from local laptop
  # spark... #when running from databricks notebook
  
  print(dbutils.fs.ls("dbfs:/"))
  
  # suppose the mnt/ is ALREADY mounted in your databricks cluster (do it in databricks, not from local)
  cwd = "/dbfs/mnt/demo/"

  # read from mnt point (could be Azure storage mounted there!)
  df = spark.read.csv("/mnt/demo/sampledata.csv")
  df.show()  
  
  +---+----------+---------+      
  |_c0|       _c1|      _c2|
  +---+----------+---------+
  | id| firstname| lastname|
  |  1|        JC|   Denton|
  +---+----------+---------+  
  
  # write to mount point
  (df.write
     .mode("overwrite")
     .parquet("/mnt/demo/sampledata_copy.parquet"))

Databricks CLI
--------------------------------------------------------------------------

Main doc: https://docs.databricks.com/dev-tools/cli/index.html

Installation and configuration:

.. sourcecode:: python

  # installation
  pip install databricks-cli 
  
  # configuration
  databricks configure --token
  
  > Databricks Host (should begin with https://): https://yourpath.azuredatabricks.net
  > Token: (put your token, get it from "Generate tokens" in User Settings)
  
  After you complete the prompts, your access credentials are stored in the file ~/.databrickscfg on Unix, Linux, or macOS, or %USERPROFILE%\.databrickscfg on Windows
  
  # list clusters:
  databricks clusters list
  > 1211-084728-chalk447  small_73ML   TERMINATED
  > 1217-223436-cab783    job-6-run-1  TERMINATED
  > 1217-222539-aunt76    job-5-run-1  TERMINATED  
  
  # delete a cluster permanently:
  databricks clusters permanent-delete --cluster-id 1217-223436-cab783
  
  # check again:
  databricks clusters list
  > 1211-084728-chalk447  small_73ML   TERMINATED
  > 1217-222539-aunt76    job-5-run-1  TERMINATED   
  
Note for multiple workspaces: 

.. sourcecode:: python

  # Multiple connection profiles are also supported with 
  databricks configure --profile <profile> [--token]
   
  # To use the profile associated to a workspace, use
  databricks <group> <command> --profile <profile-name>
  
  # For example: 
  databricks workspace ls --profile <profile>
  
The databricks cli is subdivided into sub-cli's:

* Workspace CLI: https://docs.databricks.com/dev-tools/cli/workspace-cli.html

* Clusters CLI: https://docs.databricks.com/dev-tools/cli/clusters-cli.html 

* Instance Pools CLI

* DBFS CLI: https://docs.databricks.com/dev-tools/cli/dbfs-cli.html

* Groups CLI

* Jobs CLI: https://docs.databricks.com/dev-tools/cli/jobs-cli.html

* Libraries CLI: https://docs.databricks.com/dev-tools/cli/libraries-cli.html

* Secrets CLI

* Stack CLI


More info: https://docs.databricks.com/dev-tools/cli/index.html

Centralized Databricks workspace
--------------------------------------------------------------------------

One can create a Databricks workspace which will contain centralized MLflow and Feature Store instances, that can be used from other workspaces (dev, staging, prod).

To connect such centralized workspace to each of the other ones, this is useful: 

For MLflow, simply do like here: https://cprosenjit.medium.com/mlflow-azure-databricks-7e7e666b7327

For Feature Store, one needs to use the metastore of the centralized workspace, and refer to it when working from clusters in other workspaces. See here for the metastore declaration in other workspaces: https://docs.microsoft.com/en-us/azure/databricks/data/metastores/external-hive-metastore . Then follow this to connect the different workspaces together: https://docs.microsoft.com/en-us/azure/databricks/applications/machine-learning/feature-store/multiple-workspaces. It is also very important to make sure that the hive client is connected to a centralized Azure Blob Storage: https://docs.databricks.com/data/data-sources/azure/azure-storage.html#access-azure-blob-storage-from-the-hive-client

To link a local workspace to the centralized workspace, one has first to create a personal access token into the centralized workspace, something like dapi1232142

Then, using the databricks cli (first configure it to be able to talk to the local workspace, see "Databricks CLI" section just above), in a bash shell you can:

databricks secrets create-scope --scope connection-to-data-workspace --initial-manage-principal users

where the "connection-to-data-workspace" is the name of the scope (that i chose), and the "--initial-manage-principal users" is needed when not in Premium workspace

Then 

databricks secrets put --scope connection-to-data-workspace 
--key data-workspace-host

This will request to enter the url of the CENTRALIZED workspace. Then

databricks secrets put --scope connection-to-data-workspace --key data-workspace-token

This will request the token of the PAT created in the CENTRALIZED workspace. Then finally

databricks secrets put --scope connection-to-data-workspace --key data-workspace-workspace-id\

This will request the workspace id of the CENTRALIZED workspace (contained in the URL of that workspace usually, if not can be obtained using CLI)

Note: 

* the doc on databricks secret scopes (https://docs.microsoft.com/en-us/azure/databricks/dev-tools/cli/secrets-cli) can be useful

* How to create secret scopes linked to Azure Key Vault: https://docs.microsoft.com/en-us/azure/databricks/scenarios/store-secrets-azure-key-vault


Delta Lake
--------------------------------------------------------------------------

Delta documentation: https://docs.delta.io/latest/delta-batch.html#overwrite, https://delta.io/

Introduction to delta lake: https://books.japila.pl/delta-lake-internals/installation/

Delta is the default file format in databricks.

Delta Lake brings ACID to object storage:

**Atomicity**: all transactions are either succeeded or failed completely

**Consistency**: guarantees that state of data is observed same by simultaneous operations

**Isolation**: simultaneous operations should not conflict with one another

**Durability**: changes are permanent

Problems solved by ACID:

1. Hard to append data: appends will not fail due to conflict, even when writing from multiple sources simultaneously

2. Modification of existing data difficult: upserts allow to apply updates and deletes as a simple atomic transaction

3. Jobs failing mid-way: changes are not committed until a job has succeeded. Jobs either fail or succeed completely (atomicity)

4. Real-time operations are hard: delta allows atomic micro-batch transaction processing in near real-time (within structured streaming). We can use both real-time and batch operations in the same set of delta lake tables

5. Costly to keep historical data versions: 

How to build a database in DataBricks (based on a lecture from DataBricks): time travel enabled on delta lake table using properties as atomicity, consistency and isolation of transactions

.. sourcecode:: python

  username = "my_name"
  dbutils.widgets.text("username", username)
  spark.sql(f"CREATE DATABASE IF NOT EXISTS dbacademy_{username}")
  spark.sql(f"USE dbacademy_{username}")
  health_tracker = f"/dbacademy/{username}/DLRS/healthtracker/"
  
Download some data to a raw place:

.. sourcecode:: python

  %sh
  wget https://hadoop-and-big-data.s3-us-west-2.amazonaws.com/fitness-tracker/health_tracker_data_2020_1.json
  
  # Then have a look to raw place:
  %sh ls
  
conf
derby.log
eventlogs
health_tracker_data_2020_1.json

Then mode data to raw directory:

#Step 3: Move the data to the raw directory

.. sourcecode:: python

  dbutils.fs.mv("file:/databricks/driver/health_tracker_data_2020_1.json", 
                health_tracker + "raw/health_tracker_data_2020_1.json")
                
Load the data as a Spark DataFrame from the raw directory. This is done using the .format("json") option:

.. sourcecode:: python

  file_path = health_tracker + "raw/health_tracker_data_2020_1.json"   
  health_tracker_data_2020_1_df = (spark.read.format("json").load(file_path))                
                
# Next, we remove the files in the /dbacademy/DLRS/healthtracker/processed directory. This step will make the notebook idempotent. In other words, it could be run more than once without throwing errors or introducing extra files.

.. sourcecode:: python

  dbutils.fs.rm(health_tracker + "processed", recurse=True)      
  
Then transform data:

.. sourcecode:: python

  from pyspark.sql.functions import col, from_unixtime

  def process_health_tracker_data(dataframe):
    return (
      dataframe
      .withColumn("time", from_unixtime("time"))
      .withColumnRenamed("device_id", "p_device_id")
      .withColumn("time", col("time").cast("timestamp"))
      .withColumn("dte", col("time").cast("date"))
      .withColumn("p_device_id", col("p_device_id").cast("integer"))
      .select("dte", "time", "heartrate", "name", "p_device_id")
      )
    
  processedDF = process_health_tracker_data(health_tracker_data_2020_1_df)
                
Then write the file in processed dir (Note that we are partitioning the data by device id):

.. sourcecode:: python

  (processedDF.write
   .mode("overwrite")
   .format("parquet")
   .partitionBy("p_device_id")
   .save(health_tracker + "processed"))

Next, Register the table in the metastore:

.. sourcecode:: python
  
  %sql 
  
  DROP TABLE IF EXISTS health_tracker_processed;
  
  CREATE TABLE health_tracker_processed                        
  USING PARQUET                
  LOCATION "/dbacademy/$username/DLRS/healthtracker/processed"  
  
File optimization with Delta Lake: https://docs.databricks.com/delta/optimizations/file-mgmt.html#language-python

1. Compaction (bin-packing) (https://docs.databricks.com/delta/optimizations/file-mgmt.html#compaction-bin-packing):

Delta Lake on Databricks (starting from Runtime 11) can improve the speed of read queries from a table. One way to improve this speed is to coalesce small files into larger ones. You trigger compaction by running the OPTIMIZE command:

.. sourcecode:: python

  from delta.tables import *
  deltaTable = DeltaTable.forPath(spark, "/data/events") # by table path
  deltaTable.optimize().executeCompaction()
  
  # or
  
  from delta.tables import *
  deltaTable = DeltaTable.forName(spark, "events") # by table name
  deltaTable.optimize().executeCompaction()
  
If you have a large amount of data and only want to optimize a SUBSET of it, you can specify an optional partition predicate using WHERE:

.. sourcecode:: python
  
  from delta.tables import *
  deltaTable = DeltaTable.forName(spark, "events")
  deltaTable.optimize().where("date='2021-11-18'").executeCompaction() # COMPACTION ONLY FOR THE DATE SELECTED
  
2. Z-Ordering (https://docs.databricks.com/delta/optimizations/file-mgmt.html#z-ordering-multi-dimensional-clustering)"

  
Example notebooks: https://docs.databricks.com/delta/optimizations/optimization-examples.html
  
Azure Data Factory (ADF)
--------------------------------------------------------------------------

- Airflow or ADF in Azure: https://blog.dataminded.com/batch-orchestration-on-azure-flowchart-42947008b4ca

Azure ML
--------------------------------------------------------------------------

Deployment of python ML databricks notebooks on Azure ML (through MLflow): https://medium.com/pgs-software/mlflow-tracking-ml-model-changes-deployment-in-azure-7bc6ba74f47e
