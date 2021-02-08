==========================================================================
 Azure Cloud, Databricks
==========================================================================
 
Azure Cloud
==========================================================================

Azure Key Vault
--------------------------------------------------------------------------

`This tutorial <https://docs.microsoft.com/en-us/azure/databricks/scenarios/store-secrets-azure-key-vault>`_ explains how to save the keys of a blob storage in Azure Key Vault (and use that connection in Azure Databricks)

Azure Databricks
==========================================================================

Sasha Dittmann's lectures "Databricks MLOps - Deploy Machine Learning Model On Azure". Extremely useful to link Azure Databricks and Azure DevOps: 

https://www.youtube.com/watch?v=NLXis7FlnMM 

https://www.youtube.com/watch?v=HL36Q-eU5wU&t=198s

https://www.youtube.com/watch?v=fv3p3r3ByfY&t=1016s

Here is the full chain with Databricks' MLflow and azure ML: https://databricks.com/blog/2020/10/13/using-mlops-with-mlflow-and-azure.html Very good tutorial.

`This tutorial <https://docs.microsoft.com/en-us/azure/databricks/scenarios/store-secrets-azure-key-vault>`_ explains how to connect an Azure Blob storage to Azure Databricks within Databricks notebooks (having the keys of the blob storage in Azure Key Vault).

Databricks-connect
--------------------------------------------------------------------------

Intro blog from databricks: https://databricks.com/blog/2019/06/14/databricks-connect-bringing-the-capabilities-of-hosted-apache-spark-to-applications-and-microservices.html

Also, more recent one (2021): https://docs.databricks.com/dev-tools/databricks-connect.html

To install it: https://menziess.github.io/howto/install/databricks-connect/ (short intro)

See also https://docs.microsoft.com/en-us/azure/databricks/dev-tools/databricks-connect for a in-depth documentation

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

Installation and configuration:

.. sourcecode:: python

  # installation
  pip install databricks-cli 
  
  # configuration
  databricks configure --token
  
  > Databricks Host (should begin with https://): https://yourpath.azuredatabricks.net
  > Token: (put your token, get it from "Generate tokens" in User Settings)
  
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
  
The databricks cli is subdivided into sub-cli's:

    Workspace CLI: https://docs.databricks.com/dev-tools/cli/workspace-cli.html
    Clusters CLI: https://docs.databricks.com/dev-tools/cli/clusters-cli.html 
    Instance Pools CLI
    DBFS CLI: https://docs.databricks.com/dev-tools/cli/dbfs-cli.html
    Groups CLI
    Jobs CLI: https://docs.databricks.com/dev-tools/cli/jobs-cli.html
    Libraries CLI: https://docs.databricks.com/dev-tools/cli/libraries-cli.html
    Secrets CLI
    Stack CLI




  
  
More info: https://docs.databricks.com/dev-tools/cli/index.html

Delta Lake
--------------------------------------------------------------------------

How to build a database in DataBricks (based on a lecture from DataBricks):

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
  
