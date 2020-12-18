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

`This tutorial <https://docs.microsoft.com/en-us/azure/databricks/scenarios/store-secrets-azure-key-vault>`_ explains how to connect an Azure Blob storage to Azure Databricks within Databricks notebooks (having the keys of the blob storage in Azure Key Vault).

Databricks-connect
--------------------------------------------------------------------------

To install it: https://menziess.github.io/howto/install/databricks-connect/ 

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
  
