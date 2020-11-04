==========================================================================
 Azure Cloud, DataBricks
==========================================================================
 
Azure Cloud
==========================================================================

Azure DataBricks
==========================================================================

Sasha Dittmann's lectures "Databricks MLOps - Deploy Machine Learning Model On Azure". Extremely useful to link Azure Databricks and Azure DevOps: 

https://www.youtube.com/watch?v=NLXis7FlnMM 

https://www.youtube.com/watch?v=HL36Q-eU5wU&t=198s

https://www.youtube.com/watch?v=fv3p3r3ByfY&t=1016s

Delta Lake
--------------------------------------------------------------------------

How to build a database in DataBricks:

.. sourcecode:: python

  username = "my_name"
  dbutils.widgets.text("username", username)
  spark.sql(f"CREATE DATABASE IF NOT EXISTS dbacademy_{username}")
  spark.sql(f"USE dbacademy_{username}")
  health_tracker = f"/dbacademy/{username}/DLRS/healthtracker/"
