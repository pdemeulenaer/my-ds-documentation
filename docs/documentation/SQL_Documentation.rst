
===============
 SQL Server documentation
===============

Basic SQL documentation
===============

.. topic:: Your Topic Title

    The objective here is to have everything useful for the projects, not to make a complete documentation of the whole SQL. 
    Let us have here the basics, as well as the useful (small) pieces of codes that are often used in our team.
	
Basic query
----------------

.. code-block:: sql
   :linenos:

   SELECT * FROM mytable   
   
Joining tables
----------------

Here is a graphical visualization of the different JOIN operations:

.. figure:: Images/Visual_SQL_JOINS.jpg
   :scale: 100 %
   :alt: map to buried treasure

   Taken from https://www.codeproject.com/Articles/33052/Visual-Representation-of-SQL-Joins
   
Keys in SQL
---------------------------------

Taken from https://stackoverflow.com/questions/24949676/difference-between-partition-key-composite-key-and-clustering-key-in-cassandra

* A primary key uniquely identifies a row.

* A composite key is a key formed from multiple columns.

* A partition key is the primary lookup to find a set of rows, i.e. a partition.

* A clustering key is the part of the primary key that isn't the partition key (and defines the ordering within a partition).

(is this specific to Cassandra, or valid for SQL in general?)

Examples:

PRIMARY KEY (a): The partition key is a.

PRIMARY KEY (a, b): The partition key is a, the clustering key is b.

PRIMARY KEY ((a, b)): The composite partition key is (a, b).

PRIMARY KEY (a, b, c): The partition key is a, the composite clustering key is (b, c).

PRIMARY KEY ((a, b), c): The composite partition key is (a, b), the clustering key is c.

PRIMARY KEY ((a, b), c, d): The composite partition key is (a, b), the composite clustering key is (c, d).

Import data from another server
----------------------------------

In some cases, data needs to be imported from another server. For that OPENROWSET is useful:

.. code-block:: sql
   :linenos:

   SELECT *
   INTO ETZ33839AA.dbo.HNWI_userlist
   FROM OPENROWSET(
    	'SQLNCLI', 'Server=CF4S01\INST001;Trusted_Connection=yes;',  
    	'Select *
        FROM [MCS_ModelDev_Score].[DK_PRV_2016].[test_pbe_pf_for_bda]'
   )
   
Other example, with condition:   
   
.. code-block:: sql
   :linenos:
   
   SELECT *
   INTO [ETZ3BSC1].[NT0001\BC1733].[SME_OOT_FINNISH_CUSTOMERS]
   FROM OPENROWSET(
       'SQLNCLI', 'Server=CF4S01\INST001 ;Trusted_Connection=yes;',  
       'SELECT le_unit_master, run_ts 
       FROM [MCS_ModelDev_Score].[GSM_Comp].[DEV_VAL_201511_0040_V2]
       WHERE R_NEW_B_TYPE = ''B_SC_13''
   '
   ) 
   

Data warehouse tables authorizations (works only for some tables)
--------------------------------------------------------------------------

Environment: etpew\inst004

1.	Find the table you need in MCS (example with a table EDW_HOVEDGR_H)
2.	Right click on the table properties -> permissions
3.	Under “Users and Roles” you will find something like XPEWEW2F but you do not need the XP part. The authorization name is EW-EW-2F.
 

 
 

The code below is also can be used instead of clicking Properties -> Permissions.

USE Etz3edw (use the correct environment name)
 
GO
 
exec sp_helprotect 'TABLE NAME' 

(the GRANTEE shows the package name, delete the beginning XP then it should be XX-XX-XX)
 
GO



Rank function - Exercise
---------------------------------

.. figure:: Images/Exercise_GiveScore_for_ClosestDate_result.jpg
   :scale: 100 %

.. code-block:: sql
   :linenos:
   
   --First: join on knid, build difference SCOREDATE-CREATIONDATE 
   select a.KNID,a.CREATIONDATE,b.SCOREDATE,b.SCORE, datediff(day,b.SCOREDATE,a.CREATIONDATE) as DateDifference
   into #temp
   from #t1 as a
   join #t2 as b on a.KNID = b.KNID
   where datediff(day,b.SCOREDATE,a.CREATIONDATE) > 0

   --Second: in DateDifference, the smallest positive value is the one we need. So we build a Rank on that,
   --        for each KNID--CREATIONDATE group (see the partition by clause)
   select KNID,CREATIONDATE,SCOREDATE,SCORE,DateDifference
   ,RANK() OVER   
    (PARTITION BY KNID,CREATIONDATE ORDER BY DateDifference ASC) AS Rank
   into #temp2
   from #temp

   --Third: we select Rank=1 to get the SCOREDATE AND SCORE for each KNID--CREATIONDATE combination
   select * from #temp2
   where Rank = 1
   order by CREATIONDATE desc

   select * from #result
   order by CREATIONDATE desc   
   
.. figure:: Images/Exercise_GiveScore_for_ClosestDate_result2.jpg
   :scale: 100 %   
   
   
Joining on KNID and earlier than some dates
---------------------------------------------------

We sometimes need to join data on KNID and on some date...but not exactly the same date, but table2.date <= table1.date... Seems tricky to do! Here is a way to do that:

.. code-block:: sql
   :linenos:
   
   SELECT ID, Date, Price 
   FROM (
   SELECT B.ID, B.Date, B.Price, ROW_NUMBER() OVER (PARTITION BY A.ID ORDER BY ABS(DATEDIFF(Day, A.Date, B.Date))) AS SEQ 
   FROM TableA AS A JOIN TableB AS B 
   ON A.ID=B.ID 
   WHERE B.Date<=A.Date ) AS T 
   WHERE SEQ=1
   
See https://social.msdn.microsoft.com/Forums/sqlserver/en-US/869b6f3f-a757-4a03-8704-96e4df734e29/find-closest-date-to-another-date?forum=transactsql   

Posgresql
========================================================

How to install client and server on Ubuntu 20.04: https://linuxconfig.org/ubuntu-20-04-postgresql-installation, https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-18-04

How to create user, db, create tables: https://www.digitalocean.com/community/tutorials/how-to-create-remove-manage-tables-in-postgresql-on-a-cloud-server
