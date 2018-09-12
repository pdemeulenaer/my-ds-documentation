===============
 Python, Jupyter and Pandas
===============

Basic Jupyter infos
===============

Here are useful shortcuts for the Jupyter Notebooks:

.. figure:: Cheatsheets/Jupyter_Notebook_Keyboard_Shorcuts.png
   :scale: 100 %
   :alt: map to buried treasure

   This Cheatsheet is taken from cheatography.com/weidadeyue/ 

When you launch a Jupyter notebook, you can adjust its width size using the following command at the beginning:
   
.. sourcecode:: python

   from IPython.core.display import display, HTML
   display(HTML("<style>.container { width:100% !important; }</style>"))
   
   #or better:
   
   display(HTML("<style>.container { width:95% !important; font-size:10px; font-weight:bold;}</style>"))
   
Loading package from a given (maybe different) directory:

.. sourcecode:: python

   import sys
   sys.path.append('/home/BB2907/GIT/affordabillity/pyspark') 
   
Getting Notebooks work on server and access them using ssh
=================================================================

How to keep jupyter notebook (or pyspar3Jupyter) active through ssh:
Go to server with ssh (using putty)
type: nohup pyspar3Jupyter > save.txt &  (by this the save.txt contains the address of the notebook)
type: jobs -l get pid number, this will be useful when you want to kill your pyspark session.
ps -aux | grep bc4350 (this gets the pid if the ssh has already been shut down
kill 5442 (if pid=5442)
   


Numpy basic documentation
===========================

.. figure:: Cheatsheets/Numpy_Python_Cheat_Sheet.png
   :scale: 100 %
   :alt: map to buried treasure

   This Cheatsheet is taken from DataCamp. 


Basic Pandas documentation
===============

.. topic:: Introduction

    The objective here is to have everything useful for the projects, not to make a complete documentation of the whole package. Here I will try to document both version 1.6 and >2.0. A special enphase will be done on machine learning module ml (mllib is outdated).
 
 
Good Pandas links:
----------------------------

A good link on data manipulations: https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/
   
Loading Pandas dataframe from file
------------------------------------------

.. sourcecode:: python

  #Loading a Pandas dataframe:
  df_pd = pd.read_csv("/home/BC4350/Desktop/Iris.csv")
   
   
Creation of some data in a Pandas dataframe
-----------------------------------------------

.. sourcecode:: python

  # A set of baby names and birth rates:
  names = ['Bob','Jessica','Mary','John','Mel']
  births = [968, 155, 77, 578, 973]

  #We merge the 2 lists using the zip function:
  BabyDataSet = list(zip(names,births))

  #We create the DataFrame:
  df = pd.DataFrame(data = BabyDataSet, columns=['Names', 'Births'])

   	Names 	Births
  0 	Bob 	968
  1 	Jessica 155
  2 	Mary 	77
  3 	John 	578
  4 	Mel 	973
  
  
Re-setting of index in Pandas dataframes
---------------------------------------------------

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.set_index.html  

Iterating over Pandas dataframe rows:
---------------------------------------------------

http://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.DataFrame.iterrows.html

A simple example:

.. sourcecode:: python

  for i, row in df.iterrows()
    print(row)

Check number of nulls in each feature column
-----------------------------------------------------

.. sourcecode:: python

  # This will output all column names and the number of nulls in them
  nulls_per_column = df.isnull().sum()
  print(nulls_per_column)    
  
Identify which columns are categorical and which are not (important for some ML algorithms)
--------------------------------------------------------------------  
  
.. sourcecode:: python  
  
  # Create a boolean mask for categorical columns
  categorical_feature_mask = df.dtypes == object

  # Get list of categorical column names
  categorical_columns = df.columns[categorical_feature_mask].tolist()

  # Get list of non-categorical column names
  non_categorical_columns = df.columns[~categorical_feature_mask].tolist()  
  
Deleting a column, or list of columns:  
---------------------------------------------------

.. sourcecode:: python

  df.drop(['column'], axis=1, inplace=True)
  df.drop(['column1','column2'], axis=1, inplace=True)

  
Displaying dataframes to screen
-----------------------------------------

.. sourcecode:: python

  #This allows you to display as many rows as you wish when you display the dataframe (works also for max_rows):
  pd.options.display.max_columns = 50   #By default 20 only  
  
  #This display the 5 first rows:
  df.head(5)
  
  #This display the 5 last rows:
  df.tail(5)  
  
  
Reading JSON blobs (from command line)  
-----------------------------------------------------

.. sourcecode:: python

  import pandas as pd
  import sys  
  json_string = sys.argv[1]
  print(pd.DataFrame(eval(json_string)))
  
  # We run the code like this: python test_json.py {'knid':{'0':'123456','1':'123456','2':'123457'},'score':{'0':'C2-1','1':'C2-2','2':'C4-1'},'join_dt':{'0':'2018-01-01','1':'2018-01-02','2':'2018-01-03'}}
  
.. figure:: Images/Json_output.png
   :scale: 100 %
   :alt: Json output
   
  
Getting the gender from Danish CPR number  
-----------------------------------------------------

.. sourcecode:: python

  dff = pd.DataFrame({'a': [1,2,3], 'knid': ['1305810001','1305810002','1305810004']})
  dff.loc[dff['knid'].str[9:10].astype(int) % 2 == 1,'gender'] = 'male'
  dff.loc[dff['knid'].str[9:10].astype(int) % 2 == 0,'gender'] = 'female'
  dff
  
    a knid       gender
  0 1 1305810001 male 
  1 2 1305810002 female 
  2 3 1305810004 female 
  
  
Retrieval of data from SQL data warehouse
-----------------------------------------------------

This exports the data in a simple array:

.. sourcecode:: python

  import pyodbc as odbc 

  # Some super SQL query
  sql = """SELECT top 100
  _ts_from as RUN_TS
  ,b.[AC_KEY]
  ,[PROBABILITY_TRUE]
  FROM [MCS_BATCH].[test].[B_DCS_DK_ROL] b
  JOIN mcs_batch.ctrl.run_info r ON r.RUN_ID=b.RUN_ID
  """
  conn = odbc.connect(r'Driver={SQL Server};Server=CF4S01\INST001;Database=MCS_BATCH;Trusted_Connection=yes;')
  crsr = conn.cursor()
  crsr.execute(sql)
  params=crsr.fetchall()
  crsr.close()
  conn.close()
  
  
But if we want to have the data immediately loaded into a dataframe, then we can use these functions:

.. sourcecode:: python

  import pypyodbc as odbc

  def Extract_data_from_SQLserver(Server,DataBase,SQLcommand):    
    cnxn = odbc.connect(r'Driver={SQL Server};Server='+Server+';Database='+DataBase+';Trusted_Connection=yes;') 
    cursor = cnxn.cursor()
    
    #THE EXTRACTION OF HEADER AND DATA
    res = cursor.execute(SQLcommand)
    header = [tuple[0] for tuple in res.description]
    data = cursor.fetchall()
    
    #WRITING RESULT TO CSV
    df = pd.DataFrame(data, columns=header)
    cursor.close()
    cnxn.close()
    return df
	
	
  #And we can use it like this:	
  #some SQL command: 	
  SQLcommand = """
  select *
  from ETZ33839AA.dbo.HNWI_main_data_step5
  order by inv_id, the_months
  """

  df = Extract_data_from_SQLserver('etpew\INST004','ETZ33839AA',SQLcommand)
  
  
Exporting data to SQL warehouse
-------------------------------------------

Let's say we have some dataframe, here FinalListModel1:

.. sourcecode:: python

  import pypyodbc as odbc

  conn = odbc.connect(r'Driver={SQL Server};Server=CF4S01\INST001;Database=IMD_ML;Trusted_Connection=yes;')

  rows1 = list(FinalListModel1['caseid']) 
  rows2 = list(FinalListModel1['recordkey'])
  rows3 = list(FinalListModel1['score1'])
  rows = list(zip(rows1,rows2,rows3))

  cursor = conn.cursor() 

  stm="""
  DROP TABLE [MCS_ModelDev_BigDataAnalytics].[dbo].[DEBT_COL_OUTPUT]
  CREATE TABLE [MCS_ModelDev_BigDataAnalytics].[dbo].[DEBT_COL_OUTPUT] (
      [caseid] nvarchar(255),
      [recordkey] nvarchar(255),
      [score1] float
  )
  """
  res = cursor.execute(stm)
  cursor.executemany('INSERT INTO [MCS_ModelDev_BigDataAnalytics].[dbo].[DEBT_COL_OUTPUT] VALUES (?, ?, ?)', rows)
  conn.commit()
  
  cursor.close()
  conn.close()  

  
Apply function to all rows (axis=1) or to all columns (axis=0):
--------------------------------------------------------------------------------

.. sourcecode:: python

  #We need a function: here it counts the number of NaN in a x object
  def num_missing(x):
    return sum(x.isnull())

  #Applying per column:
  print "Missing values per column:"
  print df.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

  #Applying per row:
  print "Missing values per row:"
  print df.apply(num_missing, axis=1).head() #axis=1 defines that function is to be applied on each row
  
See also http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.apply.html#pandas.DataFrame.apply    
  
Note that it is also possible to add arguments of the function (if it has) in an "args" parameter of apply:
for example: df.apply(your_function, args=(2,3,4) )  
Here other example: 

.. sourcecode:: python

  def subtract_custom_value(x, custom_value):
    return x-custom_value
    
  df.apply(subtract_custom_value, args=(5,))
    
See also http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.apply.html#pandas.Series.apply
  

  
Group by operations in Pandas
------------------------------------------------

For a dataframe df with column ID, we can create a group by ID and count like this:

.. sourcecode:: python

  df.groupby(['ID']).size().reset_index(name='count')
  
  #Or equivalently:
  df.groupby(['ID']).size().rename('count').reset_index()
  
Where the rename just gives a name to the new column created (the count) and the reset_index gives a dataframe shape to the grouped object.
  

Multiple aggregation on groups:

.. sourcecode:: python

  #Here if we want to aggregate on several standard methods, like sum and max:
  
  df.groupby(['ID'])[['age','height']].agg(['max','sum'])
  
  #We can also aggrgate using a user-defined function:
  
  def data_range(series):
    return series.max() - series.min()
	
  df.groupby(['ID'])[['age','height']].agg(data_range)
  
  #We can also use dictionaries (to add names to aggregates):
  df.groupby(['ID'])[['age','height']].agg({'my_sum':'sum','my_range':data_range)
  

In the case we want to make counts of the biggest groups in a dataframe:

.. sourcecode:: python 
  
  #If we want to group by only one feature, "ID" and see which are biggest groups, then the simplest is:
  df['ID'].value_counts()
  
  #Equivalently (same result), we can use:
  df[['ID']].groupby(['ID']).size().sort_values(ascending=False)
  #or: df[['ID']].groupby(['ID']).size().reset_index(name="count").sort_values("count",ascending=False) for a df with named column
  
.. figure:: Images/Groupby0.png
   :scale: 70 %
   :alt: map to buried treasure
   
.. sourcecode:: python 
  
  #Equivalently (same result but with named "count" column), we can use:
  df[['ID']].groupby(['ID']).size().reset_index(name="count").sort_values("count",ascending=False)   
  
In the case we want several features to be grouped, the second method hereabove is appropriate:

.. sourcecode:: python

  #Equivalently (same result), we can use:
  df[['ID','merchant','Target2']].groupby(['ID','merchant','Target2']).size().sort_values(ascending=False)
  
  #This produces the series at left, in the following figure.
  
  #An equivalent way outputs the same info but as a dataframe (with named new column), not a pandas series:
  df[['ID','merchant','Target2']].groupby(['ID','merchant','Target2']).size().reset_index(name='count').sort_values(['count'],ascending=False)
  
.. figure:: Images/Groupby1.png
   :scale: 70 %
   :alt: map to buried treasure  
   

Ranking inside groups
-----------------------------------------------------

Let's say you want to rank data grouped by some columns: (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.core.groupby.DataFrameGroupBy.rank.html )
We start from some dataframe:

.. sourcecode:: python

     caseid  merchant time
  0       1         a    1 
  1       1         a    2 
  2       1         a    3 
  3       2         b    1 
  4       2         b    2 
  5       2         c    1 

.. sourcecode:: python

  df['rank'] = df.groupby(['caseid','merchant'])['time'].rank(ascending=False).astype(int) 
  #Result:

     caseid  merchant time rank 
  0       1         a    1    3
  1       1         a    2    2
  2       1         a    3    1
  3       2         b    1    2
  4       2         b    2    1
  5       2         c    1    1
   
Apply vs transform operations on groupby objects
-----------------------------------------------------

Investigate here: https://stackoverflow.com/questions/27517425/apply-vs-transform-on-a-group-object

Comparison SQL-Pandas
------------------------------

An EXCELLENT post comparing Pandas and SQL is here: https://codeburst.io/how-to-rewrite-your-sql-queries-in-pandas-and-more-149d341fc53e

SQL-like WINDOW function... how to do in Pandas?

Here is a good example of SQL window function:
A first SQL query:
  
.. sourcecode:: python

  SELECT state_name,  
       state_population,
       SUM(state_population)
        OVER() AS national_population
  FROM population   
  ORDER BY state_name 

Pandas:

.. sourcecode:: python

  df.assign(national_population=df.state_population.sum()).sort_values('state_name')

A second SQL query:

.. sourcecode:: python

  SELECT state_name,  
       state_population,
       region,
       SUM(state_population)
        OVER(PARTITION BY region) AS regional_population
  FROM population    
  ORDER BY state_name

Pandas: (here on ONE COLUMN! the "state_population")

.. sourcecode:: python

  df.assign(regional_population=df.groupby('region')['state_population'].transform('sum')).sort_values('state_name')

  
In general, comparison between simple SQL and Pandas operations: http://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html  

A simple selection for a few different id, in SQL:

.. sourcecode:: python

  SELECT KNID,CREATIONDATE,CREDIT_SCORE,produkt_count,customer_since
  FROM table
  WHERE KNID in('0706741860','2805843406','2002821926','0711691685','0411713083')

And with pandas:  
  
.. sourcecode:: python  
  
  knid_list = ['0706741860','2805843406','2002821926','0711691685','0411713083']
  for i,item in enumerate(knid_list):
      if i==0: filter_knids = (data['KNID']==item)
      if i>0 : filter_knids = (data['KNID']==item)|filter_knids        
  data.loc[filter_knids,['KNID','CREATIONDATE','CREDIT_SCORE','produkt_count','customer_since']]
  
Merging and Concatenation operations
---------------------------------------------------
In Pandas, all types of merging operations (the "join" in SQL) are done using the  :py:func:`merge` command (see http://pandas.pydata.org/pandas-docs/stable/merging.html ): 

.. sourcecode:: python

   pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
         left_index=False, right_index=False, sort=True,
         suffixes=('_x', '_y'), copy=True, indicator=False)
		 
Note: if you need to merge 2 dataframes using several columns at the same time, it is possible:

.. sourcecode:: python

   new_df = pd.merge(A_df, B_df,  how='inner', left_on=['A_c1','c2'], right_on = ['B_c1','c2'])
		

Here is an excellent comparison between SQL and Pandas: http://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html#compare-with-sql-join


Melting operation
---------------------------------

The melt operation simply reorganizes the dataframe. Let's say we have this df:

.. sourcecode:: python

  df = pd.DataFrame([[2, 4, 7, 8, 1, 3, 2013], [9, 2, 4, 5, 5, 6, 2014]], columns=['Amy', 'Bob', 'Carl', 'Chris', 'Ben', 'Other', 'Year'])
  df
  
.. figure:: Images/PandasMelt1.png
   :scale: 100 %
   :alt: Pandas Melt  
  
Now we want to reorganize the df so that we have one column "Year" and one column "Name", which contains all name. We then expect to have a third column containing the values:

.. sourcecode:: python

  df_melt = pd.melt(df, id_vars=['Year'], var_name='Name')  #value_name='bidule' if we want to change the name of the value column.
  df_melt
  
.. figure:: Images/PandasMelt2.png
   :scale: 100 %
   :alt: Pandas Melt  
  

Pandas Cheatsheet
------------------
  
.. figure:: Cheatsheets/Python_Pandas_Cheat_Sheet_2.png
   :scale: 100 %
   :alt: map to buried treasure

   This Cheatsheet is taken from DataCamp.
   
Also have a look at the cookbook: http://pandas.pydata.org/pandas-docs/stable/cookbook.html


Assigining values to dataframe
---------------------------------------------

We have a dataframe df with column A and B, and want to assign values to a new column ln_A

.. sourcecode:: python

  df = pd.DataFrame({'A': range(1, 6), 'B': np.random.randn(5)})
  df
  
     A B

  0 1 0.846677 

  1 2 0.749287 

  2 3 -0.236784 

  3 4 0.004051 

  4 5 0.360944 

  df = df.assign(ln_A = lambda x: np.log(x.A))
  df

     A B             ln_A

  0 1 0.846677   0.00

  1 2 0.749287   0.693

  2 3 -0.236784  1.098

  3 4 0.004051   1.386

  4 5 0.360944   1.609
  
  #We can also do like this to assign to  a whole column:

  newcol = np.log(df['B'])
  df = df.assign(ln_B=newcol)
  df  
  
     A B             ln_A       ln_B

  0 1 0.846677   0.00       -0.166

  1 2 0.749287   0.693     -0.288

  2 3 -0.236784  1.098     NaN

  3 4 0.004051   1.386     -5.508

  4 5 0.360944   1.609     -1.019
  
  #Of course the assignement to a whole column is better done using the simpler command: df['ln_B2'] = np.log(df['B'])
  #But the assign command is powerful because it allows the use of lambda functions.
  #Also, user-defined functions can be applied, using assign:
  
  def function_me(row):
      if row['A'] != 2:
          rest = 5
          return rest
      else:
          rest = 2
          return rest

  df = df.assign(bidon=df.apply(function_me, axis=1))
  df  
  
     A B             ln_A       ln_B      bidon

  0 1 0.846677   0.00       -0.166   5

  1 2 0.749287   0.693     -0.288   2

  2 3 -0.236784  1.098     NaN      5

  3 4 0.004051   1.386     -5.508   5

  4 5 0.360944   1.609     -1.019   5
  

Assigning using a function (with use of the .apply method of dataframes):

.. sourcecode:: python

  #Let's say we have a dataframe with a column "credit_score", you want to encode it using your own-defined rules:
  df = pd.DataFrame(['c-1','c-3','c-2'],columns=['credit_score'])

  def set_target(row):
    if   row['credit_score'] =='c-1' :
        return 0
    elif row['credit_score'] =='c-2' :
        return 1
    elif row['credit_score'] =='c-3' :
        return 2
    else:
        return 99

  #Creating new variable called "Target"
  df = df.assign(credit_score_encoded=df.apply(set_target, axis=1))
  df
  
    credit_score credit_score_encoded
  0 c-1          0 
  1 c-3          2 
  2 c-2          1 

   
Percentiles - quantiles in Pandas
--------------------------------------------
For example, to get the 5% percentile and the 95% percentile of a dataframe (for all columns, here columns are "2015" and "2016"), we can do:

.. sourcecode:: python

  df.quantile([0.05,0.95])  
   
Saving of Pandas dataframe to LIBSVM file format and inverse
------------------

The ``LIBSVM`` file format is often used in Spark (especially <=1.6).

.. sourcecode:: python

  import pandas as pd 
  import numpy as np 
  from sklearn.datasets import dump_svmlight_file 
 
  df = pd.DataFrame() 
  df['Id'] = np.arange(10) 
  df['F1'] = np.random.rand(10,) 
  df['F2'] = np.random.rand(10,) 
  df['Target'] = np.random.randint(2,size=10) #map(lambda x: -1 if x < 0.5 else 1, np.random.rand(10,)) 
  X = df[np.setdiff1d(df.columns,['Id','Target'])] 
  y = df.Target
  dump_svmlight_file(X,y,'smvlight.dat',zero_based=True,multilabel=False) 


#Now reading a SVMLigt file into (almost) a pandas object:
from sklearn.datasets import load_svmlight_file
data = load_svmlight_file('smvlight.dat')
XX,yy = data[0],data[1]


Note: we may also load two (or more) datasets at once: load_svmlight_fileS! 
X_train, y_train, X_test, y_test = load_svmlight_files( ("/path/to/train_dataset.txt", "/path/to/test_dataset.txt") )

Pandas and memory
---------------------------------

.. sourcecode:: python

  #lists all dataframes in memory
  alldfs = [var for var in dir() if isinstance(eval(var), pd.core.frame.DataFrame)]
  print(alldfs) # df1, df2

  
Cutting a dataframe into train-test-validation sets
--------------------------------------------------------------------------

.. sourcecode:: python

  def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
      np.random.seed(seed)
      perm = np.random.permutation(df.index)
      m = len(df.index)
      train_end = int(train_percent * m)
      validate_end = int(validate_percent * m) + train_end
      train = df.iloc[perm[:train_end]]
      validate = df.iloc[perm[train_end:validate_end]]
      test = df.iloc[perm[validate_end:]]
      return train, validate, test

  np.random.seed([3,1415])
  df = pd.DataFrame(np.random.rand(10, 5), columns=list('ABCDE'))
  
  train, validate, test = train_validate_test_split(df,train_percent=0.6,validate_percent=0.2) #if validation_percent=0, then test will just be complement of train test.  
  
  
Useful plots
===========

The Swarbee plot of seaborn
--------------------------------------

.. sourcecode:: python

  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.datasets import load_iris
  import pandas as pd
  import numpy as np

  iris = load_iris()

  df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['species'])

  # Create bee swarm plot with Seaborn's default settings
  sns.swarmplot(x='species',y='petal length (cm)',data=df)
  plt.xlabel('species')
  plt.ylabel('length')
  plt.show()
  
  
.. figure:: Images/Swarbee_plot.png
   :scale: 100 %
   :alt: map to buried treasure

   This plot is taken from DataCamp.
   
   
Computation of PDF AND CDF plots (having only PDF)
--------------------------------------------------------------------   

Here I don't have the data behind, but it is roughly a dataframe with a PDF called df['fraction']. We want a multiplot with both PDF and CDF.

.. sourcecode:: python

  # This formats the plots such that they appear on separate rows
  fig, axes = plt.subplots(nrows=2, ncols=1)

  # Plot the PDF
  df.fraction.plot(ax=axes[0], kind='hist', bins=30, normed=True, range=(0,.3))
  plt.show()

  # Plot the CDF
  df.fraction.plot(ax=axes[1], kind='hist', bins=30, normed=True, cumulative=True, range=(0,.3))
  plt.show()
  
And the output is:  

.. figure:: Images/PDF_CDF.png
   :scale: 100 %
   :alt: map to buried treasure

   This plot is taken from DataCamp.  

Matplotlib: main functions
--------------------------------

fig.savefig('2016.png',dpi=600, bbox_inches='tight')   


Saving objects in Python
--------------------------------

Here are the functions for saving objects (using pickle, it is also possible and faster using cPickle, but not always available) and compressing them (using gzip):

.. sourcecode:: python

  def save(myObject, filename):
    '''
    Save an object to a compressed disk file.
    Works well with huge objects.
    '''
    #import cPickle  #(not always installed)
    #file = gzip.GzipFile(filename, 'wb')
    #cPickle.dump(myObject, file, protocol = -1)
    #file.close()

    #store the object
    #myObject = {'a':'blah','b':range(10)}
    file = gzip.open(filename,'wb') #ex: 'testPickleFile.pklz'
    pickle.dump(myObject,file)
    file.close()

  def load(filename):
    '''
    Loads a compressed object from disk
    '''
    #file = gzip.GzipFile(filename, 'rb')
    #myObject = cPickle.load(file)
    #file.close()    
    #return myObject
    
    #restore the object
    file = gzip.open(filename,'rb') #ex: 'testPickleFile.pklz'
    myObject = pickle.load(file)
    file.close() 
    return myObject

And we can use them like this:

.. sourcecode:: python

  myObject = {'a':'blah','b':range(10)}

  #store the object
  save(myObject,'bidule.pklz')

  #restore the object
  myNewObject = load('bidule.pklz')

  print( myObject )
  print( myNewObject )
   