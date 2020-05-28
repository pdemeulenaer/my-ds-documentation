===============
 Algorithms
===============

Glossary of Machine Learning: https://semanti.ca/blog/?glossary-of-machine-learning-terms

Cheatsheets on Machine Learning: https://github.com/afshinea/stanford-cs-229-machine-learning/tree/master/en

Supervised Machine Learning
============================

.. topic:: Introduction

    The objective here is to have everything useful for the projects, not to make a complete documentation of the whole package. Here I will try to document both version 1.6 and >2.0. A special enphase will be done on machine learning module ml (mllib is outdated).
    We will not review the full Pyspark documentation. For that, look at http://spark.apache.org/docs/1.6.0/programming-guide.html for version 1.6, http://spark.apache.org/docs/2.1.0/programming-guide.html for version 2.1.
    
Association rules: Apriori algorithm
-----------------------------------------------------------

https://en.wikipedia.org/wiki/Apriori_algorithm

Example 1:

Consider the following database, where each row is a transaction and each cell is an individual item of the transaction:

.. csv-table:: Some data
   :header: "Data1", "Data2", "Data3"
   :widths: 10, 10, 10

   alpha, beta, epsilon
   alpha, beta, theta
   alpha, beta, epsilon
   alpha, beta, theta

The association rules that can be determined from this database are the following:

- 100% of sets with alpha also contain beta
- 50% of sets with alpha, beta also have epsilon
- 50% of sets with alpha, beta also have theta

we can also illustrate this through a variety of examples.


Example 2:

Assume that a large supermarket tracks sales data by stock-keeping unit (SKU) for each item: each item, such as "butter" or "bread", is identified by a numerical SKU. The supermarket has a database of transactions where each transaction is a set of SKUs that were bought together.

Let the database of transactions consist of following itemsets:

-   {1,2,3,4}
-   {1,2,4}
-   {1,2}
-   {2,3,4}
-   {2,3}
-   {3,4}
-   {2,4}


We will use Apriori to determine the frequent item sets of this database. To do this, we will say that an item set is frequent if it appears in at least 3 transactions of the database: the value 3 is the support threshold.

The first step of Apriori is to count up the number of occurrences, called the support, of each member item separately. By scanning the database for the first time, we obtain the following result

.. csv-table:: Support table
   :header: "Item", "Support"
   :widths: 10,10

   {1},	3
   {2},	6
   {3},	4
   {4},	5



All the itemsets of size 1 have a support of at least 3, so they are all frequent.

The next step is to generate a list of all pairs of the frequent items.

For example, regarding the pair {1,2}: the first table of Example 2 shows items 1 and 2 appearing together in three of the itemsets; therefore, we say item {1,2} has support of three.

.. csv-table:: Support table
   :header: "Item", "Support"
   :widths: 10,10

   "{1,2}", 3
   "{1,3}", 1
   "{1,4}", 2
   "{2,3}", 3
   "{2,4}", 4
   "{3,4}", 3

The pairs {1,2}, {2,3}, {2,4}, and {3,4} all meet or exceed the minimum support of 3, so they are frequent. The pairs {1,3} and {1,4} are not. Now, because {1,3} and {1,4} are not frequent, any larger set which contains {1,3} or {1,4} cannot be frequent. In this way, we can prune sets: we will now look for frequent triples in the database, but we can already exclude all the triples that contain one of these two pairs:

.. csv-table:: Support table
   :header: "Item", "Support"
   :widths: 10,10

   "{2,3,4}", 2

in the example, there are no frequent triplets. {2,3,4} is below the minimal threshold, and the other triplets were excluded because they were super sets of pairs that were already below the threshold.

We have thus determined the frequent sets of items in the database, and illustrated how some items were not counted because one of their subsets was already known to be below the threshold. 


Implementations
-------------------------------------------

https://pypi.org/project/efficient-apriori/ (probably the fastest one, and user-friendly, but NOT pandas-friendly. Only pip, not conda)
mlxtend: https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/ (more pandas-friendly ... but I don't find the concept of confidence)
         Note: mlxtend is wider ML package (made by Sebastian Raschka!) that can do many stuff, including enemble classification (combination of different classifiers, EnsembleVoteClassifier), see https://rasbt.github.io/mlxtend/
Apriori: https://github.com/asaini/Apriori
https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/ with package apyori: https://github.com/ymoch/apyori
http://adataanalyst.com/machine-learning/apriori-algorithm-python-3-0/    

See also https://www.datacamp.com/community/tutorials/market-basket-analysis-r (in R, many different htings)
    
Collaborative filtering
-------------------------------------------

The main idea behind collaborative filtering is to adopt for a subject same item as for similar other subjects: 
    
.. figure:: Images/Collaborative_filtering1.png
   :scale: 100 %
   :alt: Collaborative_filtering1
   
       
See also https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0 
    
Naive Bayes
-------------------------------------------

First, remember the Bayes theorem:

.. figure:: Images/Bayes_theorem.png
   :scale: 100 %
   :alt: Bayes theorem
    
Decision Tree
-------------------------------------------

https://medium.com/meta-design-ideas/decision-tree-a-light-intro-to-theory-math-code-10dbb3472ec4   

Gini impurity: https://victorzhou.com/blog/gini-impurity/

Decision tree/RF: https://victorzhou.com/blog/intro-to-random-forests/ 
    
.. _RF_algo-label:    
Random Forest
-------------------------------------------

What is Random Forest?
A random forest is an ensemble of decision trees that will output a prediction value. An ensemble model combines the results from different models. A Random Forest is combination of classification and regression. The result from an ensemble model is usually better than the result from one of the individual models. In Random Forest, each decision tree is constructed by using a random subset of the training data that has predictors with known response. After you have trained your forest, you can then pass each test row through it, in order to output a prediction. The goal is to predict the response when it’s unknown. The response can be categorical(classification) or continuous (regression). In a decision tree, an input is entered at the top and as it traverses down the tree the data gets bucketed into smaller and smaller sets. The random forest takes the notion of decision trees to the next level by combining trees. Thus, in ensemble terms, the trees are weak learners and the random forest is a strong learner.
Pros and Cons of choosing Random Forest?
Random forest combines trees and hence incorporates most of the advantages of trees like handling missing values in variable, suiting for both classification and regression, handling highly non-linear interactions and classification boundaries. In addition, Random Forest gives built-in estimates of accuracy, gives automatic variable selection. variable importance, handles wide data – data with more predictors than observations and works well off the shelf – needs only little tuning, can get results very quickly. The runtimes are quite fast, and they are able to deal with unbalanced and missing data.
Random Forest weaknesses are that when used for regression they cannot predict beyond the range in the training data, and that they may over-fit data sets that are particularly noisy.



For an implementation using Scikit-learn, see :ref:`Random Forest <RF_sklearn-label>` 

.. figure:: Images/RF_algo.png
   :scale: 100 %
   :alt: Random Forest scheme
   
   An old slide...

Here are a few posts on the RF algorithm:

- Simple intro: Decision tree/RF: https://victorzhou.com/blog/intro-to-random-forests/

- Simple intro: https://towardsdatascience.com/the-random-forest-algorithm-d457d499ffcd

- Good intro: https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-5-ensembles-of-algorithms-and-random-forest-8e05246cbba7

- Quick intro: https://medium.com/@Synced/how-random-forest-algorithm-works-in-machine-learning-3c0fe15b6674

- Intro : https://medium.com/@williamkoehrsen/random-forest-simple-explanation-377895a60d2d

- Guide to Decision tree and RF: https://towardsdatascience.com/enchanted-random-forest-b08d418cb411

- Example of application: https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

- Hyperparameter tuning: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

- missing values in RF: https://medium.com/airbnb-engineering/overcoming-missing-values-in-a-random-forest-classifier-7b1fc1fc03ba

- Example of application (churn): https://blog.slavv.com/identifying-churn-drivers-with-random-forests-65bad0193e6b

- Feature importance 1: https://medium.com/the-artificial-impostor/feature-importance-measures-for-tree-models-part-i-47f187c1a2c3

- Feature importance 2: https://becominghuman.ai/feature-importance-measures-for-tree-models-part-ii-20c9ff4329b

- Interpretation of RF: http://blog.datadive.net/interpreting-random-forests/

- Interpretation of RF: https://towardsdatascience.com/intuitive-interpretation-of-random-forest-2238687cae45

- Bagging vs Boosting: https://towardsdatascience.com/how-to-develop-a-robust-algorithm-c38e08f32201

- Categorical variables in tree methods: https://medium.com/data-design/visiting-categorical-features-and-encoding-in-decision-trees-53400fa65931

Random forest from scratch:

- https://machinelearningmastery.com/implement-random-forest-scratch-python/ (for classification)

- https://towardsdatascience.com/random-forests-and-decision-trees-from-scratch-in-python-3e4fa5ae4249 (maybe even more complete, but for regression only, could be tuned for classification)


Gradient Boosting:, XGBoost
-------------------------------------------

Gradient Boosting: https://www.youtube.com/watch?v=sRktKszFmSk&t=370s

XGboost: https://medium.com/@pushkarmandot/how-exactly-xgboost-works-a320d9b8aeef

Differences between XGBoost over Gradient Boosting. Also gives great intro to all parameters:

- https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ 

Video XGboost: https://www.youtube.com/watch?v=Vly8xGnNiWs

- interpretable machine learning: XGBoost: https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27


Isolation Forest (IF)
-------------------------------------------

Isolation Forest is a tree based method for anomaly detection.

Taken from https://www.youtube.com/watch?v=5p8B2Ikcw-k :

.. figure:: Images/Isolation_forest_1.PNG
   :scale: 100 %
   :alt: Isolation_forest_1.PNG
   
.. figure:: Images/Isolation_forest_2.PNG
   :scale: 100 %
   :alt: Isolation_forest_2.PNG

.. figure:: Images/Isolation_forest_3.PNG
   :scale: 100 %
   :alt: Isolation_forest_3.PNG  
   
See also https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e    

Algorithm interpretability
-------------------------------------------

- Tree-specific: Interpretation of RF: http://blog.datadive.net/interpreting-random-forests/

- Tree-specific: Interpretation of RF: https://towardsdatascience.com/intuitive-interpretation-of-random-forest-2238687cae45

- Tree-specific (treeinterpreter for classification): http://engineering.pivotal.io/post/interpreting-decision-trees-and-random-forests/ 

- XGBoost-specific: interpretable machine learning: XGBoost: https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
 
 
Performance metrics (for SL)
-------------------------------------------

Excellent post on different metrics usually used: https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c

Discussion of ROC, GAIN, LIFT and other important quantities: http://www.saedsayad.com/model_evaluation_c.htm , http://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html

**Gain chart**: Gain or lift is a measure of the effectiveness of a classification model calculated as the ratio between the results obtained with and without the model. Gain and lift charts are visual aids for evaluating performance of classification models. However, in contrast to the confusion matrix that evaluates models on the whole population gain or lift chart evaluates model performance in a portion of the population. (Here the wizard curve is the perfect model! Different than in ROC)

In Y-axis we have the % of positive response, in X-axis we have the % of customers contacted. The principle in the blue curve (the model) is to start by the customers with best scores/rank, then as the curves evolves to the right we add worse ones.

.. figure:: Images/Gain_chart.png
   :scale: 100 %
   :alt: A gain chart 
   
So we see that if we pick the 10% best customers, we jump very high, much more than when we add from 30% to 40% for example.    

**Lift chart**: The lift chart shows how much more likely we are to receive positive responses than if we contact a random sample of customers. For example, by contacting only 10% of customers based on the predictive model we will reach 3 times as many respondents, as if we use no model.
 
The horizontal 0-line is the baseline, the random case.  
 
.. figure:: Images/Lift.png
   :scale: 100 %
   :alt: A lift chart 
   
A variant of the lift is to put in Y-axis the probability derived for the customers (for example the probability to take a loan) vs the number of customers (of course, customers are ranked by the proba, as in normal lift):

.. figure:: Images/Lift_variant.png
   :scale: 100 %
   :alt: A lift variant 
   
Simple on lift: https://www.kdnuggets.com/2016/03/lift-analysis-data-scientist-secret-weapon.html   
   
**The Z-score**: Simply put, a z-score is the number of standard deviations from the mean a data point is. But more technically it’s a measure of how many standard deviations below or above the population mean a raw score is. A z-score is also known as a standard score and it can be placed on a normal distribution curve. Z-scores range from -3 standard deviations (which would fall to the far left of the normal distribution curve) up to +3 standard deviations (which would fall to the far right of the normal distribution curve). In order to use a z-score, you need to know the mean μ and also the population standard deviation σ.

Z-scores are a way to compare results from a test to a “normal” population. Results from tests or surveys have thousands of possible results and units. However, those results can often seem meaningless. For example, knowing that someone’s weight is 150 pounds might be good information, but if you want to compare it to the “average” person’s weight, looking at a vast table of data can be overwhelming (especially if some weights are recorded in kilograms). A z-score can tell you where that person’s weight is compared to the average population’s mean weight. 
   
The Z Score Formula: For One Sample:

The basic z score formula for a sample is:

z = (x – μ) / σ

For example, let’s say you have a test score of 190. The test has a mean (μ) of 150 and a standard deviation (σ) of 25. Assuming a normal distribution, your z score would be:

z = (x – μ) / σ = 190 – 150 / 25 = 1.6.

The z score tells you how many standard deviations from the mean your score is. In this example, your score is 1.6 standard deviations above the mean.   

The Z score formula: for multiple samples (i.e. multiple data points):

When you have multiple samples and want to describe the standard deviation of those sample means (the standard error), you would use this z score formula:

z = (x – μ) / (σ / √n)

This z-score will tell you how many standard errors there are between the sample mean and the population mean.

Sample problem: In general, the mean height of women is 65″ with a standard deviation of 3.5″. What is the probability of finding a random sample of 50 women with a mean height of 70″, assuming the heights are normally distributed? 
 
z = (x – μ) / (σ / √n) = (70 – 65) / (3.5/√50) = 5 / 0.495 = 10.1

The key here is that we’re dealing with a sampling distribution of means, so we know we have to include the standard error in the formula. We also know that 99% of values fall within 3 standard deviations from the mean in a normal probability distribution (see 68 95 99.7 rule). Therefore, there’s less than 1% probability that any sample of women will have a mean height of 70″. 
 
See http://www.statisticshowto.com/probability-and-statistics/z-score/ , https://en.wikipedia.org/wiki/Standard_score for more

Z-score normalization is another term for standardization, or scaling... You just remove the mean and divide by the standard deviation:

.. figure:: Images/Standardization.png
   :scale: 100 %
   :alt: Standardization
   
In scikit-learn:

.. sourcecode:: python

  from sklearn.preprocessing import StandardScaler

  scaler = StandardScaler()

  # Fit only to the training data
  scaler.fit(X_train)

  # Now apply the transformations to the data:
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)   
 
Unsupervised Machine Learning - Clustering
====================================================
 

Deep Learning
============================



Machine Learning Ranking
=============================================	

https://mlexplained.com/2019/05/27/learning-to-rank-explained-with-code/ : 

In supervised machine learning, the most common tasks are classification and regression.  Though these two tasks will get you fairly far, sometimes your problem cannot be formulated in this way. For example, suppose you wanted to build a newsfeed or a recommendation system. In these cases, you don’t just want to know the probability of a user clicking an article or buying an item; you want to be able to prioritize and order the articles/items to maximize your chances of getting a click or purchase. 

Let’s take the example of ranking items for the newsfeed of a user. If all we cared about was clicks, then we could just train a model to predict whether a user will click on each item and rank them according to the click probability. However, we might care about more than just clicks; for instance, if the user clicks on an article but does not finish reading it, it might not be that interesting to them and we won’t want to recommend similar articles to them in the future. 

This is where learning to rank comes in. Instead of using some proxy measure (e.g. the probability of a user clicking on an item), we directly train the model to rank the items.

https://medium.com/@nikhilbd/intuitive-explanation-of-learning-to-rank-and-ranknet-lambdarank-and-lambdamart-fe1e17fac418 : 

Learning to Rank (LTR) is a class of techniques that apply supervised machine learning (ML) to solve ranking problems. The main difference between LTR and traditional supervised ML is this:

* Traditional ML solves a prediction problem (classification or regression) on a single instance at a time. E.g. if you are doing spam detection on email, you will look at all the features associated with that email and classify it as spam or not. The aim of traditional ML is to come up with a class (spam or no-spam) or a single numerical score for that instance.

* LTR solves a ranking problem on a list of items. The aim of LTR is to come up with optimal ordering of those items. As such, LTR doesn’t care much about the exact score that each item gets, but cares more about the relative ordering among all the items.

The most common application of LTR is search engine ranking, but it’s useful anywhere you need to produce a ranked list of items.

There are many ranking algo. Here is a list: https://en.wikipedia.org/wiki/Learning_to_rank

In RankNet, LambdaRank and LambdaMART techniques, ranking is transformed into a pairwise classification or regression problem. That means you look at pairs of items at a time, come up with the optimal ordering for that pair of items, and then use it to come up with the final ranking for all the results.

RankNet
---------------------------

The cost function for RankNet aims to minimize the number of inversions in ranking. Here an inversion means an incorrect order among a pair of results, i.e. when we rank a lower rated result above a higher rated result in a ranked list. RankNet optimizes the cost function using Stochastic Gradient Descent.

In RankNet the loss is agnostic to the actual ranking of the item. In other words, the loss is the same for any pair of items i, j regardless of whether i and j are ranked in 5th and 6th place or if they are in 1st and 200th place. 


LambdaRank
---------------------------

LambdaRank addresses the problem of RankNet agnosticity. 

During RankNet training procedure, you don’t need the costs, only need the gradients (λ) of the cost with respect to the model score. You can think of these gradients as little arrows attached to each document in the ranked list, indicating the direction we’d like those documents to move.

Further creaters found that scaling the gradients by the change in NDCG (discounted cumulative gain https://en.wikipedia.org/wiki/Discounted_cumulative_gain) found by swapping each pair of documents gave good results. The core idea of LambdaRank is to use this new cost function for training a RankNet. On experimental datasets, this shows both speed and accuracy improvements over the original RankNet.

LambdaMart
---------------------------

LambdaMART is the boosted tree version of LambdaRank. 