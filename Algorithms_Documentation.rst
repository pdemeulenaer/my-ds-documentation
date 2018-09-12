===============
 Algorithms
===============

Supervised Machine Learning
============================

.. topic:: Introduction

    The objective here is to have everything useful for the projects, not to make a complete documentation of the whole package. Here I will try to document both version 1.6 and >2.0. A special enphase will be done on machine learning module ml (mllib is outdated).
    We will not review the full Pyspark documentation. For that, look at http://spark.apache.org/docs/1.6.0/programming-guide.html for version 1.6, http://spark.apache.org/docs/2.1.0/programming-guide.html for version 2.1.
    
    
    
    
Decision Tree
----------------------

https://medium.com/meta-design-ideas/decision-tree-a-light-intro-to-theory-math-code-10dbb3472ec4    
    
.. _RF_algo-label:    
Random Forest
----------------------

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

XGBoost
--------------



- interpretable machine learning: XGBoost: https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27


Algorithm interpretability
---------------------------------

- Tree-specific: Interpretation of RF: http://blog.datadive.net/interpreting-random-forests/

- Tree-specific: Interpretation of RF: https://towardsdatascience.com/intuitive-interpretation-of-random-forest-2238687cae45

- Tree-specific (treeinterpreter for classification): http://engineering.pivotal.io/post/interpreting-decision-trees-and-random-forests/ 

- XGBoost-specific: interpretable machine learning: XGBoost: https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
 
Unsupervised Machine Learning - Clustering
============================
 

Deep Learning
============================