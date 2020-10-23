===============
 Sphinx
===============

Basic Sphinx commands
===============

.. topic:: Your Topic Title

   This section is just a place to store different sphinx commands that may be useful while building the documentation. 
   
LOOK AT FOR LINKING: http://www.sphinx-doc.org/en/stable/markup/inline.html

Sphinx documentation using Azure DevOps: https://medium.com/@LydiaNemec/documenting-your-data-science-project-a-guide-to-publish-your-sphinx-code-documentation-d1afeb110696

Document python project using Sphinx: https://medium.com/@richdayandnight/a-simple-tutorial-on-how-to-document-your-python-project-using-sphinx-and-rinohtype-177c22a15b5b
	
Subject Subtitle
----------------
Subtitles are set with '-' and are required to have the same length 
of the subtitle itself, just like titles.
 
Lists can be unnumbered like:
 
 * Item Foo
 * Item Bar
 
Or automatically numbered:
 
 #. Item 1
 #. Item 2
 
Inline Markup
-------------
Words can have *emphasis in italics* or be **bold** and you can define
code samples with back quotes, like when you talk about a command: ``sudo`` 
gives you super user powers!


HERE:

 * http://matplotlib.org/sampledoc/extensions.html#ipython-sessions
 * http://matplotlib.org/sampledoc/extensions.html#using-math

The :py:func:`enumerate` function can be used for ...

.. py:function:: enumerate(sequence[, start=0])

   Return an iterator that yields tuples of an index and an item of the
   *sequence*. (And so on.)
   
intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

.. doctest::
    :options: +SKIP

    >>> whatever code

.. sourcecode:: python

    In [69]: lines = plot([1,2,3])

    In [70]: setp(lines)
      alpha: float
      animated: [True | False]
      antialiased or aa: [True | False]
      ...snip

=====  =====  ======
   Inputs     Output
------------  ------
  A      B    A or B
=====  =====  ======
False  False  False
True   False  True
=====  =====  ======

.. sourcecode:: python

	>>> from pyspark.mllib.linalg import Vectors
	>>> df = sqlContext.createDataFrame([(Vectors.dense([0.0]),), (Vectors.dense([2.0]),)], ["a"])
	>>> standardScaler = StandardScaler(inputCol="a", outputCol="scaled")
	>>> model = standardScaler.fit(df)
	>>> model.mean
	DenseVector([1.0])
	>>> model.std
	DenseVector([1.4142])
	>>> model.transform(df).collect()[1].scaled
	DenseVector([1.4142])	  
	  
.. math::

  W^{3\beta}_{\delta_1 \rho_1 \sigma_2} \approx U^{3\beta}_{\delta_1 \rho_1}


.. plot::

   import matplotlib.pyplot as plt
   import numpy as np
   x = np.random.randn(1000)
   plt.hist( x, 20)
   plt.grid()
   plt.title(r'Normal: $\mu=%.2f, \sigma=%.2f$'%(x.mean(), x.std()))
   plt.show()
   
.. sidebar:: Sidebar Title
        :subtitle: Optional Sidebar Subtitle

   Subsequent indented lines comprise
   the body of the sidebar, and are
   interpreted as body elements.
   
 
