===============================================
 Python, Jupyter and Pandas
===============================================

Exercises on Python (Pandas, Numpy,...)
===============================================

- https://www.w3resource.com/python-exercises/

- https://pynative.com/python-exercises-with-solutions/

- https://edabit.com/challenge/xbZR26rHMNo32yz35

and many more ...

Basic Jupyter infos
===============================================

When you launch a Jupyter notebook, you can adjust its width size using the following command at the beginning:
   
.. sourcecode:: python

   from IPython.core.display import display, HTML
   display(HTML("<style>.container { width:100% !important; }</style>"))
   
   #or better:
   
   display(HTML("<style>.container { width:95% !important; font-size:10px; font-weight:bold;}</style>"))
   
Loading package from a given (maybe different) directory:

.. sourcecode:: python

   import sys
   sys.path.append('/home/user/folder/python-package') 
   
Useful internal address: https://sb-hdp-e4.fspa.myntet.se:8400/    

Useful Jupyter notebook tricks: https://www.dataquest.io/blog/advanced-jupyter-notebooks-tutorial/
   
Getting Notebooks work on server and access them using ssh
=================================================================

How to keep jupyter notebook (or pyspar3Jupyter) active through ssh:
Go to server with ssh (using putty)
type: nohup pyspar3Jupyter > save.txt &  (by this the save.txt contains the address of the notebook)
type: jobs -l get pid number, this will be useful when you want to kill your pyspark session.
ps -aux | grep b_number (this gets the pid if the ssh has already been shut down
kill 5442 (if pid=5442)

Python environments: pip & conda
===============================================

Very good intro: https://towardsdatascience.com/devops-for-data-science-making-your-python-project-reproducible-f55646e110fa

pip: Installing through proxy
-----------------------------------------------

.. sourcecode:: python

  pip install --proxy=https://p998phd:p998phd@proxyvip-se.sbcore.net:8080 --trusted-host pypi.python.org -U PACKAGE_NAME
  
How to check list of python packages installed through pip: https://pip.pypa.io/en/stable/reference/pip_list/

.. sourcecode:: python

  # linux
  python -m pip list
  
  # windows
  py -m pip list #(although i think "python -m pip list" works too)
  
  # or simply
  pip list
  
pip, venv & setup.py: create a simple virtual environment for model development
-----------------------------------------------

See https://madewithml.com/courses/mlops/packaging/  

.. sourcecode:: python

  python3 -m venv venv
  source venv/bin/activate
  python -m pip install --upgrade pip setuptools wheel
  pip install -e .

Let's unpack what's happening here:

- Creating a vitual environment named venv

- Activating our virtual environment. Type deactivate to exit out of the virtual environment.

- Upgrading required packages so we download the latest package wheels.

- Install (our packages) from (our) setup.py (-e, --editable installs a project in develop mode)

Example of a setup.py: https://github.com/GokuMohandas/MLOps/blob/main/setup.py

pip: How does `pip install -e .` work? 
-----------------------------------------------

This is taken from: https://www.reddit.com/r/learnpython/comments/ayx7za/how_does_pip_install_e_work_is_there_a_specific/

pip install is a command that takes a package and install it inside the **site-packages** folder of your Python installation (be it your main/system wide Python installation, or one inside a virtual environment).

Normally, when you do this by simply writing a package name, like with pip install requests, pip looks for the package in the Python Package Index, or PyPI, which is a website. However, pip can also look for packages which are in other places (including inside your computer right now), and properly copy them to your site-packages folder. This is useful in a few specific cases:

If you download the source code directly, i.e. from a github repository or another similar platform, you can use pip install <path-to-downloaded-code> to install this package without having to resort to PyPI. Now granted this is not very useful, since most people who create good packages and share them in github will also add them to PyPI anyways, but the option is there.

Install a specific version of a package which is not directly available through PyPI, but may be reachable through github and others. Think about a unstable/dev build of a project: the devs don't want to make it available through PyPI to keep unaware users from downloading broken code, but you can pip install <link-to-broken-package> to install it as a Python package anyways, at your own risk.

Install your own code as a package in your own machine. This basically copies your code over to the site-packages folder and treats it like any other package you've downloaded. Useful for testing and developing, since this makes your package behave like it would in any other system once you release it to the world. This is where pip install . comes into play: the dot is an actual argument, replacing the directory you're currently in. Most of the time you'll pip install your own packages using a terminal already inside of the project's folder, which is why you see the dot as sort of a default argument. Also keep in mind that you will some specific files in order for your package to be "installable", like a setup.py and possibly some __init.py__.

Last thing to note is that pip install will install the current package as it is right now. If you pip install a package you're developing and add some new files to it afterwards, these changes will not be reflected on the actual package installed beforehand. To avoid having to pip install the package again and again after each change, you can pass the *-e* flag to make an editable install; in this case, changes to your files inside the project folder will automatically reflect in changes on your installed package in the site-packages folder.

.. sourcecode:: python

  pip install -e .
  
Create a package out of your code
-----------------------------------------------

See this excellent post: https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/ (todo: extract example from it)
   
Conda environments
-----------------------------------------------

Once installed, in linux, the .bashrc file will contain the block:

.. sourcecode:: python

  # >>> conda initialize >>>
  # !! Contents within this block are managed by 'conda init' !!
  __conda_setup="$('/home/philippe/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      eval "$__conda_setup"
  else
      if [ -f "/home/philippe/anaconda3/etc/profile.d/conda.sh" ]; then
          . "/home/philippe/anaconda3/etc/profile.d/conda.sh"
      else
          export PATH="/home/philippe/anaconda3/bin:$PATH"
      fi
  fi
  unset __conda_setup
  # <<< conda initialize <<<

That block allows the initialization of the conda env, after reloading .bashrc file. This there is NO NEED to add a line such as export PATH="/home/philippe/anaconda3/bin:/home/philippe/anaconda3/condabin:$PATH" or similar. See https://github.com/conda/conda/issues/7980

check the environments:

.. sourcecode:: python

  conda env list
  
  # or
  conda info --envs

There should be a base, and others, if they were created.

Then, to activate a different environment:

.. sourcecode:: python

  source activate env_name  #(source in Unix, without on Windows)
  
  #or 
  conda activate env_name
  
How to create environments: 

To create a new environment with some packages: 

1. From the command line (see also the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_): 
.. sourcecode:: python

  conda create -n env_name --yes --quiet python=3.5 numpy scipy scikit-learn statsmodels
  
2. From an environment.yml file (see also the `conda documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`_):

.. sourcecode:: python

  conda env create -f environment.yml
  
To prepare such a environment.yml file, see the dedicated conda `documentation page <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-file-manually>`_. Basically:

.. sourcecode:: python

  name: stats # the name of the env
  dependencies:
    - numpy
    - pandas
  
Or more complex:

.. sourcecode:: python

  name: stats2
  channels:
    - javascript
  dependencies:
    - python=3.6   # or 2.7
    - bokeh=0.9.2
    - numpy=1.9.*
    - nodejs=0.10.*
    - flask
    - pip:
      - Flask-Testing

Pyenv & pipenv python environments
---------------------------------------

Comparison of different python environment managment tools: venv, virtualenv, pyenv, pipenv, conda (and docker): https://www.pluralsight.com/tech-blog/managing-python-environments/

**Pyenv**: Opposed to Pipenv, Pyenv is a tool for managing *multiple* python installations. 

Installation of pyenv and using pyenv to install different python versions: https://www.liquidweb.com/kb/how-to-install-pyenv-on-ubuntu-18-04/ 
See also https://menziess.github.io/howto/manage/python-versions/ for installation/uninstallation. 

- Install from git: git clone https://github.com/pyenv/pyenv.git ~/.pyenv

- config the environment:

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
bash

note: in Windows you can change a environment variable by set PATH=%PATH%;C:\Users\some\path\

- Look at available python versions: pyenv install --list

- install a specific version: pyenv install 3.8.3

- check the installed python versions: pyenv versions

Ex:
* system (set by /root/.pyenv/version)
  3.8.3

- Now easy to switch between different installed versions: 

pyenv global 3.8.3

Note: as several posts noted (for example `here <https://stackoverflow.com/questions/38701203/error-missing-the-openssl-lib-while-trying-to-install-python>`_ and `here <https://stackoverflow.com/questions/12806122/missing-python-bz2-module>`_), the python installation sometimes lacks a few things. Remedy with this:

sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

**Pipenv**: Pipenv is capable of using pyenv in the background to create and activate virtual environments that require different python versions.
Installation of pipenv: https://menziess.github.io/howto/manage/virtual-environments/#3-creating-a-virtual-environment 

Note that some people recommend to install pipenv for user only (see here, step 1 only: https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-programming-environment-on-an-ubuntu-20-04-server). A user installation prevents breaking any system-wide packages. If pipenv isn’t available in your shell after installation, you’ll need to add the user base’s binary directory to your PATH.

pip3 install --user pipenv

and be sure add ~/.local/bin to the head of your PATH environment variable: export PATH=$PATH:/home/[your_user]/.local/bin/

To upgrade pipenv at any time:

pip3 install --user --upgrade pipenv

When pipenv installed and configured, we can create a new pipenv project in a project directory with (https://www.pluralsight.com/tech-blog/managing-python-environments/)

$ pipenv --python $PYTHON_VERSION

which will initialize the project using the specified Python version (if pyenv is installed, it can even install Python versions on-demand). To start with, this creates:

- a Pipfile config file at the project home specifying Python version, sources, and any installed packages

- a fresh virtual environment housed in the pipenv working directory

We no longer have to manage installs with pip and virtual environments separately - pipenv takes care of both! To install a package, simply running

$ pipenv install $PACKAGE_NAME

will both install the package into the virtual environment, and write the package as a dependency into the Pipfile. This Pipfile is then all we need to rebuild the project elsewhere, rather than the requirements.txt used by other managers - simply running pipenv install on a directory with a Pipfile will recreate the environment. To activate the environment,

$ pipenv shell

Pipenv exhaustively builds out the dependency graph, flagging any issues and generating a validated Pipfile.lock for fully specifying every dependency in the project. We can trigger this manually for the requirements in our Pipfile with

$ pipenv lock

To deactivate your virtual environment, run:

$ deactivate

Remove the virtual environment:

$ pipenv --rm

Note switching (https://menziess.github.io/howto/manage/virtual-environments/#5-switching-to-pipenv): If you are already using another virtual environment tool, switching is quite easy. If you run pipenv install, it automatically detects the requirements.txt file:

requirements.txt found, instead of Pipfile! Converting…

Or you can explicitly pass the requirement.txt file as an argument, which may be useful if you have put development dependencies in a separate file:

$ pipenv install -r dev-requirements.txt --dev

And if you want to switch back to using requirement.txt files, you can run:

$ pipenv lock -r > requirements.txt
$ pipenv lock -r -d > dev-requirements.txt

See for this: https://pipenv.kennethreitz.org/en/latest/advanced/#generating-a-requirements-txt

Note (seehttps://github.com/pypa/pipenv/issues/3150): in Azure DevOps I have been using such a line:

$ pipenv install -d --system --deploy --ignore-pipfile

pipenv install --ignore-pipfile is nearly equivalent to pipenv sync, but pipenv sync will never attempt to re-lock your dependencies as it is considered an atomic operation. pipenv install by default does attempt to re-lock unless using the --deploy flag.


More infos: 

- https://pypi.org/project/pipenv/

- https://pipenv-fork.readthedocs.io/en/latest/basics.html




Python linting, static code analysis
=======================================

Pylint
---------------------------------------

How to create a configuration file .pylintrc in your project: pylint --generate-rcfile > .pylintrc

Also how to format the report (could we put the format in the .pylintrc?), here is an example (https://community.sonarsource.com/t/pylint-results-not-reported-uploaded-by-scanner/4208):

.. sourcecode:: python

  #Let's have a function:  
  $ cat sample.py 
  def function1(rrrr_mm_dd):
      print "We do not use any argument"
  
  $ pylint sample.py -r n --msg-template="{path}:{line}: [{msg_id}({symbol}), {obj}] {msg}" | tee pylint.txt
  No config file found, using default configuration
  ************* Module sample
  sample.py:1: [C0111(missing-docstring), ] Missing module docstring
  sample.py:1: [C0111(missing-docstring), function1] Missing function docstring

Flake8
---------------------------------------


Visual Studio Code set-up
=======================================

Taken from https://menziess.github.io/howto/enhance/your-python-vscode-workflow/

The default values of the settings.json file can be seen in https://code.visualstudio.com/docs/getstarted/settings

In settings.json (ctrl-shift-P):

.. sourcecode:: python

  {
    "python.pythonPath": ".venv/bin/python"
  }
  
For testing and linting, we can use install in the local (project) environment

pipenv install -d mypy autopep8 \
  flake8 pytest bandit pydocstyle
  
The settings of vscode can be overridden by workspace settings per project. In settings.json:

.. sourcecode:: python

  {
    "python.autoComplete.addBrackets": true,
    "python.formatting.provider": "autopep8",
    "python.jediEnabled": false,
    "python.linting.mypyEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.pydocstyleEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
      "tests"
    ]
  }

Some of these frameworks produce temporary folders, which can clutter your file explorer, and slow down file indexing. You can disable indexing for these files by passing a glob pattern to the files.watcherExclude field:

.. sourcecode:: python

  {
    "files.watcherExclude": {
      "**/build/**": true,
      "**/dist/**": true,
      "**/.ipynb_checkpoints/**": true,
      "**/*.egg-info/**": true,
      "**/.pytest_cache/**": true,
      "**/__pycache__/**": true,
      "**/.mypy_cache/**": true,
      "**/.venv/**": true
    },
    "files.exclude": {
      "**/.pytest_cache/**": true,
      "**/.mypy_cache/**": true,
      "**/__pycache__/**": true,
      "**/*.egg-info/**": true
    }
  }


Python basic info
=======================================

Formats for printing
---------------------------------------

See https://www.geeksforgeeks.org/python-output-formatting/

The general syntax for a format placeholder is:  %[flags][width][.precision]type 

.. sourcecode:: python

  # print integer and float value 
  print("Geeks : % 2d, Portal : % 5.2f" %(1, 05.333))  
  
  # print exponential value 
  print("% 10.3E"% (356.08977)) 
  
Using format():

.. sourcecode:: python

  # using format() method and refering a position of the object 
  print('{0} and {1}'.format('Geeks', 'Portal'))   
  
  # combining positional and keyword arguments 
  print('Number one portal is {0}, {1}, and {other}.'
       .format('Geeks', 'For', other ='Geeks'))   
       
  # using format() method with number  
  print("Geeks :{0:2d}, Portal :{1:8.2f}". 
        format(12, 00.546))        

How many cores in the edge node?
-----------------------------------------------

.. sourcecode:: python

  import multiprocessing
  print(multiprocessing.cpu_count())
  
56  
  
Similar to linux command nproc --all (or grep -c ^processor /proc/cpuinfo)

The command grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}' 
gives 14. Means these are true cores, 56 are the number of threads.

Basic dates in Python
-----------------------------------------------

How to add/substract some time to dates in python?

.. sourcecode:: python

  import datetime
  from dateutil.relativedelta import relativedelta
 
  sub_days = datetime.datetime.today() + relativedelta(days=-6)
  sub_months = datetime.datetime.today() + relativedelta(months=-6)
  sub_years = datetime.datetime.today() + relativedelta(years=-6)
 
  sub_hours = datetime.datetime.today() + relativedelta(hours=-6)
  sub_mins = datetime.datetime.today() + relativedelta(minutes=-6)
  sub_seconds = datetime.datetime.today() + relativedelta(seconds=-6)
 
  print("Current Date Time:", datetime.datetime.today())
  print("Subtract 6 days:", sub_days)
  print("Subtract 6 months:", sub_months)
  print("Subtract 6 years:", sub_years)
  print("Subtract 6 hours:", sub_hours)
  print("Subtract 6 mins:", sub_mins)
  print("Subtract 6 seconds:", sub_seconds)
  
How to convert dates from datetime to string:

.. sourcecode:: python

  from datetime import datetime
  datetime.today().strftime("%Y-%m-%d")  
  
How to get first day of month:

.. sourcecode:: python

  from datetime import datetime
  datetime.today().replace(day=1)  
  
  
Docstrings in functions and classes:
-----------------------------------------------

Docstring is a great tool for code understanding, especially when it is not written by you...or when you wrote it long time ago! The idea is just to supply each function and class with a consistent explanation of its aim (why is it needed for, what it does), the description of the different input and output objects. It is a good habit to use them.

There is a PEP on docstrings (PEP257): https://www.python.org/dev/peps/pep-0257/

Here some few examples, taken/adapted from https://www.geeksforgeeks.org/python-docstrings/:  

One line docstrings (for very obvious functions):
	
.. sourcecode:: python 
    
  def power(a, b):
      """Returns arg1 raised to power arg2."""
    
      return a**b
   
  # To access the function description, for example from your notebook, you can use:
  print(power.__doc__)
 
  # Or similarly:
  help(power)
  
Multi line docstrings:
	
.. sourcecode:: python 
    
  def my_function(arg1,arg2):
      """
      Summary line.
   
      Extended description of function.
   
      Parameters:
      arg1 (int): Description of arg1
      arg2 (int): Description of arg2
   
      Returns:
      result (int): Description of return value  
      """
     
      result = arg1+arg2 
 
      return result
   
  print(my_function.__doc__)
  
Class docstrings:
	
.. sourcecode:: python 
    
  class ComplexNumber:
    """
    This is a class for mathematical operations on complex numbers.
       
    Attributes:
        real (int): The real part of complex number.
        imag (int): The imaginary part of complex number.
    """
   
    def __init__(self, real, imag):
        """
        The constructor for ComplexNumber class.
   
        Parameters:
           real (int): The real part of complex number.
           imag (int): The imaginary part of complex number.   
        """
   
    def add(self, num):
        """
        The function to add two Complex Numbers.
   
        Parameters:
            num (ComplexNumber): The complex number to be added.
           
        Returns:
            ComplexNumber: A complex number which contains the sum.
        """
   
        re = self.real + num.real
        im = self.imag + num.imag
   
        return ComplexNumber(re, im)
   
  help(ComplexNumber)  # to access Class docstring
  help(ComplexNumber.add)  # to access method's docstring  
  
PEP - Code Refactoring - Autopep8
-----------------------------------------------

See  https://pypi.org/project/autopep8/

.. sourcecode:: python 

  autopep8 --in-place --aggressive --aggressive code.py

If done with Visual Studio Code, the settings should be adapted. Type 'Ctrl + ,' and this will open the options pallet. Here type in proxy and this will show all the proxy settings. Click on the settings.json file and update the contents so they look like the following:

.. sourcecode:: python 

  {
    "http.proxy": "http://{your_pid_here}:{your_pid_here}@proxyvip-se.sbcore.net:8080",
    "http.proxyStrictSSL": false,
    "python.linting.enabled": true,
    "python.linting.pep8Args": [
        "--ignore=E501,E265"
    ],
    "python.linting.pep8Enabled": true,
    "python.linting.pylintEnabled": true,
    "python.pythonPath": "C:\\Anaconda3\\python.exe",
    "window.zoomLevel": 0,
    "python.dataScience.jupyterServerURI": "http://sb-hdpdev-e3.fspa.myntet.se:4191/?token=test"
  }


Unit tests in Python: pytest
-----------------------------------------------------------

Good links: 

- https://realpython.com/pytest-python-testing/ 

- https://menziess.github.io/howto/test/python-code/

How to discover the unit tests (pytest): https://docs.pytest.org/en/stable/goodpractices.html#test-discovery

Tests outside application code: Putting tests into an extra directory outside your actual application code might be useful if you have many functional tests or for other reasons want to keep tests separate from actual application code (often a good idea):

.. sourcecode:: python

  setup.py
  mypkg/
      __init__.py
      app.py
      view.py
  tests/
      test_app.py
      test_view.py
      ...

About fixtures (from the link above):

Imagine you’re writing a function, format_data_for_display(), to process the data returned by an API endpoint. The data represents a list of people, each with a given name, family name, and job title. The function should output a list of strings that include each person’s full name (their given_name followed by their family_name), a colon, and their title. To test this, you might write the following code:

.. sourcecode:: python

  def format_data_for_display(people):
      ...  # Implement this!
  
  def test_format_data_for_display():
      people = [
          {
              "given_name": "Alfonsa",
              "family_name": "Ruiz",
              "title": "Senior Software Engineer",
          },
          {
              "given_name": "Sayid",
              "family_name": "Khan",
              "title": "Project Manager",
          },
      ]
  
      assert format_data_for_display(people) == [
          "Alfonsa Ruiz: Senior Software Engineer",
          "Sayid Khan: Project Manager",
      ]

Now suppose you need to write another function to transform the data into comma-separated values for use in Excel. The test would look awfully similar:

.. sourcecode:: python

  def format_data_for_excel(people):
      ... # Implement this!
  
  def test_format_data_for_excel():
      people = [
          {
              "given_name": "Alfonsa",
              "family_name": "Ruiz",
              "title": "Senior Software Engineer",
          },
          {
              "given_name": "Sayid",
              "family_name": "Khan",
              "title": "Project Manager",
          },
      ]
  
      assert format_data_for_excel(people) == """given,family,title
  Alfonsa,Ruiz,Senior Software Engineer
  Sayid,Khan,Project Manager
  """

If you find yourself writing several tests that all make use of the same underlying test data (or python object), then a fixture may be in your future. You can pull the repeated data into a single function decorated with @pytest.fixture to indicate that the function is a pytest fixture:

.. sourcecode:: python

  import pytest
  
  @pytest.fixture
  def example_people_data():
      return [
          {
              "given_name": "Alfonsa",
              "family_name": "Ruiz",
              "title": "Senior Software Engineer",
          },
          {
              "given_name": "Sayid",
              "family_name": "Khan",
              "title": "Project Manager",
          },
      ]
    
You can use the fixture by adding it as an argument to your tests. Its value will be the return value of the fixture function:

.. sourcecode:: python

  def test_format_data_for_display(example_people_data):
      assert format_data_for_display(example_people_data) == [
          "Alfonsa Ruiz: Senior Software Engineer",
          "Sayid Khan: Project Manager",
      ]
  
  def test_format_data_for_excel(example_people_data):
      assert format_data_for_excel(example_people_data) == """given,family,title
  Alfonsa,Ruiz,Senior Software Engineer
  Sayid,Khan,Project Manager
  """

Each test is now notably shorter but still has a clear path back to the data it depends on. Be sure to name your fixture something specific. That way, you can quickly determine if you want to use it when writing new tests in the future! 

Another simple fixture example (from https://menziess.github.io/howto/test/python-code/):

.. sourcecode:: python

  # Let's have some function
    def say_hello_to(name='World'):
        return f'Hello {name}!'
  
  
  # We define here the fixture in the test file:
  """Some data for our tests."""
  from pytest import fixture
  @fixture
  def names():
      return 'Bob', '', None, 123, [], ()
      
  # Now the test can run like this, to test many different formats at once (defined in the fixture function):
  def test_say_hello_to(names):
      assert say_hello_to('Stefan') == 'Hello Stefan!'
  
      bob, empty, none, integer, li, tup = names
  
      assert say_hello_to(bob) == 'Hello Bob!'
      assert say_hello_to(empty) == 'Hello !'
      assert say_hello_to(none) == 'Hello None!'
      assert say_hello_to(integer) == 'Hello 123!'
      assert say_hello_to(li) == 'Hello []!'
      assert say_hello_to(tup) == 'Hello ()!'

Doctest: we can also do tests using function docstrings:

.. sourcecode:: python

  # Here some function with a test in the docstring:
  def say_hello_to(name='World'):
      """Say hello.
  
      >>> say_hello_to('Stefan')
      'Hello Bob!'
      """
      return f'Hello {name}!'

Now the test will run like this:

➜ pytest --doctest-modules
...
009     >>> say_hello_to('Stefan')
Expected:
    'Hello Bob!'
Got:
    'Hello Stefan!'
    
So here, the test is defined in the docstring itself!  

Coverage (of unit test): pytest-cov
-----------------------------------------------------------

Coverage gives the fraction of the code which is covered by unit tests, in percent. You need to define a .coveragerc file that will basically tell what not to include in the coverage calculation. Pytest-cov is built on top of coverage.py package (https://coverage.readthedocs.io/en/latest/index.html). 

For example (see https://coverage.readthedocs.io/en/latest/source.html#source)

.. sourcecode:: python

  [run]
  omit =
      # omit anything in a .local directory anywhere
      */.local/*
      # omit everything in /usr
      /usr/*
      # omit this single file
      utils/tirefire.py
      
Also a single function or class can be omitted by adding the comment next to its start (see https://coverage.readthedocs.io/en/coverage-4.3.3/excluding.html, https://coverage.readthedocs.io/en/latest/config.html)

.. sourcecode:: python

  class MyObject(object):
      def __init__(self):
          blah1()
          blah2()
  
      def __repr__(self): # pragma: no cover
          return "<MyObject>"
	  
So here the "# pragma: no cover" avoids the __repr__ to be used in coverage calculation. If we want to omit the full class in coverage calculation, 

.. sourcecode:: python

  class MyObject(object): # pragma: no cover
  
Some good links on coverage:

- https://rorymurdock.github.io/2019/11/23/Code-Coverage.html, https://gist.github.com/rorymurdock/f8c1ace6e35684261823530e19510478

- https://pypi.org/project/pytest-cov/, https://coverage.readthedocs.io/en/latest/index.html



How to package an application in python
===========================================================

Good links:

- General tutorial: https://packaging.python.org/tutorials/packaging-projects/

- Here is an example of project that can be used to build a package: https://github.com/pypa/sampleproject

Numpy basic documentation
===========================================================

.. figure:: Cheatsheets/Numpy_Python_Cheat_Sheet.png
   :scale: 100 %
   :alt: map to buried treasure

   This Cheatsheet is taken from DataCamp. 


Basic Pandas documentation
============================================================

.. topic:: Introduction

    The objective here is to have everything useful for the projects, not to make a complete documentation of the whole package. Here I will try to document both version 1.6 and >2.0. A special enphase will be done on machine learning module ml (mllib is outdated).
 
 
 
Good Pandas links:
----------------------------

A good link on data manipulations: https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/
   
Loading Pandas dataframe from file
------------------------------------------------------------

.. sourcecode:: python

  #Loading a Pandas dataframe:
  df_pd = pd.read_csv("/home/BC4350/Desktop/Iris.csv")
   
   
Creation of some data in a Pandas dataframe
------------------------------------------------------------

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
  
Creating dataframe with several objects per cell
------------------------------------------------------------
  
.. sourcecode:: python

  a = ['a1','a2','a3']
  b = ['b1','b2','b3']
  uu = [[a,b] for a,b in list(zip(a,b))]
  vv = [{'a':a,'b':b} for a,b in list(zip(a,b))]
  df = pd.DataFrame()
  df['version_list'] = uu
  df['version_dico'] = vv
  df  
  
    version_list version_dico
  0 [a1, b1]     {'a': 'a1', 'b': 'b1'} 
  1 [a2, b2]     {'a': 'a2', 'b': 'b2'} 
  2 [a3, b3]     {'a': 'a3', 'b': 'b3'} 
  
  
Stacking of dataframes in Pandas
------------------------------------------------------------

This will create a new df that contains the columns of both dataframes:

.. sourcecode:: python

  df1 = pd.DataFrame([1,2,3],columns=['A'])
  df2 = pd.DataFrame([4,5,6],columns=['B'])
  df3 = pd.concat([df1,df2],axis=1)

 
How to shuffle the columns of a dataframe?
------------------------------------------------------------

Simply by using the "sample" method, which allows to shuffle rows (only). For that we first transpose the df first:

.. sourcecode:: python

  # Shuffling the columns
  df_T = df.T
  df_T = df_T.sample(frac=1) 
  df   = df_T.T

  
Pandas and memory
------------------------------------------------------------

How to estimate the size a dataframe takes in memory?

.. sourcecode:: python

  df = pd.DataFrame(np.random.random((100,100)))
  df.values.nbytes  
  
  80000 #number of bytes
  
  #Here it gives the number of bytes for EACH column:
  df.memory_usage()
  
  #info()
  df.info() gives the types of the columns and the total memory used
  
  
Re-setting of index in Pandas dataframes
---------------------------------------------------

http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.set_index.html  

https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.reset_index.html

.. sourcecode:: python

  # Use a column of df for index:
  ts_all.set_index('transactiondate',inplace=True)

  # Reset index to 0,1,2,3... (note that the old index will be as the first column of the df)
  ts_all.reset_index(inplace=True)

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
-----------------------------------------------------

.. sourcecode:: python

  df.drop(['column'], axis=1, inplace=True)
  df.drop(['column1','column2'], axis=1, inplace=True)

  
Displaying dataframes to screen
-----------------------------------------------------

.. sourcecode:: python

  #This allows you to display as many rows as you wish when you display the dataframe (works also for max_rows):
  pd.options.display.max_columns = 50   #By default 20 only  
  
  #This display the 5 first rows:
  df.head(5)
  
  #This display the 5 last rows:
  df.tail(5)  
  
  Display several dataframes in the same HTML format in one cell:
  from IPython.core import display as ICD
  ICD.display(df1.head())
  ICD.display(df2.head())  
  
Reading very big files using chunk
-----------------------------------------------------
  
For csv that can be bigger than the RAM, we can load chunks of them, and perform (for example, can be different action) a filtering on the chunks like this:
  
.. sourcecode:: python
  
  def filter_is_long_trip(data):
    "Returns DataFrame filtering trips longer than 20 minutes"
    is_long_trip = (data.trip_time_in_secs > 1200)
    return data.loc[is_long_trip]

  chunks = []

  for chunk in pd.read_csv(filename, chunksize=1000):
    chunks.append(filter_is_long_trip(chunk))

  #or in a simpler way:  
  chunks = [filter_is_long_trip(chunk) for chunk in pd.read_csv(filename,chunksize=1000) ]  
  
  #Then we can use these filtered chunks and stack them into a single dataframe:
  df = pd.concat(chunks)
  
  
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
  
Retrieval of data from SQL data warehouse
-----------------------------------------------------

This exports the data in a simple array:

.. sourcecode:: python

  import pyodbc as odbc 

  # Some super SQL query
  sql = """SELECT top 100
  table as RUN_TS
  ,b.[AC_KEY]
  ,[PROBABILITY_TRUE]
  FROM [DB].[test].[B_DCS_DK_ROL] b
  JOIN db.ctrl.run_info r ON r.RUN_ID=b.RUN_ID
  """
  conn = odbc.connect(r'Driver={SQL Server};Server=SERVER;Database=DB;Trusted_Connection=yes;')
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
  from db.dbo.table
  order by field1, field2
  """

  df = Extract_data_from_SQLserver('server','db',SQLcommand)
  
  
Exporting data to SQL warehouse
-------------------------------------------

Let's say we have some dataframe, here FinalListModel1:

.. sourcecode:: python

  import pypyodbc as odbc

  conn = odbc.connect(r'Driver={SQL Server};Server=SERVER;Database=DB;Trusted_Connection=yes;')

  rows1 = list(FinalListModel1['caseid']) 
  rows2 = list(FinalListModel1['recordkey'])
  rows3 = list(FinalListModel1['score1'])
  rows = list(zip(rows1,rows2,rows3))

  cursor = conn.cursor() 

  stm="""
  DROP TABLE [DB].[dbo].[table]
  CREATE TABLE [DB].[dbo].[table] (
      [caseid] nvarchar(255),
      [recordkey] nvarchar(255),
      [score1] float
  )
  """
  res = cursor.execute(stm)
  cursor.executemany('INSERT INTO [DB].[dbo].[table] VALUES (?, ?, ?)', rows)
  conn.commit()
  
  cursor.close()
  conn.close()  


Transform format of dataframe: collapse multiple columns into one
------------------------------------------------------------------------------------------------

https://stackoverflow.com/questions/28520036/how-to-collapse-columns-into-row-elements-in-pandas

Here the task is to collapse multiple columns into one, keeping the same index (called "level_1" in the result)

.. sourcecode:: python

    df = pd.DataFrame(np.random.rand(4,5), columns = list('abcde'))
    df.head()
       
             a        b        c        d        e
    0 0.682871 0.287474 0.896795 0.043722 0.629443  
    1 0.456231 0.158333 0.796718 0.967837 0.611682  
    2 0.499535 0.545836 0.403043 0.465932 0.733136 
    3 0.553565 0.688499 0.813727 0.183788 0.631529   
  
    df.unstack().reset_index()   
    
      level_0  level_1         0
     0      a        0  0.682871 
     1      a        1  0.456231 
     2      a        2  0.499535 
     3      a        3  0.553565 
     4      b        0  0.287474 
     5      b        1  0.158333 
     6      b        2  0.545836 
     7      b        3  0.688499 
     8      c        0  0.896795 
     9      c        1  0.796718 
    10      c        2  0.403043 
    11      c        3  0.813727 
    12      d        0  0.043722 
    ....
    19      e        3  0.631529 
    
    # A more convenient form could be:

    df2 = df.unstack().reset_index().loc[:,['level_1',0]]
    df2.columns = ['index','value']
    df2.set_index('index',inplace=True)
    df2 

             value
    index
        0 0.682871 
        1 0.456231 
        2 0.499535 
        3 0.553565 
        0 0.287474 
        1 0.158333 
        2 0.545836 
        3 0.688499 
        0 0.896795 
        1 0.796718 
        2 0.403043 
        3 0.813727 
        0 0.043722 
    ...
        3 0.631529 
    
  

  
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
  

Dataframe containing column of lists
------------------------------------------------

1. From 1 column of lists to several columns (explode operation)

Based on https://stackoverflow.com/questions/35491274/pandas-split-column-of-lists-into-multiple-columns

Containing lists in a column is handy for example when dealing with time series, or in general to contain different data format in the same dataframe.

How to explode the lists to several columns?

Let's say we have a df like this:

.. sourcecode:: python

  d1 = {'teams': [['SF', 'NYG'],['SF', 'NYG'],['SF', 'NYG'],
                  ['SF', 'NYG'],['SF', 'NYG'],['SF', 'NYG'],['SF', 'NYG']]}
  df2 = pd.DataFrame(d1)
  
  print (df2)
         teams
  0  [SF, NYG]
  1  [SF, NYG]
  2  [SF, NYG]
  3  [SF, NYG]
  4  [SF, NYG]
  5  [SF, NYG]
  6  [SF, NYG]

We can explode the column of lists in 2 columns in the same dataframe like this:

.. sourcecode:: python

  df2[['team1','team2']] = pd.DataFrame(df2.teams.values.tolist(), index= df2.index)

  print (df2)
         teams team1 team2
  0  [SF, NYG]    SF   NYG
  1  [SF, NYG]    SF   NYG
  2  [SF, NYG]    SF   NYG
  3  [SF, NYG]    SF   NYG
  4  [SF, NYG]    SF   NYG
  5  [SF, NYG]    SF   NYG
  6  [SF, NYG]    SF   NYG
  
We can also do the same and create a new dataframe:

.. sourcecode:: python

  df3 = pd.DataFrame(df2['teams'].values.tolist(), columns=['team1','team2'])

  print (df3)
  
    team1 team2
  0    SF   NYG
  1    SF   NYG
  2    SF   NYG
  3    SF   NYG
  4    SF   NYG
  5    SF   NYG
  6    SF   NYG

The same operation using apply function is a bad idea as very slow (loop).

For the same kind of operation in Spark there is the command "explode". See section "Revert from time series (list) format to traditional (exploded) format".

2. From several columns to 1 column of lists

How to do the inverse operation in Pandas? Making a column of lists from several columns? In Spark I know (See subsection "Create time series format from row time series")

In pandas a simple apply function can do it (although might be slow):

.. sourcecode:: python

  df = pd.DataFrame({'a': [1, 2, 3], 
                     'b': [4, 5, 6]}) 
  df.head() 

    a b
  0 1 4 
  1 2 5 
  2 3 6 
  
  df['ab'] = df[['a', 'b']].apply(lambda x: list(x), axis = 1) 
  df.head() 
  
    a b     ab
  0 1 4 [1, 4]
  1 2 5 [2, 5] 
  2 3 6 [3, 6]   
  
  Note that there is a MUCH faster way (try %timeit), since apply is a slow function:
  
  df['ab'] = [[a,b] for a,b in zip(df['a'], df['b'])] 
  
  The problem is that the syntax is not as flexible (does not allow long list of columns...)
  
A better way of doing this (also suited to multiple columns at once). Very useful, as I often need to do such operations to convert events into time series:

.. sourcecode:: python

  # https://stackoverflow.com/questions/40357671/apply-list-function-on-multiple-columns-pandas
  # One df with 2 columns. We want to create a column with lists of B based on column A groups
  df = pd.DataFrame({'A': [1,1,2,2,2,2,3],'B':['a','b','c','d','e','f','g']})
  
  df = df.groupby('A').agg({'B': lambda x: list(x)})
  print (df)
                B
  A              
  1        [a, b]
  2  [c, d, e, f]
  3           [g]
  
  
  
Exploding a dataframe of lists of items (with ID column) into exploded ID-item column
--------------------------------------------------------------------------------------------------------

From https://towardsdatascience.com/pandas-tips-i-wish-i-knew-before-ef4ea6a39e1a

Let’s create a DataFrame with a column that has a random number of elements in lists:

.. sourcecode:: python

  n = 10
  df = pd.DataFrame(
      {
          "list_col": [[random.randint(0, 10) for _ in range(random.randint(3, 5))] for _ in range(10)],
      }
  )
  df.shape #(10, 1) output
  
  	list_col
  0	[0, 8, 4, 10]
  1	[0, 9, 9, 7]
  2	[7, 1, 0, 9, 6]
  3	[1, 3, 7]
  4	[1, 0, 1] 
  
Now, let’s execute the explode function.  

.. sourcecode:: python

  df = df.explode("list_col")
  df.shape #(40, 1) output
  
    list_col
 0	0
 0	8
 0	4
 0	10
 1	0
 1	9
 1	9  
 1      7

  
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
   
   
In the case we want to extract N rows randomly per group. So let's say we have a dataframe and group by a key "b":

.. sourcecode:: python

  df = pd.DataFrame({'a': [1,2,3,4,5,6,7,8,9,10,11], 'b': [1,1,1,0,0,0,0,2,2,2,2]})
  
  df.head(20)
  
  #There are 2 ways to do it: 
  
  #slower, but ouptut sorted by key:
  df.groupby('b', group_keys=False).apply(pd.DataFrame.sample, n=2).head(20)
  
  #much faster, just output not sorted by key:
  df.sample(frac=1).groupby('b').head(2)   
   

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
  
  
Example of computing the cumulative sum of a quantity over 2 groups:

.. sourcecode:: python

  df = pd.DataFrame({'col1' : ['a','a','b','b','a'],
               'col2' : ['2013/01/03 00:00:00', '2013/03/05 09:43:31', '2013/03/07 00:00:00',\
                         '2013/03/07 00:00:00', '2013/03/07 00:00:00'],
               'col3' : [1,3,1,2,0]})
  df = df.sort_values(['col1','col2'])  
  
    col1 col2                col3
  0 a    2013/01/03 00:00:00 1 
  1 a    2013/03/05 09:43:31 3 
  4 a    2013/03/07 00:00:00 0 
  2 b    2013/03/07 00:00:00 1 
  3 b    2013/03/07 00:00:00 2 
  
  df = df.assign(cumsum_col3=df.groupby('col1')['col3'].transform('cumsum')).sort_values('col1')

    col1 col2                col3 cumsum_col3
  0 a    2013/01/03 00:00:00 1    1
  1 a    2013/03/05 09:43:31 3    4
  4 a    2013/03/07 00:00:00 0    4
  2 b    2013/03/07 00:00:00 1    1
  3 b    2013/03/07 00:00:00 2    3
  
In spark it would have been:

.. sourcecode:: python

  df = pd.DataFrame({'col1' : ['a','a','b','b','a'],
               'col2' : ['2013/01/03 00:00:00', '2013/03/05 09:43:31', '2013/03/07 00:00:00',\
                         '2013/03/07 00:00:00', '2013/03/07 00:00:00'],
               'col3' : [1,3,1,2,0]})
  df = df.sort_values(['col1','col2']) 
  dff = sqlContext.createDataFrame( df )  
  dff.show()
  
  +----+-------------------+----+
  |col1|               col2|col3|
  +----+-------------------+----+
  |   a|2013/01/03 00:00:00|   1|
  |   a|2013/03/05 09:43:31|   3|
  |   b|2013/03/07 00:00:00|   1|
  |   b|2013/03/07 00:00:00|   2|
  |   a|2013/03/07 00:00:00|   0|
  +----+-------------------+----+  
  
  window = Window.partitionBy('col1').orderBy(asc('col1'),asc('col2'))
  dff=dff.withColumn('cumsum_col3', sum('col3').over(window))
  dff.orderBy(asc('col1'),asc('col2')).show()  
  
  +----+-------------------+----+-----------+
  |col1|               col2|col3|cumsum_col3|
  +----+-------------------+----+-----------+
  |   a|2013/01/03 00:00:00|   1|          1|
  |   a|2013/03/05 09:43:31|   3|          4|
  |   a|2013/03/07 00:00:00|   0|          4|
  |   b|2013/03/07 00:00:00|   1|          3|
  |   b|2013/03/07 00:00:00|   2|          3|
  +----+-------------------+----+-----------+  
  
  

  
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


Pivot operations
---------------------------------

The pivot allows to change the order of columns as such. Let's say we have some data as a time series, for different customers A,B,C...:

.. sourcecode:: python

  import pandas.util.testing as tm; tm.N = 3
  def unpivot(frame):
    N, K = frame.shape
    data = {'balance' : frame.values.ravel('F'),
            'customer' : np.asarray(frame.columns).repeat(N),
            'date' : np.tile(np.asarray(frame.index), K)}
    return pd.DataFrame(data, columns=['date', 'customer', 'balance'])
  df = unpivot(tm.makeTimeDataFrame())
  
.. figure:: Images/pivot_table1.png
   :scale: 100 %
   :alt: output  
   
.. sourcecode:: python   

  df_pivot = df.pivot(index='date', columns='customer', values='balance')

.. figure:: Images/pivot_table2.png
   :scale: 100 %
   :alt: output  
  
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

Check that 2 dataframes are equal
---------------------------------------------

...and if not what differs between them:

.. sourcecode:: python

  def dataframes_comparison_tool(d1,d2):

    df1 = d1.copy()
    df2 = d2.copy()
    df1 = df1.fillna(0)
    df2 = df2.fillna(0)

    ne_stacked = (df1 != df2).stack()
    changed = ne_stacked[ne_stacked]
    difference_locations = np.where(df1 != df2)
    changed_from = df1.values[difference_locations]
    changed_to = df2.values[difference_locations]
    return pd.DataFrame({'from': changed_from, 'to': changed_to}, index=changed.index)

dataframes_comparison_tool(result,dask_result)

Pandas and memory
--------------------------------------

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

  
Dask, or parallel Pandas
=====================================

Links:

- Cheatsheet: http://docs.dask.org/en/latest/_downloads/daskcheatsheet.pdf

- Dask general documentation: http://docs.dask.org/en/latest/dataframe.html

- Intro: https://towardsdatascience.com/how-i-learned-to-love-parallelized-applies-with-python-pandas-dask-and-numba-f06b0b367138

- Intro: https://sigdelta.com/blog/dask-introduction/

- On a cluster of several machines: http://matthewrocklin.com/blog/work/2017/01/12/dask-dataframes  

- Dask overview video (16 minutes): https://www.youtube.com/watch?v=ods97a5Pzw0

- Detailed Dask overview video (40 minutes): https://www.youtube.com/watch?v=mjQ7tCQxYFQ

- Parallelizing sklearn: https://github.com/dask/dask-examples/blob/master/machine-learning.ipynb

Other package: swifter:

- https://github.com/jmcarpenter2/swifter

- https://medium.com/@jmcarpenter2/swiftapply-automatically-efficient-pandas-apply-operations-50e1058909f9 


Python API's: 
=============================================

Flask
---------------------------------------------

Flask fast tutorial: https://flask.palletsprojects.com/en/1.1.x/quickstart/

One nice example of data in and result output: https://pythonbasics.org/flask-template-data/ 

.. sourcecode:: python

  from flask import Flask, render_template, request
  app = Flask(__name__, template_folder='templates')

  @app.route('/')
  def student():
      return render_template('student.html')
 
  @app.route('/result', methods=['POST', 'GET'])
  def result():
      if request.method == 'POST':
          result = request.form
          return render_template("result.html", result=result)
  
  if __name__ == 'main':
      app.run(debug=True)

Then put these templates into a folder "templates" in the project repo:

student.html: 

.. sourcecode:: html

  <form action = "http://localhost:5000/result" method = "POST">
      <p>Name <input type = "text" name = "Name" /></p>
      <p>Physics <input type = "text" name = "Physics" /></p>
      <p>Chemistry <input type = "text" name = "chemistry" /></p>
      <p>Maths <input type ="text" name = "Mathematics" /></p>
      <p><input type = "submit" value = "submit" /></p>
   </form>
   
result.html   

.. sourcecode:: html

  <!doctype html>
    <table border = 1>
      {% for key, value in result.items() %}
  
      <tr>
         <th> {{ key }} </th>
         <td> {{ value }} </td>
      </tr>
  
      {% endfor %}
   </table>

Then to launch the stuff, run this:

.. sourcecode:: python

  python -m flask run
  
Examples of deployment of a flask app using Azure DevOps: 

- https://docs.microsoft.com/en-us/azure/devops/pipelines/ecosystems/python-webapp?view=azure-devops

- https://elevate-org.com/2019/10/15/build-devops-ci-cd-pipeline-for-python-flask-with-azure-devops/


Streamlit
----------------------------------------------

https://www.geeksforgeeks.org/deploy-a-machine-learning-model-using-streamlit-library/

Streamlit cheatsheet: https://share.streamlit.io/daniellewisdl/streamlit-cheat-sheet/app.py
