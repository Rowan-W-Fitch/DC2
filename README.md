# DC2
data challenge 2, classifies email as spam or ham

requires install of nltk, sklearn, skope-rules, pandas, numpy, matplotlib
INSTALL SKLEARN with pip install sklearn, visit https://pypi.org/project/sklearn/
INSTALL SKOPE_RULES with pip install skope-rules, visit https://pypi.org/project/skope-rules/
INSTALL PANDAS with pip install pandas, visit https://pypi.org/project/pandas/
INSTALL NUMPY with pip install numpy, visit https://pypi.org/project/numpy/
INSTALL MATPLOTLIB with pip install matplotlib, visit https://pypi.org/project/matplotlib/

This data model uses a rule based approach, where a set of features are taken from a sample of spam email, and another set is taken from the ham email. The disjoint union of those sets are caluclated, and then the occurence of words from both of those sets is taken from each email. Thses rules are used in the determination if an email is spam or ham. All data is put into a pandas dataframe
