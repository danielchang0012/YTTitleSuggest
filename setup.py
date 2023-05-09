import setuptools

setuptools.setup(
     name="YTTitleSuggest",
     version="1.0",
     author="Daniel Chang",
     author_email="d.chang@yale.edu",
     description="A Python package to suggest YouTube Title Keyword and Titles",
     packages=['YTTitleSuggest'],
     install_requires=['nltk', 'gensim', 'spacy', 'tqdm']
)