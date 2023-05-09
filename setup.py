import setuptools

setuptools.setup(
     name="YTTitleSuggest",
     version="1.0",
     author="Daniel Chang",
     author_email="d.chang@yale.edu",
     description="A Python package to suggest YouTube Title Keyword and Titles",
     packages=setuptools.find_packages(),
     install_requires=['numpy', 'pandas', 'wordcloud', 'matplotlib', 'openai', 'nltk', 'gensim', 'spacy', 'tqdm']
)
