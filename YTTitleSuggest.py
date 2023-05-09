import pandas as pd
import numpy as np
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
import openai

class Title(object):
    
    """
    A class for suggesting YouTube video titles.

    Attributes:
        df (pd.DataFrame): The DataFrame containing title data.
        category_list_all (list): A list of all available categories on YouTube.
        all_keywords (list): A list of all keywords.
        category_list (list): A list of categories with available Word2Vec models.
        keyword (str): The currently set keyword.
        category (str): The currently set category.
        api_key (str): The API key for OpenAI.

    Methods:
        set_keyword(self, keyword): Set the keyword attribute.
        set_category(self, category): Set the category attribute.
        show_wordCloud(self, category=None): Create a word cloud using titles in specified categories.
        keyword_list(self, category=None): Return keywords in a specified category.
        keyword_category(self, keyword=None): Return categories in which the keyword is found.
        example_titles(self, keywords=None, category=None): Return titles in a specified category that contain specified keywords.
        generate_keywords(self, num=10, positive=None, negative=None, category=None): Return a list of related keywords given positive and negative keywords.
        generate_title_GPT(self, positive=None, negative=None, category=None, engine='ChatGPT'): Generate a YouTube title using OpenAI's ChatGPT 3.5 or Davinci 3.0 models.

    """
    
    def __init__(self, keyword=None, category=None, df=None):
        """
        Initialize the Title class.

        Parameters:
            keyword (str, optional): The keyword to set for the title. 
            category (str, optional): The category to set for the title. 
            df (pd.DataFrame, optional): The DataFrame to use for title data. Defaults to None, if None, uses Youtube Trending Data
            in the package.

        Raises:
            Exception: If the given keyword is not within the list of available keywords.
            Exception: If the given category is not within the list of available categories with Word2Vec models.
        """
        
        if category is None:
            print('Please Use YTTitleSuggest.Title.set_category() to Set a Category')
            print('Search Available Categories Using YTTitleSuggest.Title.category_list')
            print('')
            
        if keyword is None:
            print('Please Use YTTitleSuggest.Title.set_keyword() to Set a Keyword')
            
        if df is None:
            self.df = pd.read_csv('Data/data.csv', keep_default_na=False) 
            self.category_list_all = list(pd.read_csv('Data/category.csv')['category'])
        else:
            self.df = df
            if 'cleaned_title' not in self.df.columns:
                pass
            self.category_list_all = list(np.unique([title for title in self.df['category_title'] if title is not np.nan]))
        
        all_keywords = []
        category_list = []
        
        # Search through saved models and obtain keywords
        
        for cat in self.category_list_all: 
            try:
                wv = KeyedVectors.load('model/{0}.kv'.format(cat))
                keys = list(wv.key_to_index.keys())
                for key in keys:
                    all_keywords.append(key)
                category_list.append(cat)
            except:
                pass
            
        self.all_keywords = set(all_keywords)
        self.category_list = list(category_list)
        
        if keyword is not None:
            self.set_keyword(keyword)
        else: self.keyword = None
            
        if category is not None:
            self.set_category(category)
        else: self.category = None
            
        self.api_key = None
            
        
    def set_keyword(self, keyword):
        '''Set keyword attribute.'''
        if keyword not in self.all_keywords:
            raise Exception('Invalid Keyword')
        self.keyword = keyword
        
    def set_category(self, category):
        '''Set category attribute.'''
        if category not in self.category_list:
            raise Exception('Invalid Category, Please Use YTTitleSuggest.Title.category_list to Search for Available Categories')
        self.category = category
                        
    def show_wordCloud(self, category=None):
        '''Create a wordCloud using words in titles in specified categories.'''

        if category == 'all':
            self.__wordCloud_generator(self.df['cleaned_title'], title="Top Keywords")
            return
        
        if category is None:
            
            category = self.category
            
            if category is None:
                print('No Category Selected, Showing Results for All Categories')
                self.__wordCloud_generator(self.df['cleaned_title'], title="Top Keywords")
            else: 
                self.__wordCloud_generator(self.df['cleaned_title'][self.df['category_title'] == category], title="Top Keywords")
        else: 
            if category not in self.category_list:
                raise Exception('Invalid Category, Please Use YTTitleSuggest.Title.category_list to Search for Available Categories')
            self.__wordCloud_generator(self.df['cleaned_title'][self.df['category_title'] == category], title="Top Keywords")
            
            
    def __wordCloud_generator(self, cleaned_data, title=None):
        wordcloud = WordCloud(width = 800, height = 800,
                              background_color ='black',
                              min_font_size = 10
                             ).generate(" ".join(data for data in cleaned_data if data != ''))                      
        plt.figure(figsize = (8, 8), facecolor = None) 
        plt.imshow(wordcloud, interpolation='bilinear') 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
        plt.title(title,fontsize=30)
        plt.show()
        
        
    def keyword_list(self, category=None):
        '''Return keywords in a specified category.'''
        
        if category == 'all':
            return self.all_keywords
        
        if category is None:
            
            category = self.category
            
            if category is None: 
                print('No Category Set, Showing Results for All Categories')
                print('Please Set a Category to Rank by Frequency')
                return self.all_keywords

            wv = KeyedVectors.load('model/{0}.kv'.format(category))
            keys = list(wv.key_to_index.keys())
            return keys
                
        else:
            if category not in self.category_list:
                raise Exception('Invalid Category, Please Use YTTitleSuggest.Title.category_list to Search for Available Categories')
            wv = KeyedVectors.load('model/{0}.kv'.format(category))
            keys = list(wv.key_to_index.keys())
            return keys
                
                                                                                                               
    def keyword_category(self, keyword=None):
        '''Return categories in which the keyword is found.'''
        
        if keyword is None:
            keyword = self.keyword
            if keyword is None:
                raise Exception('No Keyword Selected')
            if keyword not in self.all_keywords:
                raise Exception('Invalid Keyword')
        
        # Iterate through categories, check if keyword in category
        
        cats=[]
        for category in self.category_list:  
            wv = KeyedVectors.load('model/{0}.kv'.format(category)) 
            keys = list(wv.key_to_index.keys())
            if self.keyword in keys:
                cats.append(category)
                    
        if len(cats) > 0:
            return list(cats)
        else:
            raise Exception('Keyword in Category with Insufficient # of Samples, Please Choose Another Keyword')
            
    def example_titles(self, keywords=None, category=None):
        """
        Return titles in a specified category that contain specified keywords.

        Parameters:
            keywords (list, optional): The keywords to search for in the titles. Defaults to Set Keyword.
            category (str, optional): The category to search for titles in. Defaults to Set Category.

        Returns:
            list: The list of titles.

        Raises:
            Exception: If no keyword is set.
            Exception: If no category is set.
            Exception: If an invalid category is provided.

        """
        
        if keywords is None:
            keywords = [self.keyword]
            if self.keyword is None:
                raise Exception('No Keyword Set')
                
        if category is None:
            category = self.category
            if category is None:
                raise Exception('No Category Set')
        else: 
            if category not in self.category_list:
                raise Exception('Invalid Category, Please Use YTTitleSuggest.Title.category_list to Search for Available Categories')
                
        all_titles = self.df['title'][self.df['category_title'] == category]
        all_cleaned = self.df['cleaned_title'][self.df['category_title'] == category]
        
        titles = []
        for i in range(len(all_titles)): # Iterate through all titles
            
            #Split cleaned titles into words and check if all keywords are in the list of words
            
            for keyword in keywords:
                if keyword in str(all_cleaned.iloc[i]).split():
                    all_in_title = True
                else:
                    all_in_title = False
                    break
            if all_in_title == True:
                titles.append(all_titles.iloc[i])
        
        if len(titles) > 0:
            return list(titles)
        else:
            raise Exception('No Titles Keyword(s) {0} in Category {1}'.format(keywords, category))
    
    def generate_keywords(self, num=10, positive=None, negative=None, category=None):
        """
        Return a list of related keywords given positive and negative keywords.

        Parameters:
            num (int, optional): The number of related keywords to generate. Defaults to 10.
            positive (list, optional): The positive keywords. Defaults to Set Keyword.
            negative (list, optional): The negative keywords. Defaults to None.
            category (str, optional): The category to generate keywords for. Defaults to Set Category.

        Returns:
            list: The list of related keywords.

        Raises:
            Exception: If no category is set.
            Exception: If an invalid category is provided.

        """

        if positive is None:
            positive = [self.keyword]
                
        if category is None:
            category = self.category
            if category is None:
                raise Exception('No Category Set')
        else: 
            if category not in self.category_list:
                raise Exception('Invalid Category, Please Use YTTitleSuggest.Title.category_list to Search for Available Categories')
            
        wv = KeyedVectors.load('model/{0}.kv'.format(category))
        
        # Iterate through given keys, check if they are in the keyword list of the Word2Vec Model
        
        keys_positive = []
        if positive is not None:
            for word in positive:
                if word in wv.key_to_index.keys():
                    keys_positive.append(word)
        
        keys_negative = []
        if negative is not None:
            for word in negative:
                if word in wv.key_to_index.keys():
                    keys_negative.append(word)
                    
        if len(keys_positive) == 0 and len(keys_negative) == 0:
            raise Exception('Invalid Keyword(s)')
        
        return wv.most_similar(positive=keys_positive, negative=keys_negative, topn=num)


    
    def generate_title_GPT(self, positive=None, negative=None, category=None, engine='ChatGPT'):
        """
        Use the OpenAI API to generate a YouTube title with given positive and negative keywords.

        Parameters:
            positive (str or list, optional): The positive keyword(s) to use. Defaults to Set Keyword.
            negative (str or list, optional): The negative keyword(s) to use. Defaults to None.
            category (str, optional): The category to generate the title for. Defaults to Set Category.
            engine (str, optional): The engine to use for title generation (either 'ChatGPT' or 'DaVinci'). Defaults to 'ChatGPT'.

        Raises:
            Exception: If no category is set.
            Exception: If an invalid category is provided.
            Exception: If no API key is set.

        """
        if positive is None:
            positive = self.keyword
            
        if category is None:
            category = self.category
            if category is None:
                raise Exception('No Category Set')
        else: 
            if category not in self.category_list:
                raise Exception('Invalid Category, Please Use YTTitleSuggest.Title.category_list to Search for Available Categories')
        
        if self.api_key is None:
            raise Exception('No OpenAI API Key Set, Please set API Key using YTTitleSuggest.Title.api_key')
        openai.api_key = self.api_key
        openai.OPENAI_API_KEY= self.api_key
        
            
        if engine == 'ChatGPT':
            mode = input("Adjectives : ")
            
            messages = [ {"role": "system", "content": "You are a intelligent assistant."} ] # Set up ChatGPT
            
            # Input Question
            
            if negative is None:
                messages.append(
                    {"role": "user", "content": "Write a {0} youtube video in category {1} about \"{2}\"".format(mode, category, positive)},
                )
            else: 
                messages.append(
                    {"role": "user", "content": "Write a {0} youtube video in category {1} about \"{2}\" and not about \"{3}\"".format(mode, category, positive, negative)},
                )
            
            # Generate a response
                
            chat = openai.ChatCompletion.create( 
                model="gpt-3.5-turbo", messages=messages
            )
            reply = chat.choices[0].message.content
            print(f"ChatGPT: {reply}")
        
        if engine == 'DaVinci':

            # Define the prompt
            if negative is None:
                prompt = "Write a {0} youtube video in category {1} about \"{2}\"".format(mode, category, positive)
            else:
                prompt = "Write a {0} youtube video in category {1} about \"{2}\" and not about \"{3}\"".format(mode, category, positive, negative)
            engine="text-davinci-003"

            # Generate a response
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.7
            )
            print(response["choices"][0]["text"])
    
        
        