# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:39:33 2024

@author: berk_
"""

import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load your dataset
data_ag_news = pd.read_csv(r"D:\Ozyegin7\CS554\project\ag_news.csv")

# Save the original text and labels to a CSV file
output_df_original_ag_news = data_ag_news[['text', 'label']]
output_df_original_ag_news.to_csv(r"D:\Ozyegin7\CS554\project\ag_news_original.csv", index=False)

# Tokenization and Stemming
stemmer_ag_news = PorterStemmer()
data_ag_news['stemmed_text'] = data_ag_news['text'].apply(lambda x: ' '.join([stemmer_ag_news.stem(word) for word in nltk.word_tokenize(x)]))
data_stemmed_ag_news = data_ag_news[['stemmed_text', 'label']]  # Create a new DataFrame containing 'stemmed_text' and 'label' columns

# Tokenization and Lemmatization
lemmatizer_ag_news = WordNetLemmatizer()
data_ag_news['lemmatized_text'] = data_ag_news['text'].apply(lambda x: ' '.join([lemmatizer_ag_news.lemmatize(word, wordnet.VERB) for word in nltk.word_tokenize(x)]))
data_lemmatized_ag_news = data_ag_news[['lemmatized_text', 'label']]  # Create a new DataFrame containing 'lemmatized_text' and 'label' columns

# Save the results to CSV files without original text
output_df_stemmed_ag_news = data_stemmed_ag_news[['stemmed_text', 'label']]
output_df_stemmed_ag_news.to_csv(r"D:\Ozyegin7\CS554\project\ag_news_stemmed.csv", index=False)

output_df_lemmatized_ag_news = data_lemmatized_ag_news[['lemmatized_text', 'label']]
output_df_lemmatized_ag_news.to_csv(r"D:\Ozyegin7\CS554\project\ag_news_lemmatized.csv", index=False)
