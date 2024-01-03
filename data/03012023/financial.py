# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:29:13 2024

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
data = pd.read_csv(r"D:\Ozyegin7\CS554\project\financial_phrasebank.csv")

# Save the original text and labels to a CSV file
output_df_original = data[['text', 'label']]
output_df_original.to_csv(r"D:\Ozyegin7\CS554\project\financial_phrasebank_original.csv", index=False)

# Tokenization and Stemming
stemmer = PorterStemmer()
data['stemmed_text'] = data['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(x)]))
data_stemmed = data[['stemmed_text', 'label']]  # Create a new DataFrame containing 'stemmed_text' and 'label' columns

# Tokenization and Lemmatization
lemmatizer = WordNetLemmatizer()
data['lemmatized_text'] = data['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, wordnet.VERB) for word in nltk.word_tokenize(x)]))
data_lemmatized = data[['lemmatized_text', 'label']]  # Create a new DataFrame containing 'lemmatized_text' and 'label' columns

# Save the results to CSV files without original text
output_df_stemmed = data_stemmed[['stemmed_text', 'label']]
output_df_stemmed.to_csv(r"D:\Ozyegin7\CS554\project\financial_phrasebank_stemmed.csv", index=False)

output_df_lemmatized = data_lemmatized[['lemmatized_text', 'label']]
output_df_lemmatized.to_csv(r"D:\Ozyegin7\CS554\project\financial_phrasebank_lemmatized.csv", index=False)