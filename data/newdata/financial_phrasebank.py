# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 21:51:09 2023

@author: berk_
"""

import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import time

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load your dataset
data = pd.read_csv(r"D:\Ozyegin7\CS554\project\financial_phrasebank.csv")

# Calculate total character count and print to the console
total_chars_text = data['text'].apply(len).sum()

# Tokenization and Stemming
start_time = time.time()

# Stemming while preserving the original text column
stemmer = PorterStemmer()
data['stemmed_text'] = data['text']
data['stemmed_text'] = data['stemmed_text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in nltk.word_tokenize(x)]))
data_stemmed = data[['text', 'stemmed_text', 'label']]  # Create a new DataFrame containing 'text', 'stemmed_text', and 'label' columns

stemming_time = time.time() - start_time
print(f"Stemming time: {stemming_time:.2f} seconds")

# Tokenization and Lemmatization
start_time = time.time()

# Lemmatization while preserving the original text column
lemmatizer = WordNetLemmatizer()
data['lemmatized_text'] = data['text']
data['lemmatized_text'] = data['lemmatized_text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word, wordnet.VERB) for word in nltk.word_tokenize(x)]))
data_lemmatized = data[['text', 'lemmatized_text', 'label']]  # Create a new DataFrame containing 'text', 'lemmatized_text', and 'label' columns

lemmatization_time = time.time() - start_time
print(f"Lemmatization time: {lemmatization_time:.2f} seconds")

# Calculate total character counts and print to the console
data_stemmed['stemmed_text_chars'] = data_stemmed['stemmed_text'].apply(len)
data_lemmatized['lemmatized_text_chars'] = data_lemmatized['lemmatized_text'].apply(len)

print(f"Total character count (text): {total_chars_text}")
print(f"Total character count (stemmed_text): {data_stemmed['stemmed_text_chars'].sum()}")
print(f"Total character count (lemmatized_text): {data_lemmatized['lemmatized_text_chars'].sum()}")

# Create a DataFrame containing character counts for original, stemmed, and lemmatized text along with the label
output_df = pd.concat([data[['text']], 
                      data_stemmed[['stemmed_text', 'stemmed_text_chars']], 
                      data_lemmatized[['lemmatized_text', 'lemmatized_text_chars']], 
                      data[['label']]], 
                     axis=1)
output_df.columns = ['text', 'stemmed_text', 'stemmed_text_chars', 'lemmatized_text', 'lemmatized_text_chars', 'label']
output_df.insert(1, 'text_chars', data['text'].apply(len))  # Add a new column

output_df.to_excel(r"D:\Ozyegin7\CS554\project\financial_phrasebank_output.xlsx", index=False)
