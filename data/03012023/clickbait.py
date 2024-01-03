# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 17:43:06 2024

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
data_clickbait = pd.read_csv(r"D:\Ozyegin7\CS554\project\clickbait_notclickbait.csv")

# Save the original text and labels to a CSV file
output_df_original_clickbait = data_clickbait[['text', 'label']]
output_df_original_clickbait.to_csv(r"D:\Ozyegin7\CS554\project\clickbait_notclickbait_original.csv", index=False)

# Tokenization and Stemming
stemmer_clickbait = PorterStemmer()
data_clickbait['stemmed_text'] = data_clickbait['text'].apply(lambda x: ' '.join([stemmer_clickbait.stem(word) for word in nltk.word_tokenize(x)]))
data_stemmed_clickbait = data_clickbait[['stemmed_text', 'label']]  # Create a new DataFrame containing 'stemmed_text' and 'label' columns

# Tokenization and Lemmatization
lemmatizer_clickbait = WordNetLemmatizer()
data_clickbait['lemmatized_text'] = data_clickbait['text'].apply(lambda x: ' '.join([lemmatizer_clickbait.lemmatize(word, wordnet.VERB) for word in nltk.word_tokenize(x)]))
data_lemmatized_clickbait = data_clickbait[['lemmatized_text', 'label']]  # Create a new DataFrame containing 'lemmatized_text' and 'label' columns

# Save the results to CSV files without original text
output_df_stemmed_clickbait = data_stemmed_clickbait[['stemmed_text', 'label']]
output_df_stemmed_clickbait.to_csv(r"D:\Ozyegin7\CS554\project\clickbait_notclickbait_stemmed.csv", index=False)

output_df_lemmatized_clickbait = data_lemmatized_clickbait[['lemmatized_text', 'label']]
output_df_lemmatized_clickbait.to_csv(r"D:\Ozyegin7\CS554\project\clickbait_notclickbait_lemmatized.csv", index=False)
