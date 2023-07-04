import os
import re
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

## initialise the inbuilt Stemmer
stemmer = PorterStemmer()

## We can also use Lemmatizer instead of Stemmer
lemmatizer = WordNetLemmatizer()

def preprocess(raw_text: str, flag: str) -> str:
    # change sentence to lower case
    sentence = raw_text.lower().strip()

    # Replace certain special characters with their string equivalents
    mappings = {'%': ' percent', '$': ' dollar ',
                '₹': ' rupee ', '€': ' euro ', '@': ' at'}

    for k, v in mappings.items():
        sentence = sentence.replace(k, v)
    
    # Removing special characters
    sentence = re.sub("\W", " ", sentence).strip()

    # tokenize into words
    tokens = sentence.split()
    
    # remove stop words                
    clean_tokens = [t for t in tokens if not t in stopwords.words("english")]
    
    # Stemming/Lemmatization
    if(flag == 'stem'):
        clean_tokens = [stemmer.stem(word) for word in clean_tokens]
    else:
        clean_tokens = [lemmatizer.lemmatize(word) for word in clean_tokens]
    
    return " ".join(clean_tokens)
