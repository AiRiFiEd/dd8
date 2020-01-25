# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:02:31 2019

@author: LIM YUAN QING

Module contains classes and functions that aims to combine functionalities from 
different libraries (nltk, spacy, textblob etc...) to automate process of 
text mining / sentiment analysis for the purposes of Syntactic Analysis and 
Semantic Analysis
1. tokenization
2. part of speech tagging
3. hyponyms/hypernyms determination
4. 


Classes
-------


"""
import nltk
nltk.download('omw')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import words as wd
import enum
import spacy

@enum.unique
class ENUM_DICTIONARY_SOURCE(enum.Enum):
    NLTK_WORDS = 'nltk.corpus.words'

class ENUM_DICTIONARY_LANGUAGE(enum.Enum):
    ENGLISH = 'en'
    CHINESE = 'ch'

class Dictionary():
    ## load various sources of word dictionaries and their respective scores
    def __init__(self, enum_dictionary_source, 
                 enum_dictionary_language):
        self.source = enum_dictionary_source
        self.language = enum_dictionary_language        
        
    def get_words(self):
        if self.source == ENUM_DICTIONARY_SOURCE.NLTK_WORDS:
            if self.language == ENUM_DICTIONARY_LANGUAGE.ENGLISH:
                return wd.words()

class Text():
    def __init__(self, str_content):
        self.__str_content = str_content
    
    def __repr__(self):
        pass
    
    def __len__(self):
        pass
    
    def tokenize(self, bln_sentence = True):
        if bln_sentence:
            return sent_tokenize(self.__str_content)
        else:
            return word_tokenize(self.__str_content)
    
    def tag_part_of_speech(self):
        return nltk.pos_tag(self.tokenize())
    
    def get_synset(self, str_word, str_pos):
        pass
    
    
if __name__ == '__main__':
    print(wn.langs())
    
        
    