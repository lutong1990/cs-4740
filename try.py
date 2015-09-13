from nltk.tokenize import RegexpTokenizer
import codecs
import os
import nltk
import sys
from sets import Set
import numpy as np

def bigram_count(filename):
    Bigram = {}
    wordSet = Set()
    tokenizer = RegexpTokenizer('\w+')    
    with codecs.open(filename, 'r') as f:
        for line in f:
            words = tokenizer.tokenize(line.lower())
            if len(words) > 0:
                for i, word in enumerate(words): 
                    if i < len(words)-1:    
                        if word not in wordSet:
                            wordSet.add(word)    
                            Bigram[word] = {}    
                        bigram = word +' '+ words[i+1]        
                        if bigram in Bigram[word]:
                            Bigram[word][bigram] += 1
                        else:
                            Bigram[word][bigram] = 1

#calculating the probabilities   
    for key,value in Bigram.iteritems():
        n = sum(Bigram[key].values())
        for key2,value2 in Bigram[key].iteritems():
            Bigram[key][key2] = float(value2)/n
    return Bigram


if __name__ == '__main__':
    word_count_dict = bigram_count('/home/tong/Documents/CS5740/Project1/train_books/children/the_junior_classics_vol_1.txt')

print word_count_dict['a']['a fire']
