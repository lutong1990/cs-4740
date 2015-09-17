from nltk.tokenize import RegexpTokenizer, word_tokenize, WhitespaceTokenizer
import codecs
import os
import nltk
import sys
import re
#Bigram
from sets import Set
    
def bigram_count(filename):
    Bigram = {}
    wordSet = Set()
    sing_line = ""    
      
    with codecs.open(filename, 'r') as f:
        for line in f:
            if len(line)>0:
                sing_line=sing_line + line
    sing_line = sing_line.replace("_", "")
    sing_line = sing_line.replace("*", "")
    sing_line = re.sub('\s+',' ',sing_line)
    sing_line = re.sub(' +',' ',sing_line)

    words = re.findall(r"[^\W\d_]+|\d+|[\W+^\s]", sing_line.lower())        
    word_1 = []       
    for word_0 in words: 
        if word_0 is not ' ':
            word_1.append(word_0)
    
    for i, word in enumerate(word_1):
        if i < len(word_1)-1:    
            if word not in wordSet:
                wordSet.add(word)    
                Bigram[word] = {}    
            bigram = word_1[i+1]        
                               
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


#Random sentance generator for Bigram
import numpy as np
def find_next_word(word):
	'''
	Here I use multinomial distribution to model the word generation

	A little background for multinomial distribution:
		it is a generlization of binomial distribution, in a way that the number of possible outcomes is greater than two 

	Use of the function:
		input a word -> it will use the probability of all words after it in the corpora, to randomly select a following word
						-> output the following word

	test of the function:
		run codes several times, change 'new' to something else, and run it multiple times again, you will see how the outputs change
	'''
	next_word = np.random.multinomial(1,word_count_dict[word].values(),1)
	index = np.transpose(np.nonzero(next_word))[:,1]
	return word_count_dict[word].keys()[index]

#print find_next_word("new")

def sen_gernerator(start):
    next_word = find_next_word(start)
    sentance = start + ' ' + next_word    
    while next_word is not ('.'or'?'or'!'):
        next_word = find_next_word(next_word)    
        if next_word is not ('.'or'?'or'!'):
            sentance = sentance + ' ' +  next_word
        
        else:
             sentance = sentance + next_word
             
    return sentance
    
    

print sen_gernerator("new")       
       