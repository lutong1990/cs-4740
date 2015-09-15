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
                        bigram = words[i+1]        
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

s = 'a'
x = find_next_word(s)
y = find_next_word(x)
z = find_next_word(y)
w = find_next_word(z)
q = find_next_word(w)

print s + ' ' + x + " " + y + " " + z + " "+ w +' '+ q + "."
