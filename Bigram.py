from nltk.tokenize import RegexpTokenizer, word_tokenize, WhitespaceTokenizer
import codecs
import os
import nltk
import sys
import re
import numpy as np
from sets import Set

'''
At this stage, we have generated unigram and bigram corporas from one book;
For the random sentences generation, we use the bigram model
Please find annotated 'print' command at the end of these codes
'''

def unigram(filename):
    '''
    the function gives the pure unigram corpra
    '''
    word_count_dict = {}
    with codecs.open(filename,  'r') as f:
        sing_line=""        
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
        
    for word in word_1:    
        if re.match(r'\W+',word) is None:
            #for word in words:
            if word not in word_count_dict:
                word_count_dict[word] = 1
            else:
                word_count_dict[word] += 1 
        elif re.match(r'\W+',word) is not None:
            words_ch = re.findall('.', word)
            for word in words_ch:
                if word not in word_count_dict:
                    word_count_dict[word] = 1
                else:
                    word_count_dict[word] += 1
    
    n = sum(word_count_dict.values())
    for key,value in word_count_dict.iteritems():
        word_count_dict[key] = float(value)/n
  
    return word_count_dict
	
    
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
    return Bigram

def bigram_prob(Bigram):
	
	#calculating the probabilities
    for key,value in Bigram.iteritems():
        n = sum(Bigram[key].values())
        for key2,value2 in Bigram[key].iteritems():
            Bigram[key][key2] = float(value2)/n
    return Bigram


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
	word_count_dict = bigram_prob(justCount)
	next_word = np.random.multinomial(1,word_count_dict[word].values(),1)
	index = np.transpose(np.nonzero(next_word))[:,1]
	return word_count_dict[word].keys()[index]


def random_start(dictionary):
	'''
	dictionary needs to be two layers and be counts
	'''
	count_dot = sum(dictionary['.'].values())
	count_question = sum(dictionary['?'].values())
	count_exclamation = sum(dictionary['!'].values())
	n = count_dot + count_exclamation + count_question
	prob =  np.array((count_dot,count_question,count_exclamation))
	prob = prob/float(n)
	choice = np.random.multinomial(1,prob,1)
	index = np.transpose(np.nonzero(choice))[:,1]
	#print index

	if index == 0:
		word = find_next_word('.')
	elif index == 1:
		word = find_next_word('?')
	else:
		word = find_next_word('!')

	if word == "\"": #take out words that we do not want it to be a start
		return random_start(justCount) 
	return word


def sen_generator(sentence,word):
	if len(word) <= 0:
		word = random_start(justCount)
	new = find_next_word(word)
	if len(sentence) <= 0:
		return sen_generator(word,new)
	elif word not in ['.', '?', '!']:
		if word not in [",", ":", ";", "s", "t"]:
			return sen_generator(sentence + " " + word,new)
		else:
			return sen_generator(sentence + word,new)
	else:
		return sentence + word

#------------------------------------------------------------------------------------------------------------------
'''
please run the following one by one

this gives the unigram corpora


if __name__ == '__main__': 
    uniCor = unigram('/home/tong/Documents/CS5740/Project1/train_books/children/the_junior_classics_vol_1.txt') 

print uniCor
'''
#---------------------------------------------------------------------------------------------------------------------
'''
this gives the bigram corpora and sentence generation
I am not able to assign probabilities, reason is noted below.

if __name__ == '__main__': 
    justCount = bigram_count('/home/tong/Documents/CS5740/Project1/train_books/children/the_junior_classics_vol_1.txt')

    
    ## word_count_dict = bigram_prob(justCount) 
    ## I don't know why every time I do it, the 'justCount' changes its values to probabilities as well.
    ## therefore, I embedded the probability computation in 'find_next_word', and aim at solving the problem later

## start with seeding
#print sen_generator("I do not like", 'your') #the seeding sentence is "I do not like your ....." 
#print sen_generator("",'new') # the seeding sentence is "new ......"

## start without seeds
#print sen_generator("","")
'''