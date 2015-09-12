import os
import nltk
import sys
"""Tokenizing"""
def word_count(filename):

    word_count_dict = {}

    with open(filename, 'r') as f:
        for line in f:
            words = line.lower().split()
            for word in words:
                if word in word_count_dict:
                    word_count_dict[word] += 1
                else:
                    word_count_dict[word] = 1
                                   
    return word_count_dict


if __name__ == '__main__':
    word_count_dict = word_count('C:/Users/Saram/OneDrive/Cornell/General Language Process/project/books/train_books/children/the_junior_classics_vol_1.txt')



#Tokeninzing 2
import codecs
def word_count(filename):
    """A function that returns a dictionary with tokens as keys
    and counts of how many times each token appeared as values in
    the file with the given filename.

    Inputs:
        filename - the name of a plaintext file
    Outputs:
        A dictionary mapping tokens to counts.
    """

    word_count_dict = {}
    tokenizer = RegexpTokenizer('\w+')

    with codecs.open(filename, 'r') as f:
        for line in f:
            words = tokenizer.tokenize(line.lower())
            for word in words:
                if word not in word_count_dict:
                    word_count_dict[word] = 1
                else:
                    word_count_dict[word] += 1

    return word_count_dict


if __name__ == '__main__':
    path = 'C:/Users/Saram/OneDrive/Cornell/General Language Process/project/books/train_books/children/the_junior_classics_vol_1.txt'  
    #path = nltk.data.find('corpora/unicode_samples/polish-lat2.txt')
    word_count_dict = word_count(path)
    
"""Unigram probability??"""
#word_count_dict.head(5)
print word_count_dict
dict(sorted(word_count_dict.items()[:30]))


#tokenize for bigram

tokenizer = RegexpTokenizer('\w+')
test=tokenizer.tokenize("SONNETS TO THE ONLY BEGETTER OF THESE INSUING SONNETS MR. W. H. ALL HAPPINESS AND THAT ETERNITY PROMISED BY OUR EVER-LIVING".lower())
Bigram = {}
for word in test:
                #bigram = prev_word +' '+ word
                i=test.index(word)                
                bigram = word +' '+ test[i+1]        
                if bigram in Bigram:
                    Bigram[bigram]+=1
                else:
                    Bigram[bigram]= 1
print Bigram

tokenizer = RegexpTokenizer('\w+')
test=tokenizer.tokenize("SONNETS TO THE ONLY BEGETTER OF THESE INSUING SONNETS MR. W. H. ALL HAPPINESS AND THAT ETERNITY PROMISED BY OUR EVER-LIVING".lower())
Bigram = {}
for i,word in enumerate(test):        
                #i = test.index(word)                 
                if i < len(test):
                    bigram = word +' '+ test[i+1]        
                    if bigram in Bigram:
                        Bigram[bigram]+=1
                    else:
                        Bigram[bigram]= 1
                
print Bigram


len(test)
print test[2]
test2 =test[2] 
print test[2]+test[1+test.index(test[2])]
print " ".join(test[2:4])


def bigram_count(filename):
    Bigram = {}
    #prev_word = "START"
    tokenizer = RegexpTokenizer('\w+')    
    with codecs.open(filename, 'r') as f:
        for line in f:
            words = tokenizer.tokenize(line.lower())
            for i, word in enumerate(words):
                #bigram = prev_word +' '+ word
                #i = words.index(word)                 
                if i <len(words):    
                    bigram = word +' '+ words[i+1]        
                    if bigram in Bigram:
                        Bigram[bigram]+=1
                    else:
                        Bigram[bigram]= 1
               
if __name__ == '__main__':
    path = 'C:/Users/Saram/OneDrive/Cornell/General Language Process/project/books/train_books/children/the_junior_classics_vol_1.txt'  
    word_count_dict_2 = bigram_count(path)

bigram_count('C:/Users/Saram/OneDrive/Cornell/General Language Process/project/books/train_books/children/the_junior_classics_vol_1.txt')

#" "=words[1+len(words)]

###############################################################                 Why?  
Bigram = {}
tokenizer = RegexpTokenizer('\w+')    
with codecs.open('C:/Users/Saram/OneDrive/Cornell/General Language Process/project/books/train_books/children/the_junior_classics_vol_1.txt', 'r') as f:
   for line in f:
       words = tokenizer.tokenize(line.lower())
       for i, word in enumerate(words):
          if len(words)>0: 
              bigram = word +' '+ words[i+1]        
              if bigram in Bigram:
                  Bigram[bigram]+=1
              else:
                  Bigram[bigram]= 1
          elif len(words)==0: 
              pass
print Bigram
#prev_word = word
bigram_count()


Bigram.items()[:30]

bigram_count("SONNETS TO THE ONLY BEGETTER OF THESE INSUING SONNETS MR. W. H. ALL HAPPINESS AND THAT ETERNITY PROMISED BY OUR EVER-LIVING".lower())
f="SONNETS TO THE ONLY BEGETTER OF THESE INSUING SONNETS MR. W. H. ALL HAPPINESS AND THAT ETERNITY PROMISED BY OUR EVER-LIVING".codecs.lower()


help(list)
RegexpTokenizer?
tokenize?


"""Bigram"""
# initialize variables:
Bigrams = {}
prev_word = "START"
# loop over words in input:
for word in tokens:
# concatenate words to get bigram:
    bigram = prev_word + ' ' + word
    if bigram in Bigrams:
        Bigrams[bigram] += 1
    else:
        Bigrams[bigram] = 1
# change value of prev_word
prev_word = word




"""Calculating the n-gram
def ngrams(tokens, MIN_N, MAX_N):
    n_tokens = len(tokens)
    for i in xrange(n_tokens):
        for j in xrange(i+MIN_N, min(n_tokens, i+MAX_N)+1):
            yield tokens[i:j]
"""
#test

