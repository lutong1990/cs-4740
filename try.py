from rewrite import n_gram_model, sentence_generator, genre
import codecs
import os
import sys
import re
from sets import Set


def tokenize(path):
	word_count_dict = {}
	'''
	with codecs.open(path,  'r') as f:
		sing_line=""        
		for line in f:
			if len(line)>0:
				sing_line=sing_line + line

	sing_line = sing_line.replace("_", "")
	sing_line = sing_line.replace("*", "")
	sing_line = re.sub('\s+',' ',sing_line)
	sing_line = re.sub(' +',' ',sing_line)
	'''
	sing_lines = []    
	#for i in range(len(filename)):     
	for filesing in path:
		with codecs.open(filesing,  'r') as f1:
		#f1=filesing        
			sing_line1=""        
			for line in f1:
				if len(line)>0:
					sing_line1=sing_line1 + line        
		sing_lines.append(sing_line1)

	for sing_line in sing_lines:
		sing_line = sing_line.replace("_", "")
		sing_line = sing_line.replace("*", "")
		sing_line = re.sub('\s+',' ',sing_line)
		sing_line = re.sub(' +',' ',sing_line)

	words = re.findall(r"[^\W\d_]+|\d+|[\W+^\s]", sing_line.lower())
	word_1 = []       

	for word_0 in words: 
		if word_0 is not ' ':
			word_1.append(word_0)

	return word_1

children_books = ['/home/tong/Documents/CS5740/Project1/train_books/children/the_junior_classics_vol_1.txt',
'/home/tong/Documents/CS5740/Project1/train_books/children/the_junior_classics_vol_4.txt',
'/home/tong/Documents/CS5740/Project1/train_books/children/the_junior_classics_vol_5.txt',
'/home/tong/Documents/CS5740/Project1/train_books/children/the_junior_classics_vol_6.txt',
'/home/tong/Documents/CS5740/Project1/train_books/children/the_junior_classics_vol_7.txt',
'/home/tong/Documents/CS5740/Project1/train_books/children/the_junior_classics_vol_8.txt']

children_test = ['/home/tong/Documents/CS5740/Project1/test_books/children/the_magic_city.txt']


if __name__ == '__main__': 
	docs_children = tokenize(children_books) 
	docs_test = tokenize(children_test)



children = n_gram_model(docs_children)
children.unigram_model()
children.bigram_model()
#print Set(n_gram_object.uni_count.values())

children.get_prob(1)
children.get_prob(2)

children_t = n_gram_model(docs_test)
children_t.unigram_model()
children_t.bigram_model()

children_t.get_prob(1)
children_t.get_prob(2)



crime_books = ['/home/tong/Documents/CS5740/Project1/train_books/crime/arsene_lupin.txt',
'/home/tong/Documents/CS5740/Project1/train_books/crime/a_thief_in_the_night.txt',
'/home/tong/Documents/CS5740/Project1/train_books/crime/crime_and_punishment.txt',
'/home/tong/Documents/CS5740/Project1/train_books/crime/the_adventures_of_sherlock_holmes.txt',
'/home/tong/Documents/CS5740/Project1/train_books/crime/the_extraordinary_adventures_of_arsene_lupin.txt',
'/home/tong/Documents/CS5740/Project1/train_books/crime/the_mysterious_affair_at_styles.txt']

crime_test = ['/home/tong/Documents/CS5740/Project1/test_books/crime/the_daffodil_mystery.txt',
'/home/tong/Documents/CS5740/Project1/test_books/crime/the_moon_rock.txt']



if __name__ == '__main__': 
	docs_crime = tokenize(crime_books) 
	docs_crime_test = tokenize(crime_test)



crime = n_gram_model(docs_crime)
crime.unigram_model()
crime.bigram_model()
crime.get_prob(1)
crime.get_prob(2)

crime_t = n_gram_model(docs_crime_test)
crime_t.unigram_model()
crime_t.bigram_model()
crime_t.get_prob(1)
crime_t.get_prob(2)

train = [crime,children]
t_label = ['crime', 'children']

d = genre(train, t_label, children_t, 'children')
d.by_perplexity(6,2)



'''
n_gram_object.plot(1)
s = sentence_generator(n_gram_object)
s.simulate('There is', 'no', 20, 2, output=True)
'''

