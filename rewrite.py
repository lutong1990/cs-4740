from sets import Set
import sys
import numpy as np
import matplotlib.pyplot as plt 
from operator import itemgetter
import math



class n_gram_model(object):
	'''
	This class implements a n_gram_model
	'''

	def __init__(self, docs):
		'''
		Construct a n-gram model, 
		
			self.docs: a list
				the document (Books, Paper...) to be processed after tokenization
		
			self.uni_count: a dictionary
				unigram of counts
		
			self.bi_count: a dictionary 
				bigram of counts

			self.uni_prob: a dictionary
				univirate corpora of probabilities

			self.bi_prob: a dictionary
				bigram of probabilities
		'''
		self.docs = docs
		self.uni_count = {}
		self.bi_count = {}
		self.uni_prob = {}
		self.bi_prob = {}

	
	def unigram_model(self):
		'''
		convert self.docs to a unigram model of counts
		'''
		wordSet = Set()
		for i, word in enumerate(self.docs):
			if i < len(self.docs):
				if word not in wordSet:
					wordSet.add(word)
					self.uni_count[word] = 1
				else:
					self.uni_count[word] += 1

	def bigram_model(self):
		'''
		convert self.docs to a bigram model of counts
		'''
		wordSet = Set()
		for i, word in enumerate(self.docs):
			if i < len(self.docs)-1:
				if word not in wordSet:
					wordSet.add(word)
					self.bi_count[word] = {}
					self.bi_prob[word] = {}
				next_word = self.docs[i+1]

				if next_word in self.bi_count[word]:
					self.bi_count[word][next_word] += 1
					self.bi_prob[word][next_word] += 1
				else:
					self.bi_count[word][next_word] = 1 	
					self.bi_prob[word][next_word] = 1

	def get_prob(self,n):
		'''
		change the 'value' in corporas from the number of counts into probabilities
		n: the type of models (n=1:unigram; n=2:bigram)
		'''
		if n == 1:
			total_count = sum(self.uni_count.values())
			for key,value in self.uni_count.iteritems():
				self.uni_prob[key] = value/float(total_count)

		elif n == 2:
			for key,value in self.bi_prob.iteritems():
				sub_total = sum(self.bi_prob[key].values())
				for key2,value2 in self.bi_prob[key].iteritems():
					self.bi_prob[key][key2] = value2/float(sub_total)

	def plot(self, n, previous=None):
		'''
		visualize the number of counts, along with distribution curves
		consisting of:
			1. histogram of counts (descending sorted) and a fitted exponential curve
			2. if prob = True, then add the plot of probabilities

		n: integer
			the type of models (unigram, or bigram)
		if n == 2: 'previous' is needed
		'''
		if n == 1:
			xs = self.uni_count.keys()
			ys = self.uni_count.values()

		elif n == 2:
			if previous == None:
				sys.exit('previous word needs to be given')
			xs = self.bi_count[previous].keys()
			ys = self.bi_count[previous].values()

		new = sorted(zip(ys,xs),reverse=True)
		frequency = [y for y, x in new]
		word = [x for y,x in new]

		plt.figure(1)
		
		plt.subplot(121)
		plt.bar(range(len(xs)),ys)
		if n == 2:
			plt.title('plot of unsorted counts after'+': '+ previous)
			plt.xticks(range(len(xs)),xs,rotation=45)
		else:
			plt.title('plot of unsorted counts')
		plt.ylabel('counts')

		plt.subplot(122)
		plt.bar(range(len(word)),frequency)
		if n == 2:
			plt.title('plot of sorted counts after'+': '+ previous)
			plt.xticks(range(len(word)),word,rotation=45)
		else:
			plt.title('plot of sorted counts')
		plt.ylabel('counts')
		plt.show()


	def smoothing(self, ff_upper, method, training_corpora):
		'''
		applying smoothing techniques to the counts data
		This method will return a smoothed training n_gram object
		n: an integer
			1: unigram; 2:bigram
		ff_upper: an integer
			the upper limit of frequency of frequency in the smoothing model, always start from 0
			e.g. to change up to 5, ff_upper needs to be 6
		'''
		smooth = n_gram_model(training_corpora.docs)

		if method == 'GT':
			smooth.unigram_model()
			N_counts = []
			for i in range(ff_upper):
				N = sum(count == i+1 for count in smooth.uni_count.values())
				N_counts.append(N)
			N_zero = 0
			for word in self.docs:
				if word not in smooth.uni_count:
					N_zero += 1
					smooth.uni_count[word] = 0
			N_counts.insert(0,N_zero)
			# start smoothing
			for i in range(ff_upper):
				counts_change =(i+1)*float(N_counts[i+1])/N_counts[i]
				for key, value in smooth.uni_count.iteritems():
					if value == i:
						smooth.uni_count[key] = counts_change	
			smooth.get_prob(1)

			smooth.bigram_model()
			N_counts = []
			for i in range(ff_upper):
				N = 0
				for key, value in smooth.bi_count.iteritems():
					for key2, value2 in smooth.bi_count[key].iteritems():
						if value2 == i + 1:
							N += 1
				N_counts.append(N)
				
			if 0 in N_counts:
				sys.exit("bi_gram models first several ff not continuous")
				
			N_zero = 0
			for i in range(len(self.docs)-1):
				word = self.docs[i]
				next_word = self.docs[i+1]
				if word not in smooth.bi_count:
					N_zero += 1
					smooth.bi_count[word] = {}
					smooth.bi_prob[word] = {}
					smooth.bi_count[word][next_word] = 0
					smooth.bi_prob[word][next_word] = 0
				elif next_word not in smooth.bi_count[word]:
					N_zero += 1
					smooth.bi_count[word][next_word] = 0 
					smooth.bi_prob[word][next_word] = 0 
			N_counts.insert(0, N_zero)
			# start smoothing
			for i in range(ff_upper):
				counts_change =(i+1)*float(N_counts[i+1])/N_counts[i]
				for key, value in smooth.bi_count.iteritems():
					for key2, value2 in smooth.bi_count[key].iteritems():
						if value2 == i:
							smooth.bi_count[key][key2] = counts_change
							smooth.bi_prob[key][key2] = counts_change
			smooth.get_prob(2)
			return smooth


	def get_perplexity(self, n, training_corpora):
		'''
		calculate the perplexity for test data

		training_corpora: a n_gram_model object
			the trained data
		'''
		sum_log = 0
		N = len(self.docs)
		if n == 2: #bigram model
			doc_list = iter(self.docs)
			for word in doc_list:
				sum_log += training_corpora.bi_count[word][next(doc_list)]
			sum_log += training_corpora.uni_count[self.docs[0]]

		elif n == 1: # unigram model
			doc_list = iter(self.docs)
			for word in doc_list:
				sum_log += training_corpora.uni_count[word]

		return math.exp(float(-1)*sum_log/N)


class sentence_generator(object):
	'''
	The class implements random senetence generation
	'''

	def __init__(self, corpora):
		'''
		initialize with an n_gram_model object
		the objects should have at least one full model (e.x. both uni_count and uni_prob)
		'''
		self.corpora = corpora
	
	
	def random_start(self):
		'''
		randomly return the first word of a sentence
		'''
		if self.corpora.bi_count != {}:
			dictionary = self.corpora.bi_count
		else:
			self.corpora.bigram_model()
			dictionary = self.corpora.bi_count

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
			word = self.find_next_word('.',2)
		elif index == 1:
			word = self.find_next_word('?',2)
		else:
			word = self.find_next_word('!',2)

		if word == "\"": #take out words that we do not want it to be a start
			return self.random_start() 
		return word

	def find_next_word(self,word,n):
		'''
		Here I use multinomial distribution to model the word generation
		A little background for multinomial distribution:
		it is a generlization of binomial distribution, 
		in a way that the number of possible outcomes is greater than two. 
		Use of the function:
		input a word -> it will use the probability of all words after it in the corpora, 
						to randomly select a following word
					-> output the following word
		'''
		if n == 1: # unigram
			if self.corpora.uni_prob == {}:
				sys.exit('model type is wrong')
			else: 
				next_word = np.random.multinomial(1,self.corpora.uni_prob.values(),1)
				index = np.transpose(np.nonzero(next_word))[:,1]
				if self.corpora.uni_prob.keys()[index] != word: 
					# make sure it won't give the same word as proceeding word
					return self.corpora.uni_prob.keys()[index]
				else:
					return self.find_next_word(word,1)
		
		elif n == 2: # bigram
			if self.corpora.bi_prob == {}:
				sys.exit('model type is wrong')
			else:
				next_word = np.random.multinomial(1,self.corpora.bi_prob[word].values(),1)
				index = np.transpose(np.nonzero(next_word))[:,1]
				return self.corpora.bi_prob[word].keys()[index]


	def generator(self, sentence, word, n):
		'''
		generate one sentence based on seed and previous_word

		sentence: the beginning of a senetence, or the seed excluding the previous word(s)
		word: the number of previous words based on the model		
		'''
		if len(word) <= 0:
			word = self.random_start()
		new = self.find_next_word(word,n)

		if len(sentence) <= 0:
			return self.generator(word,new,n)
		elif word not in ['.', '?', '!']:
			if word not in [",", ":", ";", "s", "t"]:
				return self.generator(sentence + " " + word,new,n)
			else:
				return self.generator(sentence + word,new,n)
		else:
			return sentence + word


	def simulate(self, sentence, word, number_sentence, model_type, output = False):
		'''
		Start the random sentence generation
		number_sentence: number of sentences to simulate
		output = True: export the sentences to a txt file
		model_type = 1: unigram
		model_type = 2: bigram
		'''
		if output == True:
			f = open('sentence output.txt','w')

		i = 0
		while i < number_sentence:
			result =  str(i+1) + '.' + ' '+ self.generator(sentence, word, model_type)
			if output == True:
				f.write(result + '\n')
			else:
				print result
			i += 1

		if output == True:
			f.close()



class genre(object):
	'''
	The class implements genre classification
	'''

	def __init__(self, corpora_training, corpora_label, corpora_testing, testing_genre):
		'''
		initialize with n_gram_model objects 
		Note: training and testing data need to be the same model, e.x. bigram model

		corpora_training: a list 
			training corporas of different genre
		corpora_label: a list
			the genre of training corporas
		corpora_testing: a corpora
			testing corpora
		train_genre: a string
			the true genre of testing corpora
		'''
		self.train = corpora_training
		self.test = corpora_testing
		self.train_label = corpora_label
		self.test_label = testing_genre

	def by_perplexity(self, upper_counts, n):
		'''
		Genre classification by Comparing Perplexity 
		
		upper_counts: integer
			the max count number to be smoothed
		n: integer
			1: unigram 2: bigram
		'''
		perplexity_scores = []
		for training in self.train:
			smoothed = self.test.smoothing(upper_counts+1,'GT',training)
			score = self.test.get_perplexity(n, smoothed)
			perplexity_scores.append(score)
		index = perplexity_scores.index(min(perplexity_scores))
		classification_label = self.train_label[index]
		print "The classification returns the genre to be: " + classification_label
		print "The true genre is: " + self.test_label






