def bigram_count(filename):
    Bigram = {}
    #prev_word = "START"
    tokenizer = RegexpTokenizer('\w+')    
    with codecs.open(filename, 'r') as f:
        for line in f:
            words = tokenizer.tokenize(line.lower())
            if len(words) > 0:
            	for i, word in enumerate(words):
                #bigram = prev_word +' '+ word
                #i = words.index(word)                 
                	if i < len(words)-1:    
                    	bigram = word +' '+ words[i+1]        
                    	if bigram in Bigram:
                        	Bigram[bigram]+=1
                    	else:
                        	Bigram[bigram]= 1
