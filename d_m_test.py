import pandas as pd
import gensim
import nltk
import random
import numpy as np

random.seed(21)
np.random.seed(21)
sr = pd.Series (['word location culture sport media bike football runnning food',
        'hotel holiday location study bathroom dinner sleep'])

dataframe = pd.DataFrame({'docs': sr})

print(dataframe)
dataframe['docs'] = dataframe.apply(lambda row: nltk.word_tokenize(str(row['docs'])), axis=1)

print(dataframe)

dictionary = gensim.corpora.Dictionary(dataframe['docs'])
#dictionary.filter_extremes(no_below=10, no_above=0.05, keep_n= 125000)

# convert dictionary to bag-of-words (BoW) corpus - iterates through all words in text incrementing the frequency count for mutiple instances
bow_corpus = [dictionary.doc2bow(doc) for doc in dataframe['docs']]

print(bow_corpus)
topic_data = pd.DataFrame()
if __name__ == '__main__':
    lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                        num_topics = 2, 
                                        id2word = dictionary,
                                        random_state=21,                                    
                                        passes = 10,
                                        workers = 2)
    
    for idx, topic in lda_model.print_topics(-1):
        print("Neighborhood_Profile: {} \nWords: {}".format(idx, topic ))
        print("\n")


    for i, row_list in enumerate(lda_model[bow_corpus]):
        row = row_list        
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic, prob) in (enumerate(row)):
            if j == 0:
                topic_data = pd.concat([topic_data, pd.Series([topic])], axis=0, ignore_index=True)
        

    dataframe = pd.concat([dataframe, topic_data], axis=1, ignore_index=True)
    dataframe = dataframe.rename(columns={1: 'topic_number'})
    print(dataframe)
                                 

