import pandas as pd
import numpy as np
import nltk.corpus as ncorp
import nltk
import nltk.stem as stem
from wordcloud import WordCloud
import gensim
import random
import pyLDAvis.gensim_models
import plotly.express as px

# set seed to ensure results do not change significantly between runs
random.seed(21)
np.random.seed(21)

# create listing data DataFrame
listing_data = pd.read_csv('.\Data\listingsdetailed.csv')

# remove rows with no description to process and reset index
listing_data = listing_data[listing_data['neighborhood_overview'].notna()].reset_index(drop=True)

# remove unwanted text items from the description found by visual inspection
patterns = ['<br /><br />','<br />â€¢','<br />',' <br />','<br /> ','/>']

for s in patterns:
    listing_data['neighborhood_overview'] = listing_data['neighborhood_overview'].str.replace(s,' ')

# remove all punctuation from descriptions
punc = ['!','(',')','-','[',']','{','}',';',':',"'",'"','\\','<','>','.','/','?','@','#','$','%','^','&','*','_','~',',','★','➞']

for p in punc:
    listing_data['neighborhood_overview'] = listing_data['neighborhood_overview'].str.replace(p,'')

# update double spaces to single
listing_data['neighborhood_overview'] = listing_data['neighborhood_overview'].str.replace('  ',' ')

# tokenize the description to comma separate the strings
listing_data['tokenized_description'] = listing_data.apply(lambda row: nltk.word_tokenize(str(row['neighborhood_overview'])), axis=1)

# remove STOPWORDS   
stop_words = ncorp.stopwords.words('english')
listing_data['token_description_clean'] = listing_data['tokenized_description'].apply(lambda x: ([word for word in x if word not in (stop_words)]))

# stemming - tested with Snowball Stemmer method but found the results were not desirable
# listing_data['token_description_clean'] = listing_data['token_description_clean'].apply(lambda x: ([stem.SnowballStemmer(language='english').stem(word) for word in x]))

# lemmentize words - this nicely reduces words down to their root
listing_data['token_description_clean'] = listing_data['token_description_clean'].apply(lambda x: ([stem.WordNetLemmatizer().lemmatize(word) for word in x]))

# create word cloud object - this generates a png word cloud image
list_of_words = listing_data['token_description_clean'].apply(lambda x: ', '.join([word for word in x]))
string_of_words = (''.join(list_of_words.to_list()))
wordcloud = WordCloud(collocations=False, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
wordcloud.generate(string_of_words)
wordcloud.to_file('word_cloud.png')

# defining the dictionary with the gensim library and filter out words with 10 fewer occurances and top 5% of words
dictionary = gensim.corpora.Dictionary(listing_data['token_description_clean'])
dictionary.filter_extremes(no_below=10, no_above=0.05, keep_n= 125000)

# convert dictionary to bag-of-words (BoW) corpus (disregards word order) - iterates through all words in text incrementing the frequency count for mutiple instances
bow_corpus = [dictionary.doc2bow(doc) for doc in listing_data['token_description_clean']]

# initalise DataFrame to store topic_data lda model output
topic_data = pd.DataFrame()

# run LDA model unsupervised learning
# run model within following if statement to prevent errors relating to multiple iterations
if __name__ == '__main__':
    lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                        num_topics = 6, 
                                        id2word = dictionary,
                                        random_state=21,                                    
                                        passes = 10,
                                        workers = 2)
    
    # print summary results for each topic number
    for idx, topic in lda_model.print_topics(-1):
        print("Neighborhood_Profile: {} \nWords: {}".format(idx, topic ))
        print("\n")
    
    # Visualisation of word distribution across different topics - this will only work in jupyter labs or equivalent environment
    vis = pyLDAvis.gensim_models.prepare(lda_model, bow_corpus, dictionary,sort_topics=False) #Setting the sort_topics parameter to False to esnure that the topic order remains consistent across different runs
    pyLDAvis.display(vis)

    # Get main topic in each document with greatest probability
    for i, row_list in enumerate(lda_model[bow_corpus]):
        row = row_list        
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic, prob) in (enumerate(row)):
            if j == 0:
                topic_data = pd.concat([topic_data, pd.Series([topic])], axis=0, ignore_index=True)
    
    # add column to listing data df and set it equal to topic number for each row
    listing_data['topic_number'] = topic_data[0]

    # topics dict -> assigning topic categories to each lda topic   
    topics_dict = {
        0.0: 'Transport and Central Ammenities',
        1.0: 'Parks and Gardens',
        2.0: 'Urban and Trendy',
        3.0: 'Shopping and Eating Out',
        4.0: 'Suburbs and Space for Families',
        5.0: 'Museums and Culture'
    }

    # initalise topics DataFrame
    topics_df = pd.DataFrame(topics_dict.items(), columns=['topic_number', 'Listing Category'])

    # inner join listing data and topics data frames to achieve one topic description per row
    # results output to csv file to allow data visualisation updates without the need to re-run lda which takes time
    listing_data = listing_data.merge(topics_df, on="topic_number", how="inner")
    listing_data.drop(['neighborhood_overview', 'tokenized_description','token_description_clean'], axis=1)
    listing_data.to_csv('modelling_output.csv', encoding='utf-8', index=False)

