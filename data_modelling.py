import pandas as pd
import numpy as np
import nltk.corpus as ncorp
import nltk
import nltk.stem as stem

listing_data = pd.read_csv('.\Data\listingsdetailed.csv')

# remove unwanted text items from the description
patterns = ['<br /><br />','<br />â€¢','<br />',' <br />','<br /> ','/>']

for s in patterns:
    listing_data['neighborhood_overview'] = listing_data['neighborhood_overview'].str.replace(s,' ')

# remove all punctuation
punc = ['!','(',')','-','[',']','{','}',';',':',"'",'"','\\','<','>','.','/','?','@','#','$','%','^','&','*','_','~',',']

for p in punc:
    listing_data['neighborhood_overview'] = listing_data['neighborhood_overview'].str.replace(p,'')

# update double spaces to single
listing_data['neighborhood_overview'] = listing_data['neighborhood_overview'].str.replace('  ',' ')

# tokenize 
listing_data['tokenized_description'] = listing_data.apply(lambda row: nltk.word_tokenize(str(row['neighborhood_overview'])), axis=1)

# remove STOPWORDS   
stop_words = ncorp.stopwords.words('english')
listing_data['token_description_clean'] = listing_data['tokenized_description'].apply(lambda x: ', '.join([word for word in x if word not in (stop_words)]))

print(listing_data['token_description_clean'].loc[0])

# stemming
listing_data['token_description_clean'] = listing_data['token_description_clean'].apply(lambda x: ', '.join([stem.WordNetLemmatizer().lemmatize(x)]))

# lemmentize words
listing_data['token_description_clean'] = listing_data['token_description_clean'].apply(lambda x: ', '.join([stem.SnowballStemmer(language='english').stem(x)]))

# print(stop_words)
print(listing_data['token_description_clean'].loc[0])

# create world cloud object

# run LDA

# visualise results

# use results

# generate 2nd geomap with results per property cat

