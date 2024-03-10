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

random.seed(21)
np.random.seed(21)
# C:\Users\RobertKean\Documents\matplotlib_project\AirbnbEDA\Data
listing_data = pd.read_csv('.\Data\listingsdetailed.csv')
topic_data = pd.read_csv('modelling_output.csv')
# print(len(listing_data.index))

# remove rows with no description to process
listing_data = listing_data[listing_data['neighborhood_overview'].notna()].reset_index(drop=True)
 # print(len(listing_data.index))
print(len(listing_data))
print(len(topic_data))
listing_data.reset_index()
listing_data['topic_number'] = topic_data['0']
print(len(listing_data))
print(listing_data.tail(10))

topics_dict = {
        0.0: 'Transport and Central Ammenities',
        1.0: 'Parks and Gardens',
        2.0: 'Urban and Trendy',
        3.0: 'Shopping and Eating Out',
        4.0: 'Suburbs and Space for Families',
        5.0: 'Museums and Culture'
    }
topics_df = pd.DataFrame(topics_dict.items(), columns=['topic_number', 'Listing Category'])
result_df = listing_data.merge(topics_df, on='topic_number', how="inner")

print(result_df.tail(10))
