# This script imports LondonAirbnb listing data and does the following:
#   1. read data into DataFrame
#   2. cleans and prepares data for visualisation
#   3. generates horizontal bar plot of average price per night for each London borough
#   4. generates a grouped bar plot for comparing review scores across different categories for superhosts vs non-superhosts

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (FormatStrFormatter) 

# read in detailed listings data
listing_data_detailed = pd.read_csv('.\Data\listingsdetailed.csv')

# create detailed listings dataframe
listing_data_detailed_df = pd.DataFrame(listing_data_detailed)

# update neighbourhood column name
listing_data_detailed_df = listing_data_detailed_df.rename(columns={"neighbourhood_cleansed": "London Borough"})

# create a new dataframe containing only borough and price columns
borough_price_comparison_df = listing_data_detailed_df[['London Borough', 'price']].copy()

# remove thousands separator from price data
borough_price_comparison_df['price'] = borough_price_comparison_df['price'].replace(',','', regex=True)

# remove currency symbol from price data and update the dtype to float
borough_price_comparison_df['price'] = borough_price_comparison_df['price'].str.removeprefix('$').astype(float)

# create a new dataframe to store average price for each borough
chart_data_df = borough_price_comparison_df.groupby('London Borough', as_index=False).price.mean()

# update the price column name
chart_data_df = chart_data_df.rename(columns={'price': 'Average Price Per Night'})

# sort in ascending order by price
chart_data_df.sort_values(by=['Average Price Per Night'], ascending=True, inplace=True, ignore_index=True)

# generate series of average prices
price_hieght_y = chart_data_df.iloc[:,1]

# generate series of boroughs
boroughs_bars_x = chart_data_df.iloc[:,0]

fig, (price_barh, reviews_bar) = plt.subplots(1, 2)

# create barh plot
hbars = price_barh.barh(boroughs_bars_x, price_hieght_y, color='#66cc99')

# format bar labels displaying rounded average price
price_barh.bar_label(hbars, fmt='{:.0f}', padding=5, fontsize=7)

# set chart title
price_barh.set_title('London Borough Average Airbnb Price Per Night ')

# remove axis splines 
for spine in ['top', 'bottom', 'left', 'right']:
    price_barh.spines[spine].set_visible(False)

# remove x and y axis ticks
price_barh.xaxis.set_ticks_position('none')
price_barh.yaxis.set_ticks_position('none')

# format axis tick markers format
price_barh.xaxis.set_tick_params(labelsize=10, pad=4)
price_barh.yaxis.set_tick_params(labelsize=6, pad=10)

# add x axis label
price_barh.set_xlabel('($) Average Price Per Night')

# add gridlines
price_barh.grid(color='grey', which='major', axis='x', linewidth=0.2)

# --- do super hosts achieve better reviews than non-super hosts across different review categories ---

# create a new dataframe containing only superhost flag and review columns
review_data_df = listing_data_detailed_df[['host_is_superhost', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value']].copy()

# remove any rows with no superhost flag or review scorces
review_data_df = review_data_df.dropna()

# reset the index of dataframe after dropping rows
review_data_df.reset_index(inplace=True)

# group data for each superhost flag calculate the mean for each review category
grouped_review_data = review_data_df.groupby('host_is_superhost', as_index=False).agg({'review_scores_cleanliness': 'mean', 'review_scores_checkin': 'mean', 'review_scores_communication': 'mean','review_scores_location': 'mean','review_scores_value': 'mean'})

# rename column names to display names for chart
grouped_review_data = grouped_review_data.rename(columns={'review_scores_cleanliness': 'Cleanliness', 'review_scores_checkin': 'Check In', 'review_scores_communication': 'Communication','review_scores_location': 'Location','review_scores_value': 'Value'})

# generate dictionary from DataFrame
review_dict = grouped_review_data.set_index('host_is_superhost').T.to_dict('list')

# rename superhost true
review_dict['Non-Super Host'] = review_dict['f']
del review_dict['f']

# rename superhost false
review_dict['Super Host'] = review_dict['t']
del review_dict['t']

# list of review categories
review_cats = ['Cleanliness', 'Check In', 'Communication','Location','Value']

# reviews_bar
# x sets positions of bars on axis
x = np.arange(len(review_cats))
width = 0.2
counter = 1

# lamda function to set colour for series
bar_color = lambda x : "#ffdf80" if(x=='Super Host') else "#ff6666"

# for each superhost flag t/f plot bars for each review category
for flag, review_score in review_dict.items():
    bar = reviews_bar.bar(x + counter * width, review_score, width, label=flag, color = bar_color(flag))
    reviews_bar.bar_label(bar, padding = 3, fontsize=5, fmt='{:,.2f}')
    counter += 1

# set position of x ticks on x axis
reviews_bar.set_xticks(x + width + 0.1, review_cats, fontsize=8)
reviews_bar.xaxis.set_ticks_position('none')
# set y axis to start at 4.0 review score
reviews_bar.set_ylim(ymin=4)

# add gridlines across the chart
reviews_bar.grid(color='grey', which='major', axis='y', linewidth=0.2)

# format y axis number labels
reviews_bar.yaxis.set_major_formatter(FormatStrFormatter('% 1.1f'))

# set y axis title
reviews_bar.set_ylabel('Average Review Score', fontsize=10)

# set x axis title
reviews_bar.set_xlabel('Review Category')

# set chart title
reviews_bar.set_title('Average Scores Super Hosts vs Non-Super Hosts')

# set position of legend
reviews_bar.legend(loc='upper center', ncols=2)

# remove axis spines from all sides
for spine in ['top', 'bottom', 'left', 'right']:
    reviews_bar.spines[spine].set_visible(False)

# show plot
plt.subplots_adjust(right=0.97,\
                    left=0.03,\
                    bottom=0.03,\
                    top=0.97,\
                    wspace=0.1,\
                    hspace=0.1)
plt.tight_layout()
plt.show()