import numpy as np
import plotly.express as px
import pandas as pd

# read in listing data into DataFrame
listing_data = pd.read_csv('.\Data\listingsdetailed.csv')

# update superhost flag to yes/no
listing_data.loc[listing_data['host_is_superhost'] == 't', 'host_is_superhost'] = "Yes"
listing_data.loc[listing_data['host_is_superhost'] == 'f', 'host_is_superhost'] = "No"

# generate mapbox scatter plot filtered by london borough
fig = px.scatter_mapbox(listing_data, lat="latitude", lon="longitude", hover_name="neighbourhood_cleansed", hover_data=["neighbourhood_cleansed", "host_is_superhost", "bathrooms_text", "beds"],
                        color='neighbourhood_cleansed', size_max=1, opacity=0.75, zoom=9, height=600, 
                        labels={
                            "neighbourhood_cleansed": "London Borough",
                            "latitude": "Latitude",
                            "longitude": "Longitude",
                            "host_is_superhost": "Super Host",
                            "bathrooms_text": "Bathrooms",
                            "beds": "Bedrooms"
                        })

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

