import pandas as pd
import plotly.express as px

# read in 'data_modelling.py' results from lda topic modelling exercise
result_df = pd.read_csv('.\modelling_output.csv')

# update superhost flag data to yes/no
result_df.loc[result_df['host_is_superhost'] == 't', 'host_is_superhost'] = "Yes"
result_df.loc[result_df['host_is_superhost'] == 'f', 'host_is_superhost'] = "No"

# generage mapbox scatter plot for each listing filtered by listing category
fig = px.scatter_mapbox(result_df, lat="latitude", lon="longitude", hover_name="neighbourhood_cleansed", hover_data=["neighbourhood_cleansed", "Listing Category", "host_is_superhost", "bathrooms_text", "beds"],
                        color='Listing Category', size_max=1, opacity=0.75, zoom=9, height=600, 
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