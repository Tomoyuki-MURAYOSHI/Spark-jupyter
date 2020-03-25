#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import folium


# In[2]:


d1=gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
gdf = d1.copy()
pdf = pd.DataFrame(gdf)
gdf


# In[3]:


d1.plot()


# In[5]:


url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
state_geo = f'{url}/us-states.json'
state_unemployment = f'{url}/US_Unemployment_Oct2012.csv'
state_data = pd.read_csv(state_unemployment)


# In[6]:


state_geo


# In[7]:


state_data


# In[9]:


gdf2 = gdf.copy()
gdf2["value"] = np.random.randint(20, size=(gdf.index.size))

data = pd.DataFrame(
                   data={
                       "nation": gdf.iso_a3,
                       "value": np.random.randint(20, size=(gdf.index.size))
                   })


m = folium.Map()
folium.Choropleth(
    geo_data=gdf.to_json(),
    data=data,
    key_on="feature.properties.iso_a3",
    columns=["nation", "value"],
    fill_color="OrRd",
    legend_name='Value(int)',
    name="Choropleth test",
    highlight=False,
).add_to(m)
folium.LayerControl().add_to(m)
m


# In[8]:


gdf2 = gdf.copy()
gdf2["value"] = np.random.randint(20, size=(gdf.index.size))

data = pd.DataFrame(
                   data={
                       "nation": gdf.iso_a3,
                       "value": np.random.randint(20, size=(gdf.index.size))
                   })


m = folium.Map()
folium.Choropleth(
    geo_data=gdf.to_json(),
    data=data,
    key_on="feature.properties.iso_a3",
    columns=["nation", "value"],
    fill_color="OrRd",
    legend_name='Value(int)',
    name="Choropleth test",
    highlight=True,
).add_to(m)
folium.LayerControl().add_to(m)
m


# - popupを作るのは依然として難しい
# - 今全く触ってないけどgeoviewsとかも面白そう
#     - そういうツールの方が柔軟に色々出来るかもしれない

# In[45]:


import json


# In[49]:


state_geo


# In[50]:


pd.read_json(state_geo)


# In[60]:


display(pd.read_json(gdf2.to_json()).iloc[0, 1])
pd.read_json(gdf2.to_json())


# In[57]:


url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'
state_geo = f'{url}/us-states.json'
state_unemployment = f'{url}/US_Unemployment_Oct2012.csv'
state_data = pd.read_csv(state_unemployment)

m = folium.Map(location=[48, -102], zoom_start=3)

folium.Choropleth(
    geo_data=state_geo,
    name='choropleth',
    data=state_data,
    columns=['State', 'Unemployment'],
    key_on='feature.id',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Unemployment Rate (%)'
).add_to(m)

folium.LayerControl().add_to(m)

m


# In[ ]:




