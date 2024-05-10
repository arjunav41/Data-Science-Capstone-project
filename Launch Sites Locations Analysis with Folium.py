#!/usr/bin/env python
# coding: utf-8

# In[1]:


import folium
import pandas as pd


# In[2]:


# Import folium MarkerCluster plugin
from folium.plugins import MarkerCluster
# Import folium MousePosition plugin
from folium.plugins import MousePosition
# Import folium DivIcon plugin
from folium.features import DivIcon


# In[3]:


df = pd.read_csv(r"C:\Users\arjun\Downloads\spacex_launch_geo.csv")


# In[4]:


spacex_df = df[['Launch Site', 'Lat', 'Long', 'class']]


# In[5]:


launch_sites_df = spacex_df.groupby(['Launch Site'], as_index=False).first()


# In[6]:


launch_sites_df = launch_sites_df[['Launch Site', 'Lat', 'Long']]
launch_sites_df


# In[7]:


# Start location is NASA Johnson Space Center
site_map = folium.Map(location=[29.559684888503615, -95.0830971930759], zoom_start=5)
site_map


# In[8]:


# Start location is NASA Johnson Space Center
nasa_coordinate = [29.559684888503615, -95.0830971930759]
site_map = folium.Map(location=nasa_coordinate, zoom_start=5)


# In[9]:


# Create a blue circle at NASA Johnson Space Center's coordinate with a popup label showing its name
circle = folium.Circle(nasa_coordinate, radius=1000, color='#d35400', fill=True).add_child(folium.Popup('NASA Johnson Space Center'))
# Create a blue circle at NASA Johnson Space Center's coordinate with a icon showing its name
marker = folium.map.Marker(
    nasa_coordinate,
    # Create an icon as a text label
    icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' % 'NASA JSC',
        )
    )
site_map.add_child(circle)
site_map.add_child(marker)


# ## Task 1: Mark all launch sites on a map

# In[10]:


formatter = "function(num) {return L.Util.formatNum(num, 3) + ' &deg; ';};"

MousePosition(
    position="topright",
    separator=" Long",
    empty_string="NaN",
    lng_first=False,
    num_digits=20,
    prefix="Coordinates: Lat",
    lat_formatter=formatter,
    lng_formatter=formatter,
).add_to(site_map)

site_map


# In[11]:


# Get list of launch sites
launch_sites = launch_sites_df['Launch Site'].tolist()

# Loop through launch sites and create markers and circles
for site, lat, lon in zip(launch_sites, launch_sites_df['Lat'], launch_sites_df['Long']):
    # Create circle
    circle = folium.Circle(location=[lat, lon], radius=1000, color='#d35400', fill=True).add_child(folium.Popup(site))
    site_map.add_child(circle)

    # Create marker with custom icon
    marker = folium.Marker(
        [lat, lon],
        icon=DivIcon(
        icon_size=(20,20),
        icon_anchor=(0,0),
        html='<div style="font-size: 12; color:#d35400;"><b>%s</b></div>' %site,  # Use a rocket icon from Font Awesome
        ),
        popup=site,  # Set popup text to launch site name
    )
    site_map.add_child(marker)

# Display the map (assuming 'site_map' is initialized with your base map)
site_map


# # Task 2: Mark the success/failed launches for each site on the map

# In[12]:


spacex_df.tail()


# In[13]:


marker_cluster = MarkerCluster()


# In[14]:


def assign_marker_color(row):
    if row['class'] ==1:
        return 'green'
    else:
        return 'red'

spacex_df['marker_color'] = spacex_df.apply(assign_marker_color, axis=1)


# In[15]:


spacex_df


# In[16]:


marker_cluster = MarkerCluster()

# Step 2: Create a MarkerCluster object
marker_cluster = MarkerCluster().add_to(site_map)

# Step 3: Loop through each record in spacex_df and add a marker to the marker cluster
for index, record in spacex_df.iterrows():
    # Extract the latitude and longitude
    lat = record['Lat']
    long = record['Long']
    
    # Customize the marker's icon based on the marker_color
    icon = folium.Icon(color='white', icon_color=record['marker_color'], icon='info-sign')
    
    # Create the marker
    marker = folium.Marker(location=[lat, long], icon=icon)
    
    # Add the marker to the marker cluster
    marker.add_to(marker_cluster)

# Step 4: Add marker cluster to the site map
site_map.add_child(marker_cluster)
site_map


# # TASK 3: Calculate the distances between a launch site to its proximities
# 

# In[17]:


from math import sin, cos, sqrt, atan2, radians

def calculate_distance(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


# In[18]:


launch_site_lat = 34.632834
launch_site_lon = -120.610745
coastline_lat = 34.639
coastline_lon = -120.625
distance_coastline = calculate_distance(launch_site_lat, launch_site_lon, coastline_lat, coastline_lon)
distance_coastline


# In[19]:


coordinate = (coastline_lat, coastline_lon)  # Closest coastline point coordinates

distance_marker = folium.Marker(
    location=coordinate,
    icon=DivIcon(
        icon_size=(40, 40),
        icon_anchor=(0, 0),
        html=f'<div style="font-size: 12; color:#d35400;"><b>{distance_coastline:.2f} KM</b></div>',
    )
)
site_map.add_child(distance_marker)


# In[20]:


lines = folium.PolyLine(
    locations=[[launch_site_lat, launch_site_lon], [coastline_lat, coastline_lon]],
    weight=1
)
site_map.add_child(lines)


# In[21]:


site_map


# In[22]:


Road_launch_site_lat = 28.563197
Road_launch_site_lon = -80.57682
Highway_lat = 28.563
Highway_lon = -80.571
distance_road = calculate_distance(Road_launch_site_lat, Road_launch_site_lon, Highway_lat, Highway_lon)
distance_road


# In[23]:


spacex_df[spacex_df['Launch Site']== 'CCAFS SLC-40']


# In[25]:


H_coordinate = (Highway_lat, Highway_lon)  # Closest coastline point coordinates

distance_marker = folium.Marker(
    location=H_coordinate,
    icon=DivIcon(
        icon_size=(40, 40),
        icon_anchor=(0, 0),
        html=f'<div style="font-size: 12; color:#d35400;"><b>{distance_road:.2f} KM</b></div>',
    )
)
site_map.add_child(distance_marker)


# In[35]:


lines = folium.PolyLine(
    locations=[[Road_launch_site_lat, Road_launch_site_lon], [Highway_lat, Highway_lon]],
    weight=2
)
site_map.add_child(lines)


# In[26]:


spacex_df[spacex_df['Launch Site']=='KSC LC-39A']


# In[27]:


Railway_launch_site_lat = 28.573255
Railway_launch_site_lon = -80.646895
Railway_lat = 28.574
Railway_lon = -80.654
distance_railway = calculate_distance(Railway_launch_site_lat, Railway_launch_site_lon, Railway_lat, Railway_lon)
distance_railway


# In[28]:


R_coordinate = (Railway_lat, Railway_lon)  # Closest coastline point coordinates

distance_marker = folium.Marker(
    location=R_coordinate,
    icon=DivIcon(
        icon_size=(40, 40),
        icon_anchor=(0, 0),
        html=f'<div style="font-size: 12; color:#d35400;"><b>{distance_railway:.2f} KM</b></div>',
    )
)
site_map.add_child(distance_marker)


# In[168]:


lines = folium.PolyLine(
    locations=[[Railway_launch_site_lat, Railway_launch_site_lon], [Railway_lat, Railway_lon]],
    weight=2
)
site_map.add_child(lines)


# In[30]:


City_launch_site_lat = 28.573255
City_launch_site_lon = -80.646895
City_lat = 28.078
City_lon = -80.609
distance_city = calculate_distance(City_launch_site_lat, City_launch_site_lon, City_lat, City_lon)
distance_city


# In[162]:


C_coordinate = (City_lat, City_lon)  # Closest coastline point coordinates

distance_marker = folium.Marker(
    location=C_coordinate,
    icon=DivIcon(
        icon_size=(80, 80),
        icon_anchor=(0, 0),
        html=f'<div style="font-size: 12; color:#d35400;"><b>{distance_city:.2f} KM</b></div>',
    )
)
site_map.add_child(distance_marker)


# In[33]:


lines = folium.PolyLine(
    locations=[[City_launch_site_lat, City_launch_site_lon], [City_lat, City_lon]],
    weight=2
)
site_map.add_child(lines)


# In[36]:


df_city = pd.read_csv(r'C:\Users\arjun\Downloads\City Coordinates.csv')


# In[37]:


df_city.head()


# In[38]:


launch_sites_df


# In[112]:


# Given launch site coordinates
CCAFS_LC_launch_site_lat = 28.573255
CCAFS_LC_launch_site_lon = -80.646895

CCAFS_SLC_launch_site_lat= 28.563197
CCAFS_SLC_launch_site_lon = -80.576820

KSC_LC_launch_site_lat= 28.573255
KSC_LC_launch_site_lon= -80.646895

VAFB_SLC_launch_site_lat= 34.632834
VAFB_SLC_launch_site_lon= -120.610745

df_city['CCAFS_LC-40'] = df_city.apply(lambda row: calculate_distance(CCAFS_LC_launch_site_lat, CCAFS_LC_launch_site_lon, row['lat'], row['lng']), axis=1).round(2)

df_city['CCAFS_SLC-40'] = df_city.apply(lambda row: calculate_distance(CCAFS_SLC_launch_site_lat, CCAFS_SLC_launch_site_lon, row['lat'], row['lng']), axis=1).round(2)

df_city['KSC_LC-39A'] = df_city.apply(lambda row: calculate_distance(KSC_LC_launch_site_lat, KSC_LC_launch_site_lon, row['lat'], row['lng']), axis=1).round(2)

df_city['VAFB_SLC-4E'] = df_city.apply(lambda row: calculate_distance(VAFB_SLC_launch_site_lat, VAFB_SLC_launch_site_lon, row['lat'], row['lng']), axis=1).round(2)


# In[150]:


df_city[df_city['city']== 'Melbourne']


# In[54]:


LC_launch_site_lat = 28.573255
LC_launch_site_lon = -80.646895
M_City_lat = 28.078
M_City_lon = -80.609
M_distance_city = calculate_distance(LC_launch_site_lat, LC_launch_site_lon, M_City_lat, M_City_lon)
M_distance_city


# In[154]:


City_min_distance = df_city[['CCAFS_LC-40', 'CCAFS_SLC-40', 'KSC_LC-39A', 'VAFB_SLC-4E']].min()


# In[155]:


City_min_distance


# In[84]:


launch_sites = launch_sites_df['Launch Site']


# In[85]:


launch_sites = pd.DataFrame(launch_sites)


# In[100]:


df_city.to_csv(r'C:\Users\arjun\Documents\DataScience\Assignment\df_city.csv')


# In[107]:


df_city.info()


# In[119]:


# Calculate the minimum value of the 'CCAFS_LC-40' column
min_value = df_city['CCAFS_LC-40'].min()

# Create a boolean mask where 'CCAFS_LC-40' column is equal to the minimum value
mask = df_city['CCAFS_LC-40'] == min_value

# Use the mask to filter rows from df_city that have the minimum value in 'CCAFS_LC-40'
City_distance_min = df_city[mask]

# Display the rows with minimum distance
print(City_distance_min)


# In[120]:


City_distance_min


# In[142]:


rows_with_min_values


# In[140]:


# Create a boolean mask where both conditions are true
mask = ((df_city['CCAFS_SLC-40'] == 17.92) & (df_city['CCAFS_LC-40'] == 14.45) & (df_city['KSC_LC-39A'] == 14.45) & (df_city['VAFB_SLC-4E'] == 13.15))

# Use the mask to filter rows from df_city that meet both conditions
resulting_rows = df_city[mask]

# Display the resulting rows
print(resulting_rows)


# In[141]:


# Adjusting conditions to use ranges for more flexible matching
mask = (
    df_city['CCAFS_SLC-40'].between(17.9, 17.95) &
    df_city['CCAFS_LC-40'].between(14.4, 14.5) &
    df_city['KSC_LC-39A'].between(14.4, 14.5) &
    df_city['VAFB_SLC-4E'].between(13.1, 13.2)
)

# Filter rows from df_city that meet the adjusted conditions
resulting_rows = df_city[mask]

# Display the resulting rows
print(resulting_rows)


# In[169]:


df_city[df_city['VAFB_SLC-4E']== 13.15]


# In[174]:


distance_city = round(distance_city, 2)


# In[175]:


distance_city


# In[177]:


C_coordinate = (City_lat, City_lon)  # Closest coastline point coordinates

distance_marker = folium.Marker(
    location=C_coordinate,
    icon=DivIcon(
        icon_size=(80, 80),
        icon_anchor=(0, 0),
        html=f'<div style="font-size: 24; color:#d35400;"><b>{distance_city:.2f} KM</b></div>',
    )
)
site_map.add_child(distance_marker)


# In[183]:


VAFB_SLC_site_lon = -120.610745


# In[184]:


lines = folium.PolyLine(
    locations=[[VAFB_SLC_launch_site_lat, VAFB_SLC_site_lon], [City_lat, City_lon]],
    weight=2
)
site_map.add_child(lines)


# In[ ]:




