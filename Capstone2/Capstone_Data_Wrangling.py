# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
#import cdsapi
import pandas as pd
#import netcdf4
import xarray as xarr # pandas based library for 
            # labeled data with N-D tensors at each dimensions
import matplotlib.pyplot as plt
# %matplotlib inline 
import cartopy
import cartopy.crs as ccrs
import numpy as np
import geopandas
import salem
import pandas_profiling

# %%
# Read the data Path where it is stored on the Computer
#data_dir = input('Path to the data\n')
data_dir = "C:\\Users\\kurt_\\Data\\crop_climotology\\"

# %%
# Import data as xarray dataset from the directory
dask = True
if dask:
    # Import with dask
    clim = xarr.open_mfdataset(data_dir+'*.nc', parallel=True, 
                              combine='by_coords', chunks={'time': 50}
                             , engine='netcdf4')
    print(f'The chunk size for time dimension is {clim.chunks["time"][0]}\n')
    print(f'dataset, thus, have {len(clim.time)/clim.chunks["time"][0]} chunks')
else:
    # Import without dask for debugging
    clim = xarr.open_mfdataset(data_dir+'*.nc', parallel=False, 
                          combine='by_coords', engine='netcdf4')

# %%
#print(clim.data_vars)
#print(clim.coords)
clim

# %%
# The shape of a variable
mnt_sub.TG

# %%
for var in clim:
    print(f'Variables in the data: {clim[var].attrs}')
# Let's select the first time step and plot the 2m-air temperature

# Let's check the dimensions
for dim in clim.dims:
    dimsize = clim.dims[dim]
    print(f'\nData has {dimsize} {dim} ')
    if dim == 'latitude':
        print(f' latitudes: from {float(clim[dim].min())} degree South',
     f'to {float(clim[dim].max())} degree North')
    if dim == 'longitude':
        print(f' Longitudes: from {float(clim[dim].max())} degree East',
     f'to {float(clim[dim].min())} degree West')
    if dim == 'time':
        print(f'time: from {pd.to_datetime(clim["time"].min().values)} to {pd.to_datetime(clim["time"].max().values)} ')

# %% [markdown]
# ### Alfalfa Hay
# Alfalfa hay is produced mostly in North-Western States. Among them it is produced throughout all Montana and in most of the Idaho which makes them more convenient for agroclimatic analysis.

# %% [markdown]
# * Masking climate data only to keep the relavant states using **Salem**
# * Geospatial data for the state boundaries are from US Census
# * Let's examine the shape file for US States using __Geopandas__
#
# Here is the map that shows where Alfala hay is produced
# Source: https://www.nass.usda.gov/Charts_and_Maps/Crops_County/al-ha.php
#
# ![Alt text](https://www.nass.usda.gov/Charts_and_Maps/graphics/AL-HA-RGBChor.png)

# %%
# Let's read the geospatial data for the states
path = 'C:\\Users\\kurt_\\Data\\usstates\\'
geo_usa = geopandas.read_file(path)
print(type(geo_usa))
print('The coordinate Reference System Info:')
print(geo_usa.crs)
geo_usa.head()

# %%
# Let's see the state boundaries on a map to see
# if there is an error

# Getting rid of oversees territories from the map
geo_usa = geo_usa[geo_usa.STATEFP.apply(lambda x: int(x)) < 60]
#Let's remove the Alaska too
geo_usa = geo_usa[geo_usa.NAME != 'Alaska']
print('Please Double Click The Map To Zoom In')
fig,ax = plt.subplots(figsize=(12, 10))
geo_usa.plot(ax=ax, cmap='OrRd')
ax.set_xlim(-127,-65)
ax.set_ylim(17,55)
geo_usa.apply(lambda x: ax.annotate(s=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=7),axis=1);
#plt.tight_layout()
plt.show()
plt.close()

# %%
# Plotting a random time step jus to see the data on a map
alfala_states = ['Montana', 'Idaho']
fig = plt.figure(figsize=(10, 8))
# plotting on a map using cartopy
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.add_feature(cartopy.feature.RIVERS)
ax.add_feature(cartopy.feature.STATES)

# plotting using xarray plot method
# Montana and Idaho For Alfala Barley
MT_coord = salem.read_shapefile(path+'cb_2018_us_state_500k.shp')
MT_coord = MT_coord[(MT_coord.NAME.isin(alfala_states))]
mnt_sub = clim.salem.subset(shape=MT_coord, margin=10)
MT_coord.apply(lambda x: ax.annotate(s=x.NAME, xy=x.geometry.centroid.coords[0], ha='center', fontsize=7),axis=1);
# Let's plot the daily average tempreture on a random time
randm_day = mnt_sub['TG'].isel( time=np.random.randint(len(mnt_sub.time)))
randm_day.salem.roi(shape=MT_coord).plot(ax=ax)
#Montana_anm = clim_loc['TG'].isel( time=np.random.randint(len(clim_loc.time))) - clim_loc['TG'].mean(dim='time')
#Montana_anm.plot(ax=ax)
plt.show()
plt.close()

# %%
mnt_sub.DTR

# %% [markdown]
# #### Now Reading the Hay Yield Data For Idaho and Montana
# Data link: https://quickstats.nass.usda.gov/results/347988B6-8746-305D-9147-D1A31FE09FD2

# %%
hay = pd.read_csv("C:\\Users\\kurt_\\Data\\HAY.csv")
#hay.dropna(axis=1, inplace=True)
hay.info()

# %%
hay.dropna(axis=1, inplace=True)
print(hay.info())
hay.head(3)

# %%
hay['Year'].value_counts()

# %%
#Looks like we have 6 values for 1999 while only 2 expected
# Let's see what is going on there
print(hay[hay['Year'] == 1999])
# Looks like the problem is this years forecasts are in the data

# %%
# Let's get rid of the rows where Period is smth other than Year
period_to_rid = set(hay['Period']).difference(['YEAR'])
Period_rows = hay['Period'].isin(period_to_rid)
hay = hay[~Period_rows]
hay['Period'].value_counts()

# %%

# %%
hay.profile_report()

# %%

# %%
#Turkey
#clim_loc = clim.where((clim.lat > 30) & (clim.lat < 50) & (clim.lon >20 ) & (clim.lon < 45), drop=True)
#shdf = salem.read_shapefile(salem.get_demo_file('world_borders.shp'))
#shdf = shdf.loc[shdf['CNTRY_NAME'] == 'Turkey']
