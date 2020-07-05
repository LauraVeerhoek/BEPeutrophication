import netCDF4
from netCDF4 import Dataset
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib
from sklearn.cluster import KMeans
from gap_statistic import OptimalK
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.io.img_tiles as cimgt
from matplotlib.ticker import LogFormatter
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader
from sklearn.cluster import AgglomerativeClustering
from hdbscan import HDBSCAN
from collections import Counter
from sklearn import preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
import copy
import seaborn as sns
from scipy import stats

matplotlib.use('module://backend_interagg')
matplotlib.use('qt5agg')


shp_path = r"C:\\Users\laura\Downloads\OSPAR_Eutrophication_Pg_NL\OSPAR_Eutrophication_Pg_NL.shp"
nc_path = os.path.abspath(r'MERIS2WAQ_2003_2009_dd_gridV03_chlfa_GF.nc')

dataset = Dataset(nc_path)

data = Dataset(nc_path, mode='r')
time = data.variables['time']
time_dates = netCDF4.num2date(time[:],time.units, only_use_cftime_datetimes=False)
lons = data.variables['lon'][:]
lats = data.variables['lat'][:]
Chlfa = data.variables['Chlfa'][:]
data.close()
del data


waddensea_index_2D = pd.read_csv(r'C:\Users\laura\Data_prelim\waddensea_index_2D.csv')

shapes = lons.shape
lons_flat = lons.flatten()
lons_flat[waddensea_index_2D] = float('NaN')
lats_flat = lats.flatten()
lats_flat[waddensea_index_2D] = float('NaN')
lons = lons_flat.reshape(shapes)
lats = lats_flat.reshape(shapes)


for i in range(1503):
    chlfa_flat = Chlfa[i, :, :].flatten()
    chlfa_flat[waddensea_index_2D] = float('NaN')
    chlfa_reshaped = chlfa_flat.reshape(shapes)
    Chlfa[i, :, :] = chlfa_reshaped

del chlfa_reshaped, chlfa_flat, lons_flat, lats_flat, shapes, waddensea_index_2D
# Spatial subset loop

# region coordinates
ylat_north = 53.75;
ylat_south = 51.5;
xlon_east = 6.5;
xlon_west = 3.5;

chlfa_sub = np.full([int(Chlfa.shape[0]), int(Chlfa.shape[1]), int(Chlfa.shape[2])], np.nan)
lat_sub = np.full([int(Chlfa.shape[1]), int(Chlfa.shape[2])], np.nan)
lon_sub = np.full([int(Chlfa.shape[1]), int(Chlfa.shape[2])], np.nan)
for j in range(0, int(Chlfa.shape[2])):
    for i in range(0, int(Chlfa.shape[1])):
        # if the element by element lat and lons lie within the lat and lons specified for the region, then the indices for each point are saved in idxi and idxj, while the actual data itself is written to the 'domainrun' matrix (previously entirely filled with NaNs). This results in a matrix for the region, containing only the data for that specific subdomain along with NaNs everywhere else

        if lats[i, j] <= ylat_north and lats[i, j] >= ylat_south and lons[i, j] >= xlon_west and lons[
            i, j] <= xlon_east:
            # lat_sub and lon_sub contain the actual lat and lons for the subregion
            lat_sub[i, j] = lats[i, j]
            lon_sub[i, j] = lons[i, j]
            chlfa_sub[:, i, j] = Chlfa[:, i, j]
        else:
            pass


#find peak in data:
means = []
maxs = []
for i in range(chlfa_sub.shape[1]):
    for j in range(chlfa_sub.shape[2]):
        temp = copy.copy(chlfa_sub[:234, i, j])
        tempdf = pd.DataFrame(temp)
        tempdf = tempdf.dropna()
        if tempdf.size >0:
            tempdf = tempdf.reset_index(drop=True)
            maxs.append(np.where(tempdf[0] == np.max(tempdf)[0])[0][0])
    # means.append(np.mean(tempdf)[0])




#plot 17 feb
timestamp = 26
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, chlfa_sub[timestamp,:,:], 60,
             transform=ccrs.PlateCarree(), #locator=matplotlib.ticker.LogLocator(),
             norm=matplotlib.colors.LogNorm(), levels=np.logspace(0, 0.8, 20), cmap='viridis')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
cbar = plt.colorbar(label = 'Chlorophyll-a (mg m-3)', ticks = [0,1,2,3,4,5,6])#, format = LogFormatter(10, labelOnlyBase=False))
cbar.set_ticklabels([0,'','','',''])
plt.title('Chlorophyll-a concentration on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.show()


#plot 22 maart
timestamp = 42
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, chlfa_sub[timestamp,:,:], 60,
             transform=ccrs.PlateCarree(),
             norm=matplotlib.colors.LogNorm(), levels=np.logspace(0, 1.5, 20), cmap='viridis')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
cbar = plt.colorbar(label = 'Chlorophyll-a (mg m-3)', format = LogFormatter(20, labelOnlyBase=False))
# cbar.set_ticks([0, 10])
cbar.minorticks_off()
cbar.set_ticklabels([round(num,2) for num in np.logspace(0, 1.5, 20)[::2]])
plt.title('Chlorophyll-a concentration on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.show()


##Separate coast and offshore:
timestamp = 25
cluster_data = chlfa_sub[timestamp,:,:]
mask = np.isnan(cluster_data)

for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        mask[i,j] = not mask[i,j]

cluster_data = cluster_data[mask]
cluster_data_non_scaled = cluster_data.tolist()
# cluster_data_non_scaled = [item for sublist in cluster_data_non_scaled for item in sublist]
cluster_data = preprocessing.scale(cluster_data.reshape((cluster_data.shape[0], 1)))

lons_flat = lons[mask].tolist()
lats_flat = lats[mask].tolist()

df_clusterdata = pd.DataFrame(list(zip(lons_flat, lats_flat, cluster_data_non_scaled)))
df_clusterdata.columns = ['lon', 'lat', 'chlfa']
df_clusterdata.to_csv(r'C:\Users\laura\Data_prelim\cluster_data.csv', encoding='utf-8', index=False)

#Kmeans

#elbow method & silhouette analysis
inertias = []
silhouettes = []
for i in range(1,10):
    cluster_kmeans = KMeans(n_clusters=i, random_state=0).fit(cluster_data)
    if i!=1:
        silhouette_avg = silhouette_score(cluster_data, cluster_kmeans.labels_)
        silhouettes.append(silhouette_avg)
    inertias.append(cluster_kmeans.inertia_)

plt.plot(range(1,10), inertias)
plt.xlabel('Number of clusters')
plt.ylabel('Total Within Cluster Sum of Squares')
plt.title('Elbow method: k-means on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.grid(True)
plt.show()

plt.plot(range(2,10), silhouettes)
plt.xlabel('Number of clusters')
plt.ylabel('Average silhouette score')
plt.title('Silhouette analysis: kmeans on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.grid(True)
plt.show()

#gap statistic
random.seed(30)
optimalK = OptimalK(parallel_backend='joblib')
optimalK(cluster_data, cluster_array=[2, 3,4,5, 6, 7, 8, 9])
# optimalK.plot_results() #gives optimal k as 1
plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value)
# plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == optimalK.n_clusters].n_clusters,
#             optimalK.gap_df[optimalK.gap_df.n_clusters == optimalK.n_clusters].gap_value,
#             s=250,
#             c="r",
#         )
plt.grid(True)
plt.xlabel("Number of clusters")
plt.ylabel("Gap Statistic")
plt.title("Gap Statistic: kmeans on %s/%s/%s"
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.show()


#clustering kmeans
cluster_kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_data)

result_kmeans_split = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_kmeans_split[mask] = cluster_kmeans.labels_

timestamp = 25
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_kmeans_split, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('spring', 2))
# plt.contour(lon_sub, lat_sub, result_kmeans, 60,
#              transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('spring', 2))
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
# plt.colorbar(ticks=range(2))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('K-means clustering: 2 clusters')
plt.show()
# Counter(cluster_kmeans.labels_)


## hierarchial clustering
cluster_hier = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit_predict(cluster_data)

result_hier = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hier[mask] = cluster_hier

timestamp = 25
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hier, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('spring', 2))
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
# plt.colorbar(ticks=range(2))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Hierarchichal clustering: 2 clusters')
plt.show()
# Counter(cluster_hier) #--> exactly the same as kmeans

##HDBSCAN
clustering_hdbscan = HDBSCAN(min_cluster_size=250, min_samples= 14, gen_min_span_tree=True).fit(cluster_data)

result_hdbscan = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hdbscan[mask] = clustering_hdbscan.labels_

timestamp = 25
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hdbscan, 60,
             transform=ccrs.PlateCarree(), cmap ='jet')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(ticks=range(-1,10))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('HDBSCAN clustering: %s clusters, noise = -1'%9)
plt.show()

#removes noise
hdbscan_without_noise = [float('nan') if x==-1 else x for x in clustering_hdbscan.labels_]
result_hdbscan_without_noise = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hdbscan_without_noise[mask] = hdbscan_without_noise



timestamp = 25
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hdbscan_without_noise, 60,
             transform=ccrs.PlateCarree(), cmap ='jet')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(ticks=range(10))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('HDBSCAN clustering: %s clusters, noise removed'%9)
plt.show()


#Combine clusters
hdbscan_combined = clustering_hdbscan.labels_
hdbscan_combined = [float('nan') if x==-1 else x for x in hdbscan_combined]
hdbscan_combined = [1 if x in [0,1,2,3,4,5,6,8,9] else x for x in hdbscan_combined]
hdbscan_combined = [0 if x==7 else x for x in hdbscan_combined]

result_hdbscan_combined = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hdbscan_combined[mask] = hdbscan_combined



timestamp = 25
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hdbscan_combined, 60,
             transform=ccrs.PlateCarree(), cmap ='spring')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
# plt.colorbar()#ticks=range(2))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('HDBSCAN clustering: %s clusters, noise removed'%21)
plt.show()



hdbscan_combined2 = clustering_hdbscan.labels_
hdbscan_combined2 = [float('nan') if x==-1 else x for x in hdbscan_combined2]
hdbscan_combined2 = [1 if x in [0,1,2,3,4,5,6,8] else x for x in hdbscan_combined2]
hdbscan_combined2 = [0 if x in [9,7] else x for x in hdbscan_combined2]

result_hdbscan_combined2 = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hdbscan_combined2[mask] = hdbscan_combined2



timestamp = 25
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hdbscan_combined, 60,
             transform=ccrs.PlateCarree(), cmap ='spring')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
# plt.colorbar()#ticks=range(2))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('HDBSCAN clustering: %s clusters, noise removed'%21)
plt.show()
# Counter(clustering_hdbscan.labels_) #--> exactly the same as kmeans

## Random forest
# based on all data
RF_proximity_split = pd.read_csv(r'C:\Users\laura\Data_prelim\random_forest_split.csv')
RF_distance_split = 1-RF_proximity_split

RF_cluster_labels_split = AgglomerativeClustering(n_clusters=2, affinity='precomputed', linkage='average').fit_predict(RF_distance_split)

result_RF_split = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_RF_split[mask] = RF_cluster_labels_split


timestamp = 25
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_RF_split, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('spring', 2))
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
# plt.colorbar()#ticks=range(2))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Random Forest clustering %s clusters'%2)
plt.show()
#
# RF_cluster_labels_split3 = AgglomerativeClustering(n_clusters=3, affinity='precomputed', linkage='average').fit_predict(RF_distance_split)
#
# result_RF_split3 = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
# result_RF_split3[mask] = RF_cluster_labels_split3

#
# timestamp = 25
# proj = ccrs.Mercator()
# m = plt.axes(projection=proj)
# stamen_terrain = cimgt.Stamen('terrain-background')
# m.add_image(stamen_terrain, 8)
# # Add coastlines
# m.coastlines(resolution='10m')
# m.add_feature(cfeature.BORDERS.with_scale('10m'))
# shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
#                                 ccrs.PlateCarree(), edgecolor='black')
# m.add_feature(shape_feature, facecolor='none', linewidth=1)
# gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
#                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.xlabels_top = False
# gl.ylabels_right = False
# plt.contourf(lon_sub, lat_sub, result_RF_split3, 60,
#              transform=ccrs.PlateCarree(), cmap ='spring')
# m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
# # plt.colorbar()#ticks=range(2))
# plt.title('on %s/%s/%s'
#           %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
# plt.suptitle('Random Forest clustering %s clusters'%3)
# plt.show()

# removing coast from data

timestamp = 42

cluster_kmeans = KMeans(n_clusters=2, random_state=0).fit(cluster_data)

cluster_kmeans_remove_coast = [float('nan') if x==1 else x for x in cluster_kmeans.labels_]
cluster_kmeans_coast = [float('nan') if x==0 else x for x in cluster_kmeans.labels_]
coast = copy.copy(chlfa_sub[26,:,:])
offshore = copy.copy(chlfa_sub[42,:,:])
temp = copy.copy(chlfa_sub[timestamp,:,:])
temp[mask] = cluster_kmeans_remove_coast
temp2 = copy.copy(chlfa_sub[timestamp,:,:])
temp2[mask] = cluster_kmeans_coast

mask2 = np.isnan(temp)

for i in range(mask2.shape[0]):
    for j in range(mask2.shape[1]):
        mask2[i,j] = not mask2[i,j]

mask3 = np.isnan(temp2)

for i in range(mask3.shape[0]):
    for j in range(mask3.shape[1]):
        mask3[i,j] = not mask3[i,j]

coast = coast[mask3]
offshore = offshore[mask2]
offshore_non_scaled = offshore.tolist()
coast_non_scaled = offshore.tolist()
# cluster_data_non_scaled = [item for sublist in cluster_data_non_scaled for item in sublist]
offshore = preprocessing.scale(offshore.reshape((offshore.shape[0], 1)))
coast = preprocessing.scale(offshore.reshape((offshore.shape[0], 1)))
#
lons_offshore = lons[mask2].tolist()
lats_offshore = lats[mask2].tolist()
lons_coast = lons[mask3].tolist()
lats_coast = lats[mask3].tolist()

df_offshore = pd.DataFrame(list(zip(lons_offshore, lats_offshore, offshore_non_scaled)))
df_offshore = pd.DataFrame(preprocessing.scale(df_offshore))
df_offshore.columns = ['lon', 'lat', 'chlfa']
df_coast = pd.DataFrame(list(zip(lons_coast, lats_coast, coast_non_scaled)))
df_coast = pd.DataFrame(preprocessing.scale(df_coast))
df_coast.columns = ['lon', 'lat', 'chlfa']

df_offshore.to_csv(r'C:\Users\laura\Data_prelim\df_offshore.csv', encoding='utf-8', index=False)



offshore_chlfa = np.full((chlfa_sub.shape[0], chlfa_sub.shape[1], chlfa_sub.shape[2]), np.nan)

for i in range(chlfa_sub.shape[0]):
    offshore_chlfa[i, mask2] = chlfa_sub[i, mask2]


#visualize offshore data
timestamp = 42
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
# shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
#                                 ccrs.PlateCarree(), edgecolor='black')
# m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, offshore_chlfa[timestamp, :, :], 60,
             transform=ccrs.PlateCarree(),
             norm=matplotlib.colors.LogNorm(), levels=np.logspace(0, 1.5, 20), cmap='viridis')#, cmap='jet')
# plt.contour(lon_sub, lat_sub, result_kmeans_offshore,
#              transform=ccrs.PlateCarree(), colors = 'red')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
cbar = plt.colorbar(label = 'Chlorophyll-a (mg m-3)', format = LogFormatter(20, labelOnlyBase=False))
# cbar.set_ticks([0, 10])
cbar.minorticks_off()
cbar.set_ticklabels([round(num,2) for num in np.logspace(0, 1.5, 20)[::2]])
plt.title('on %s/%s/%s  (offshore area)'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Chlorophyll-a concentration')
plt.show()


#Kmeans
cluster_kmeans = KMeans(n_clusters=9, random_state=0).fit(df_offshore)
labels = cluster_kmeans.labels_
for i,j in enumerate(labels):
    if j==0:
        labels[i] = 8
    elif j==1:
        labels[i] = 7
    elif j==2:
        labels[i] = 3
    elif j==3:
        labels[i] = 0
    elif j==4:
        labels[i] = 2
    elif j==5:
        labels[i] = 1
    elif j==6:
        labels[i] = 4
    elif j==7:
        labels[i] = 6
    elif j==8:
        labels[i] = 5


result_kmeans_offshore = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_kmeans_offshore[mask2] = labels

timestamp = 42
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_kmeans_offshore, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('jet', 9))
# plt.contour(lon_sub, lat_sub, result_kmeans_offshore,
#              transform=ccrs.PlateCarree(), colors = 'red')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(ticks=range(9))
plt.title('on %s/%s/%s (offshore area)'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('K-means clustering: 9 clusters')
plt.show()


#hierarcichal

cluster_hier_offshore = AgglomerativeClustering(n_clusters=9, affinity='euclidean',
                                                linkage='ward').fit_predict(df_offshore)
labels = cluster_hier_offshore
for i,j in enumerate(labels):
    if j==0:
        labels[i] = 6
    elif j==1:
        labels[i] = 0
    elif j==2:
        labels[i] = 7
    elif j==3:
        labels[i] = 3
    elif j==4:
        labels[i] = 8
    elif j==5:
        labels[i] = 4
    elif j==6:
        labels[i] = 1
    elif j==7:
        labels[i] = 5
    elif j==8:
        labels[i] = 2
result_hier_offshore = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hier_offshore[mask2] = labels

timestamp = 25
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hier_offshore, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('jet', 9))
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(ticks=range(9))
plt.title('on %s/%s/%s (offshore area)'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Hierarchical clustering: 9 clusters')
plt.show()

#hdbscan
clustering_hdbscan_offshore = HDBSCAN(min_cluster_size=2, min_samples=1, gen_min_span_tree=True).fit(df_offshore)


result_hdbscan_offshore = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hdbscan_offshore[mask2] = clustering_hdbscan_offshore.labels_

timestamp = 42
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hdbscan_offshore, 60,
             transform=ccrs.PlateCarree(), cmap ='jet')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar()#ticks=range(2))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('HDBSCAN clustering: noise = -1')
plt.show()

#removes noise
hdbscan_offshore_without_noise = [float('nan') if x==-1 else x for x in clustering_hdbscan_offshore.labels_]
result_hdbscan_offshore_without_noise = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hdbscan_offshore_without_noise[mask2] = hdbscan_offshore_without_noise



timestamp = 42
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hdbscan_offshore_without_noise, 60,
             transform=ccrs.PlateCarree(), cmap ='jet')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar()#ticks=range(2))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('HDBSCAN clustering: noise removed')
plt.show()


#random forrest offshore

RF_proximity_offshore = pd.read_csv(r'C:/Users/laura/Data_prelim/random forest_offshore_prox')
RF_distance_offshore = 1-RF_proximity_offshore

RF_cluster_labels_offshore = AgglomerativeClustering(n_clusters=9, affinity='precomputed', linkage='average').fit_predict(RF_distance_offshore)
labels = RF_cluster_labels_offshore
for i,j in enumerate(labels):
    if j == 8:
        labels[i] = 1
    elif j==1:
        labels[i] = 8


result_RF_offshore = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_RF_offshore[mask2] = labels

timestamp = 42
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_RF_offshore, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('jet', 9))
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(ticks=range(9))
plt.title('on %s/%s/%s (offshore area)'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Random Forest: %s clusters'%9)
plt.show()


#define subset of chlfa based on coast


coast_chlfa = np.full((chlfa_sub.shape[0], chlfa_sub.shape[1], chlfa_sub.shape[2]), np.nan)

for i in range(chlfa_sub.shape[0]):
    coast_chlfa[i, mask3] = chlfa_sub[i, mask3]


df_coast_2003 = pd.DataFrame(zip(lons_coast,  lats_coast))
df_offshore_2003 = pd.DataFrame(zip(lons_offshore,  lats_offshore))
for i in range(234):
    df_coast_2003['%s'%(i+2)] = chlfa_sub[i, mask3]
    df_offshore_2003['%s'%(i+2)] = chlfa_sub[i,mask2]

df_coast_2003 = preprocessing.scale(df_coast_2003)
df_offshore_2003 = preprocessing.scale(df_offshore_2003)

df_coast_2003_pandas = pd.DataFrame(df_coast_2003[:,[0,1,26+2]])
df_coast_2003_pandas.to_csv(r'C:\Users\laura\Data_prelim\coast.csv', encoding='utf-8', index=False)



#compute metrics

# inertias = []
# silhouettes = []
# for i in range(1,10):
#     cluster_kmeans = KMeans(n_clusters=i, random_state=0).fit(df_coast_2003)
#     if i!=1:
#         silhouette_avg = silhouette_score(df_coast_2003, cluster_kmeans.labels_)
#         silhouettes.append(silhouette_avg)
#     inertias.append(cluster_kmeans.inertia_)
#     # plt.scatter(lons_coast, lats_coast, c=cluster_kmeans.labels_, s=10)
#     # plt.show()
#
# plt.plot(range(1,10), inertias)
# plt.xlabel('Number of clusters')
# plt.ylabel('Total Within Cluster Sum of Squares')
# plt.title('Elbow method: k-means on %s/%s/%s'
#           %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
# plt.grid(True)
# plt.show()
#
# plt.plot(range(2,10), silhouettes)
# plt.xlabel('Number of clusters')
# plt.ylabel('Average silhouette score')
# plt.title('Silhouette analysis: kmeans on %s/%s/%s'
#           %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
# plt.grid(True)
# plt.show()
#
#
# random.seed(30)
# optimalK = OptimalK(parallel_backend='joblib')
# optimalK(df_coast_2003, cluster_array=[2, 3,4,5, 6, 7, 8, 9])
# # optimalK.plot_results() #gives optimal k as 1
# plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value)
# # plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == optimalK.n_clusters].n_clusters,
# #             optimalK.gap_df[optimalK.gap_df.n_clusters == optimalK.n_clusters].gap_value,
# #             s=250,
# #             c="r",
# #         )
# plt.grid(True)
# plt.xlabel("Number of clusters")
# plt.ylabel("Gap Statistic")
# plt.title("Gap Statistic: kmeans on %s/%s/%s"
#           %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
# plt.show()

#visualize coastal area 17/2
timestamp = 25
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, coast_chlfa[timestamp, :, :], 60,
             transform=ccrs.PlateCarree(), cmap='jet')
# plt.contour(lon_sub, lat_sub, result_kmeans_offshore,
#              transform=ccrs.PlateCarree(), colors = 'red')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(label = 'Chlorophyll-a (mg m-3)')#ticks=range(9))
plt.title('on %s/%s/%s offshore'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Chlorophyll-a concentration (coast area)')
plt.show()

#visualize choice of day 25/2
timestamp = 26
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
# shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
#                                 ccrs.PlateCarree(), edgecolor='black')
# m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, coast_chlfa[timestamp, :, :], 60,
             transform=ccrs.PlateCarree())#, cmap='jet')
# plt.contour(lon_sub, lat_sub, result_kmeans_offshore,
#              transform=ccrs.PlateCarree(), colors = 'red')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(label = 'Chlorophyll-a (mg m-3)')#ticks=range(9))
plt.title('on %s/%s/%s  (coastal area)'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Chlorophyll-a concentration')
plt.show()

#Kmeans clustering
timestamp = 26

cluster_kmeans_coast = KMeans(n_clusters=4, random_state=0).fit(df_coast_2003[:,[0,1,timestamp+2]])
labels = cluster_kmeans_coast.labels_
for i,j in enumerate(labels):
    if j==0:
        labels[i] = 2
    elif j==1:
        labels[i] = 0
    elif j==2:
        labels[i] = 1
    elif j==3:
        labels[i] = 3



result_kmeans_coast = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_kmeans_coast[mask3] = labels


proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_kmeans_coast, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('jet', 4))
# plt.contour(lon_sub, lat_sub, result_kmeans_offshore,
#              transform=ccrs.PlateCarree(), colors = 'red')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(ticks=range(4))
plt.title('on %s/%s/%s (coastal area)'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('K-means clustering: 4 clusters')
plt.show()


#Hierarchical clustering
timestamp = 26

cluster_hier_coast = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward').fit_predict(df_coast_2003[:,[0,1,timestamp+2]])
labels = cluster_hier_coast
for i,j in enumerate(labels):
    if j==0:
        labels[i] = 2
    elif j==1:
        labels[i] = 0
    elif j==2:
        labels[i] = 3
    elif j==3:
        labels[i] = 1

result_hier_coast = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hier_coast[mask3] = labels

proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hier_coast, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('jet', 4))
# plt.contour(lon_sub, lat_sub, result_kmeans_offshore,
#              transform=ccrs.PlateCarree(), colors = 'red')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(ticks=range(9))
plt.title('on %s/%s/%s (coastal area)'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Hierarchical clustering: 4 clusters')
plt.show()

#random forest

RF_proximity_coast = pd.read_csv(r"C:/Users/laura/Data_prelim/random_forest_coast")
RF_distance_coast = 1-RF_proximity_coast

RF_cluster_labels_coast = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average').fit_predict(RF_distance_coast)
labels = RF_cluster_labels_coast
for i,j in enumerate(labels):
    if j==0:
        labels[i] = 3
    elif j==1:
        labels[i] = 0
    elif j==2:
        labels[i] = 2
    elif j==3:
        labels[i] = 1

result_RF_coast = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_RF_coast[mask3] = labels

timestamp = 26
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_RF_coast, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('jet', 4))
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(ticks=range(4))
plt.title('on %s/%s/%s (coastal area)'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Random Forest clustering: %s clusters'%4)
plt.show()

#five clusters
RF_cluster_labels_coast5 = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average').fit_predict(RF_distance_coast)
labels = RF_cluster_labels_coast5

for i,j in enumerate(labels):
    if j==0:
        labels[i] = 1
    elif j==1:
        labels[i] = 0
    elif j==2:
        labels[i] = 2
    elif j==3:
        labels[i] = 3
    elif j==4:
        labels[i] = 4


result_RF_coast5 = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_RF_coast5[mask3] = labels

timestamp = 26
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_RF_coast5, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('jet', 5))
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar(ticks=range(5))
plt.title('on %s/%s/%s (coastal area)'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('Random Forest clustering : %s clusters'%5)
plt.show()

#HDBSCAN

clustering_hdbscan_coast = HDBSCAN(min_cluster_size=25, min_samples=1, gen_min_span_tree=True).fit(df_coast_2003[:,[0,1,timestamp+2]])


result_hdbscan_coast = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hdbscan_coast[mask3] = clustering_hdbscan_coast.labels_

timestamp = 26
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hdbscan_coast, 60,
             transform=ccrs.PlateCarree(), cmap ='jet')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar()#ticks=range(2))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('HDBSCAN clustering: noise = -1')
plt.show()

#removes noise
hdbscan_coast_without_noise = [float('nan') if x==-1 else x for x in clustering_hdbscan_coast.labels_]
result_hdbscan_coast_without_noise = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_hdbscan_coast_without_noise[mask3] = hdbscan_coast_without_noise



timestamp = 26
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_hdbscan_coast_without_noise, 60,
             transform=ccrs.PlateCarree(), cmap ='jet')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
plt.colorbar()#ticks=range(2))
plt.title('on %s/%s/%s'
          %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.suptitle('HDBSCAN clustering: noise removed')
plt.show()



## compare to different day
timestamp = 75
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, chlfa_sub[timestamp,:,:], 60,
             transform=ccrs.PlateCarree(),
             norm=matplotlib.colors.LogNorm(), levels=np.logspace(0, 2, 20))
plt.contour(lon_sub, lat_sub, result_RF_offshore,
             transform=ccrs.PlateCarree(), colors='red')
plt.contour(lon_sub, lat_sub, result_RF_coast5,
             transform=ccrs.PlateCarree(), colors = 'red')
plt.contour(lon_sub, lat_sub, result_kmeans_split,
             transform=ccrs.PlateCarree(), colors = 'red')
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
# plt.colorbar()
# plt.title('compared to %s/%s/%s'
#           %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.title('Eutrophication zones \n defined by Random Forest clustering\n compared to %s/%s/%s'
             %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.show()

#distributions


for k in range(9):
    index = np.where(cluster_hier == k)[0]
    temp = np.full(9724, np.nan)
    temp[index] = 1
    temp2 = np.full((chlfa_sub[timestamp, :, :].shape[0], chlfa_sub[timestamp, :, :].shape[1]), np.nan)
    temp2[mask2] = temp

    mask4 = np.isnan(temp2)

    for i in range(mask4.shape[0]):
        for j in range(mask4.shape[1]):
            mask4[i, j] = not mask4[i, j]
    zone = copy.copy(chlfa_sub[:234, mask4]).flatten()
    sns.kdeplot(zone, label='Zone %s'%k)#actual density

# sns.distplot(ROFI, fit=stats.lognorm, kde=False, bins = 100)#fitted with normal distribution
# plt.legend(["data density", "fitted lognorm density"])
# shape, loc, scale = stats.lognorm.fit(ROFI)
# mu, sigma = np.log(scale), shape
plt.title('Distribution of chlorophyll-a concentration in each offshore zone in 2003')
# plt.text(50, 0.2, 'mean: %s \nsigma: %s'%(round(mu,2), round(sigma,2)))
# plt.suptitle('mean: %s, standard deviation: %s'%(round(mu,2), round(sigma,2)))
plt.xlabel("chlfa (mg m^-3)")
plt.ylabel("density")
plt.show()




col_coast = pl.cm.jet(np.linspace(0,1,9))
means_offshore = []
stdevs_offshore = []
for k in range(9):
    index = np.where(labels == k)[0]
    temp = np.full(9724, np.nan)
    temp[index] = 1
    temp2 = np.full((chlfa_sub[timestamp, :, :].shape[0], chlfa_sub[timestamp, :, :].shape[1]), np.nan)
    temp2[mask2] = temp

    mask4 = np.isnan(temp2)

    for i in range(mask4.shape[0]):
        for j in range(mask4.shape[1]):
            mask4[i, j] = not mask4[i, j]
    zone = copy.copy(chlfa_sub[:234, mask4]).flatten()
    shape, loc, scale = stats.lognorm.fit(zone)
    mu, sigma = np.log(scale), shape
    means_offshore.append(mu)
    stdevs_offshore.append(sigma)
    sns.kdeplot(zone, color = col_coast[k], label='Zone %s'%k)#actual density
plt.title('Distribution of chlorophyll-a concentration in each offshore zone in 2003')
plt.xlabel("chlfa (mg m^-3)")
plt.ylabel("density")
plt.show()

means_offshore #
stdevs_offshore #

# to find shape of zone
# index = np.where(labels == 8)[0]
# temp = np.full(9724, np.nan)
# temp[index] = 1
# plt.scatter(lons_offshore, lats_offshore, c=temp)
# plt.show()


#coast distribution
col_coast = pl.cm.jet(np.linspace(0,1,5))
means_coast = []
stdevs_coast = []
for k in range(5):
    index = np.where(labels == k)[0]
    temp = np.full(2198, np.nan)
    temp[index] = 1
    temp2 = np.full((chlfa_sub[timestamp, :, :].shape[0], chlfa_sub[timestamp, :, :].shape[1]), np.nan)
    temp2[mask3] = temp

    mask4 = np.isnan(temp2)

    for i in range(mask4.shape[0]):
        for j in range(mask4.shape[1]):
            mask4[i, j] = not mask4[i, j]
    zone = copy.copy(chlfa_sub[:234, mask4]).flatten()
    shape, loc, scale = stats.lognorm.fit(zone)
    mu, sigma = np.log(scale), shape
    means_coast.append(mu)
    stdevs_coast.append(sigma)
    sns.kdeplot(zone, color = col_coast[k], label='Zone %s'%k)#actual density

plt.title('Distribution of chlorophyll-a concentration in each coastal zone in 2003')
plt.xlabel("chlfa (mg m^-3)")
plt.ylabel("density")
plt.show()
means_coast #[1.3391302736780042,
 # 1.8552757671256035,
 # 1.8637627359223816,
 # 1.869945377485227,
 # 1.6034639112716418]
stdevs_coast #[0.7575650929250966,
 # 0.6520494246671483,
 # 0.5603069438952742,
 # 0.6675007844698606,
 # 0.5839315033918728]

#ROFI zone over time
# for i in range(len(time_dates)):
#     print('%s, %s'%(time_dates[i].strftime("%Y"),i ))
# 234, 468, 717, 955, 1160, 1322

ROFI_kmeans_index = np.where(labels == 8)[0]
temp = np.full(9724, np.nan)
temp[ROFI_kmeans_index] = 1
temp2 = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
temp2[mask2] = temp

mask4 = np.isnan(temp2)

for i in range(mask4.shape[0]):
    for j in range(mask4.shape[1]):
        mask4[i,j] = not mask4[i,j]


ROFI_2003 =  copy.copy(chlfa_sub[:234,mask4]).flatten()
ROFI_2004 =  copy.copy(chlfa_sub[234:468,mask4]).flatten()
ROFI_2005 =  copy.copy(chlfa_sub[468:717,mask4]).flatten()
ROFI_2006 =  copy.copy(chlfa_sub[717:955,mask4]).flatten()
ROFI_2007 =  copy.copy(chlfa_sub[955:1160,mask4]).flatten()
ROFI_2008 =  copy.copy(chlfa_sub[1160:1322,mask4]).flatten()
ROFI_2009 =  copy.copy(chlfa_sub[1322:,mask4]).flatten()
col_rofi = pl.cm.jet(np.linspace(0,1,7))
years = [ROFI_2003, ROFI_2004, ROFI_2005, ROFI_2006, ROFI_2007, ROFI_2008, ROFI_2009]
means_rofi = []
stdevs_rofi = []
for i,zone in enumerate(years):
    sns.kdeplot(zone, label='200%s'%(i+3))#actual density
    shape, loc, scale = stats.lognorm.fit(zone)
    mu, sigma = np.log(scale), shape
    means_rofi.append(mu)
    stdevs_rofi.append(sigma)
plt.title('Distribution of chlorophyll-a concentration in the Rhine ROFI zone over all years')
plt.xlabel("chlfa (mg m^-3)")
plt.ylabel("density")
plt.show()





#final clustering ->  cover image + conclusion image
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
tab10 = cm.get_cmap('tab10', 10)
newcolors = tab10([0,2,4,6,9])
Pastel1 = cm.get_cmap('Pastel1', 9)
babypink = Pastel1([0])
newcolors[0, :] = babypink
newcmp = ListedColormap(newcolors)

RF_cluster_labels_offshore = AgglomerativeClustering(n_clusters=9, affinity='precomputed', linkage='average').fit_predict(RF_distance_offshore)
labels = RF_cluster_labels_offshore
for i,j in enumerate(labels):
    if j == 8:
        labels[i] = 1
    elif j==1:
        labels[i] = 8


result_RF_offshore = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_RF_offshore[mask2] = labels

RF_cluster_labels_coast5 = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average').fit_predict(RF_distance_coast)
labels = RF_cluster_labels_coast5

for i,j in enumerate(labels):
    if j==0:
        labels[i] = 1
    elif j==1:
        labels[i] = 0
    elif j==2:
        labels[i] = 2
    elif j==3:
        labels[i] = 3
    elif j==4:
        labels[i] = 4


result_RF_coast5 = np.full((chlfa_sub[timestamp,:,:].shape[0], chlfa_sub[timestamp,:,:].shape[1]), np.nan)
result_RF_coast5[mask3] = labels



timestamp = 42
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
stamen_terrain = cimgt.Stamen('terrain-background')
m.add_image(stamen_terrain, 8)
# Add coastlines
m.coastlines(resolution='10m')
m.add_feature(cfeature.BORDERS.with_scale('10m'))
shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
                                ccrs.PlateCarree(), edgecolor='black')
m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
plt.contourf(lon_sub, lat_sub, result_RF_offshore, 60,
             transform=ccrs.PlateCarree(), cmap =plt.cm.get_cmap('jet', 9))
# plt.colorbar(ticks = range(9), label = 'Cluster number in offshore area')
# cbar = plt.colorbar()
# cbar.set_ticks([])
plt.contourf(lon_sub, lat_sub, result_RF_coast5, 60,
             transform=ccrs.PlateCarree(), cmap =newcmp)
m.set_extent([xlon_west, xlon_east, ylat_south, ylat_north])
# plt.colorbar(ticks = range(5), label = 'Cluster number in coastal area')
# cbar = plt.colorbar()
# cbar.set_ticks([])
plt.suptitle('Eutrophication zones in the Dutch North Sea\n defined by Random Forest clustering')
# plt.suptitle('Defined Eutrophication zones in the Dutch Continental Shelf')
plt.show()


## OSPAR zones
counter = 0
proj = ccrs.Mercator()
m = plt.axes(projection=proj)
# stamen_terrain = cimgt.Stamen('terrain-background')
# m.add_image(stamen_terrain, 8)
# Add coastlines
# m.coastlines(resolution='10m')
# m.add_feature(cfeature.BORDERS.with_scale('10m'))
# shape_feature = ShapelyFeature(Reader(shp_path).geometries(),
#                                 ccrs.PlateCarree(), edgecolor='black')
for i in Reader(shp_path).geometries():
    if counter ==0: #outhern bight
        southershape = m.add_geometries(i, ccrs.PlateCarree(), facecolor = (0,0,1,0.5),
            edgecolor='black')
    elif counter ==4: #coastal waters
        coastalshape = m.add_geometries(i, ccrs.PlateCarree(), facecolor = (1,0,0,0.5),
            edgecolor='black')
    elif counter == 1:# waddensea
        m.add_geometries(i, ccrs.PlateCarree(), facecolor = (0,1,0,0.5),
                         edgecolor='black')
    elif counter == 2:# Western scheldt
        m.add_geometries(i, ccrs.PlateCarree(), facecolor = (1,1,0,0.5),
                         edgecolor='black')
    elif counter == 3:# Oyster grounds
        m.add_geometries(i, ccrs.PlateCarree(), facecolor = (0,1,1,0.5),
                         edgecolor='black')
    elif counter == 5:# Doggar bank
        m.add_geometries(i, ccrs.PlateCarree(), facecolor = (1,0,1,0.5),
                         edgecolor='black')
    counter = counter + 1
    print(counter)
# m.add_feature(shape_feature, facecolor='none', linewidth=1)
gl = m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=2, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.xlabels_top = False
gl.ylabels_right = False
# plt.contourf(lons, lats, Chlfa[timestamp,:,:], 60,
#              transform=ccrs.PlateCarree(),
#              norm=matplotlib.colors.LogNorm(), levels=np.logspace(0, 2, 20))
m.set_extent([2.5, 6.7, 51.3, 55.8])
# plt.colorbar()
# plt.title('compared to %s/%s/%s'
#           %(time_dates[timestamp].strftime("%d"), time_dates[timestamp].strftime("%m"), time_dates[timestamp].strftime("%Y")))
plt.title('OSPAR eutrophication monitoring zones')
red_patch = matplotlib.patches.Patch(color=(0,0,1,0.5), label='Coastal Waters')
blue_patch = matplotlib.patches.Patch(color=(1,0,0,0.5), label='Southern Bight')
waddensea_patch = matplotlib.patches.Patch(color=(0,1,0,0.5), label='Wadden Sea')
doggar_patch = matplotlib.patches.Patch(color=(1,0,1,0.5), label='Doggar Bank')
oyster_patch = matplotlib.patches.Patch(color=(0,1,1,0.5), label='Oyster Grounds')
wester_patch = matplotlib.patches.Patch(color=(1,1,0,0.5), label='Wester Scheldt')
plt.legend(handles=[red_patch, blue_patch, waddensea_patch, doggar_patch, wester_patch, oyster_patch])
plt.show()