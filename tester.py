import geopandas as gpd
import rasterio

# Read the 'Stations' shapefile
stations = gpd.read_file(r'F:\GeoDelta\Extracting Elevation Data from DEM to Points\stations.shp')

# Create an empty column for elevation values
stations['elevations'] = 0

for index, row in stations.iterrows():
 # Extracting attriburtes from the stations shapefile
 name = row['name']
 latitude = row['geometry'].y
 longitude = row['geometry'].x
 
 # Reading the DEM using rasterio 
 dem = rasterio.open(r'F:\GeoDelta\Extracting Elevation Data from DEM to Points\DEM.tif')
 row, col = dem.index(longitude, latitude)
 dem_data = dem.read(1)
 
 print('The elevation of '+ name + ": " + str(dem_data[row,col]) + ' meters')
 
 # Add the elevation values to the 'stations' geodataframe
 stations['elevations'].loc[index] = dem_data[row,col]
 
# Save the infomation into a csv file 
elevations = stations[['name', 'elevations']]
elevations.to_csv('topographic_elevations.csv')