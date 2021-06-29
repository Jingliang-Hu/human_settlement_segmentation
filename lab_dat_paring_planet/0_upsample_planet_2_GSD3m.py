import glob
from osgeo import gdal, osr
import os

# directory to the folder saving 3m GSD OSM labels
directory_2_osm_3m_label = '../data/lcz_osm_rasters/'
# directory to the folder saving planet data
directory_2_planet_data = '/data/wang/DATA/PLANET/BASEMAP_LCZ1692/'
# directory to save the output upsampled data
directory_2_planet_3m_data_out = '../data/planet/'

# get the available 3m GSD OSM label files and loop over these cities
label_files = glob.glob(directory_2_osm_3m_label+'*_3m.tif')
for idx in range(len(label_files)):
    # get EPSG code in UTM/wgs84
    f = gdal.Open(label_files[idx])
    proj = osr.SpatialReference(wkt=f.GetProjection())
    epsg = proj.GetAttrValue('AUTHORITY',1)
    # use city_code to find the corresponding label and data
    city_code = label_files[idx].split('/')[-1].split('_')[1]
    planet_dat = glob.glob(directory_2_planet_data+'*_'+city_code+'_*/*.tif')
    city_code_name = planet_dat[0].split('/')[-2]

    # planet data are saved in tiles for each city, resample and reproject each of the tile.
    for idx_tile in range(len(planet_dat)):
        tile_name = planet_dat[idx_tile].split('/')[-1].split('.')[0]+'-3m.tif'
        if not os.path.exists(directory_2_planet_3m_data_out+city_code_name):
            os.makedirs(directory_2_planet_3m_data_out+city_code_name)
        command = "gdalwarp -tr 3 3 -r bilinear -t_srs EPSG:"+epsg+' '+planet_dat[idx_tile]+' '+directory_2_planet_3m_data_out+city_code_name+'/'+tile_name
        os.system(command)

