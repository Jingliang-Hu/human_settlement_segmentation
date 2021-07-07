import glob
import os

# Retrieve availabel OSM label files 
osm_label_files = glob.glob('/datastore/exchange/jingliang/human_settlement_segmentation/data/lcz_osm_rasters/*tif.tif')
# Directory to So2Sat S2 database
s2_data_store = '/datastore/DATA/classification/SEN2/global_utm/'
# The percentage of a image patch that are labeled
patch_label_perc_thres = '0.7'
# The size of a image patch
patch_size = '32'
# The overlaping rate of adjacent image patches 
shift_perc = '0.2'


for idx in range(len(osm_label_files)):
    city_code = osm_label_files[idx].split('/')[-1].split('.')[0].split('_')[1]
    label_file = osm_label_files[idx]

    command = 'python 0_generate_dat_and_lab_classification.py '+label_file+' '+s2_data_store+' '+patch_label_perc_thres+' '+patch_size+' '+shift_perc
    os.system(command)









