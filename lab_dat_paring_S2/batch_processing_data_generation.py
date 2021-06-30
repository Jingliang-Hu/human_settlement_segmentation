import glob
import os


osm_label_files = glob.glob('/datastore/exchange/jingliang/human_settlement_segmentation/data/lcz_osm_rasters/*tif.tif')
s2_data_store = '/datastore/DATA/classification/SEN2/global_utm/'

patch_label_perc_thres = '0.7'
patch_size = '32'
shift_perc = '0.2'


for idx in range(len(osm_label_files)):
    city_code = osm_label_files[idx].split('/')[-1].split('.')[0].split('_')[1]
    label_file = osm_label_files[idx]

    command = 'python 0_generate_dat_and_lab_classification.py '+label_file+' '+s2_data_store+' '+patch_label_perc_thres+' '+patch_size+' '+shift_perc
    os.system(command)









