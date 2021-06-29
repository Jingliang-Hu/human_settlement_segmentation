import glob
import os


osm_label_files = glob.glob('/datastore/exchange/jingliang/human_settlement_segmentation/data/lcz_osm_rasters/*tif.tif')
s2_data_store = '/datastore/DATA/classification/SEN2/global_utm/'


for idx in range(len(osm_label_files)):
    city_code = osm_label_files[idx].split('/')[-1].split('.')[0].split('_')[1]
    label_file = osm_label_files[idx]
    data_file = glob.glob(s2_data_store+'*_'+city_code+'_*/*/*.tif')

    for s2_file in data_file:
        if 'LCZ_results' in s2_file:
            continue
        print('Extract data from: {}'.format(s2_file))
        city = s2_file.split('/')[-3]
        command = 'python 0_generate_dat_and_lab_classification.py '+label_file+' '+s2_file
        os.system(command)
        #command = 'python 1_visual_dat_and_lab.py '+city
        #os.system(command)









