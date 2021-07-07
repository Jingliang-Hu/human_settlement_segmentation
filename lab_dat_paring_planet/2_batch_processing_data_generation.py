import glob
import os


osm_label_files = glob.glob('../data/lcz_osm_rasters/*.tif.tif')
planet_data_store = '../data/planet/'

patch_label_percentage = '0.7'
data_patch_size = '128'
stride_percentage = '0.2'


for idx in range(len(osm_label_files)):
    city_code = osm_label_files[idx].split('/')[-1].split('_')[1]
    label_file = osm_label_files[idx]
    data_file = glob.glob(planet_data_store+'*_'+city_code+'_*')

    command = 'python 1_lab_dat_paring_v2.py '+label_file+' '+data_file[0]+' '+patch_label_percentage+' '+data_patch_size+' '+stride_percentage
    os.system(command)
    #command = 'python 1_visual_dat_and_lab.py '+city
    #os.system(command)









