import glob
import os

directory_2_10m_osm = "../data/lcz_osm_rasters"
directory_2_3m_osm = "../data/lcz_osm_rasters"

lcz_files = glob.glob(directory_2_10m_osm+'/*.tif.tif')

for idx in range(len(lcz_files)):
    out_file_name = lcz_files[idx].split('/')[-1].split('.')[0]+'_3m.tif'    
    command = "gdalwarp -tr 3 3 "+lcz_files[idx]+" "+directory_2_3m_osm+"/"+out_file_name
    os.system(command)



