import os
import sys
sys.path.append(os.path.abspath("../src/uil"))
import h5py
import numpy as np
import geo_uil as uil

from osgeo import gdal
from tqdm import tqdm

'''
load 10 meter GSD label
'''
label_dir = '../data/lcz_osm_rasters/00085_206167_sydney_3m.tif'
label_dir = sys.argv[1]

data_dir = '../data/planet/00085_206167_Sydney'
data_dir = sys.argv[2]
city = data_dir.split('/')[-1]
print('Data of {}'.format(city))

f = gdal.Open(label_dir)
dat = f.ReadAsArray()
lab_geoInfoGrid = f.GetGeoTransform()
del f

'''
generate label:
1: commercial buildings
2: industrial buildings
3: residential buildings
4: others buildings
'''
# assign a pixel with one class which has the maximum probability
dat = np.transpose(dat,[1,2,0])
lab = np.argmax(dat,axis=2)+1
# set the non-building pixels as the class others
mask = np.sum(dat,axis=2)>0
lab[mask==0]=0


'''
patch location indication
'''
patch_label_perc_thres = sys.argv[3]
patch_label_perc_thres = 0.8 

patch_size = sys.argv[4]
patch_size = 128
half_patch = np.array(patch_size/2).astype(np.int8)

# setting overlaping rate of adjacent data patches
shift_perc = sys.argv[5]
shift_perc = 0.2
shift_gap = np.round(patch_size * shift_perc).astype(np.int64)

# calculating the labeling percentage of each data patch
lab_perc = uil.patch_labeling_percentage(mask, patch_size)
lab_idx = lab_perc >= patch_label_perc_thres

# find the image coordinate
nb_patches = np.sum(lab_idx>0)
order_img_coord = np.unravel_index(np.argsort(lab_idx, axis=None), lab_idx.shape)
order_img_coord = np.transpose(np.stack((np.flip(order_img_coord[0]),np.flip(order_img_coord[1])),axis=0))
order_img_coord = order_img_coord[:nb_patches,:]


# get rid of data patches that are too closed to each other
i=0
while i<order_img_coord.shape[0]:
    neibor_south_idx = np.logical_and((order_img_coord[:,0]>order_img_coord[i,0]), (order_img_coord[:,0]<order_img_coord[i,0]+shift_gap))
    neibor_north_idx = np.logical_and((order_img_coord[:,0]<order_img_coord[i,0]), (order_img_coord[:,0]>order_img_coord[i,0]-shift_gap))
    neibor_west_idx  = np.logical_and((order_img_coord[:,1]<order_img_coord[i,1]), (order_img_coord[:,1]>order_img_coord[i,1]-shift_gap))
    neibor_east_idx  = np.logical_and((order_img_coord[:,1]>order_img_coord[i,1]), (order_img_coord[:,1]<order_img_coord[i,1]+shift_gap))
    idx = np.logical_or(neibor_south_idx, np.logical_or(neibor_north_idx, np.logical_or(neibor_west_idx, neibor_east_idx)))
    order_img_coord = np.delete(order_img_coord, idx, axis=0)
    i += 1


'''
cut patches
'''
# cut label patches
_, lab_in_boundary_index = uil.cut_patch(label_dir, order_img_coord, patch_size)
lab_img_coord = order_img_coord[lab_in_boundary_index,:]
lab_patch, _ = uil.cut_patch_from_array(lab, lab_img_coord.astype(np.int32), patch_size)

# find corresponding data patches in multiple planet tiles
world_coord = uil.image_coord_2_world_coord(lab_img_coord, label_dir)

data_tiff_files = glob.glob(data_dir+'/*-3m.tif')
dat_patch_list = []
lab_patch_list = []
for idx in range(len(data_tiff_files)):
    dat_img_coord = uil.world_coord_2_image_coord(world_coord, data_tiff_files[idx])
    # cut data patches
    #dat_patch, dat_in_boundary_index = uil.cut_patch_from_array(lab, dat_img_coord.astype(np.int32), patch_size)
    dat_patch_tile, dat_in_boundary_index = uil.cut_patch(data_dir, dat_img_coord.astype(np.int32), patch_size)
    lab_patch_tile = lab_patch[dat_in_boundary_index,:,:]
    dat_patch_list.append(dat_patch_tile)
    lab_patch_list.append(lab_patch_tile)

dat_patch = np.concatenate(dat_patch_list[:],axis=0)
del dat_patch_list 
lab_patch = np.concatenate(lab_patch_list[:],axis=0)
del lab_patch_list

print('Shape of data patch: {}'.format(dat_patch.shape))
print('Shape of label patch: {}'.format(lab_patch.shape))

'''
save data patches
'''
out_directory = '../data/planet_train_and_test/'+city
if not os.path.exists(out_directory):
    os.makedirs(out_directory)

file_name = out_directory+'/PatchSz'+str(patch_size)+'_LabPerc'+str(int(patch_label_perc_thres*100))+'.h5'

f = h5py.File(file_name,'w')
f.create_dataset("dat", data=dat_patch)
f.create_dataset("lab", data=lab_patch)
f.close()




