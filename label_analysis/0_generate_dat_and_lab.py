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
# Nairobi
label_dir = '00097_21711_nairobi_10m.tif'
data_dir = '/datastore/DATA/classification/SEN2/global_utm/00097_21711_Nairobi/autumn/21711_autumn.tif'

# New York
label_dir ='00010_23083_newyork_10m.tif' 
data_dir  ='/datastore/DATA/classification/SEN2/global_utm/00010_23083_NewYork/autumn/23083_autumn.tif'

f = gdal.Open(label_dir)
dat = f.ReadAsArray()
lab_geoInfoGrid = f.GetGeoTransform()
del f



'''
generate label:
1: commercial buildings
2: industrial buildings
3: residential buildings
4: others, not buildings
'''
# assign a pixel with one class which has the maximum probability
lab = np.argmax(dat,axis=0)+1
# set the buildings without any label as zeros
lab[lab[:]==4]=0
# set the non-building pixels as the class others
sum_tmp = np.sum(dat,axis=0)
lab[sum_tmp[:]==0]=0


'''
patch location indication
'''

patch_label_perc = 1 
patch_size = 32
patch_label_nb_thres = np.round(patch_size * patch_size * patch_label_perc).astype(np.int32)
half_patch = np.array(patch_size/2).astype(np.int8)

# setting overlaping rate of adjacent data patches
shift_perc = 0.1
shift_gap = np.round(patch_size * shift_perc).astype(np.int64)

# indication of where are the labels for the first three classes
lab_tmp = np.logical_and((lab>0), (lab<4))

# find the center points of patches whose labeled contents occupy a percentage higher than setting
lab_tmp_pad = np.pad(lab_tmp,((half_patch,half_patch),(half_patch,half_patch))).astype(np.int32)
lab_idx = np.zeros(lab_tmp_pad.shape).astype(np.int32)
print('Calculating the patch label percentage:')
for i in tqdm(range(-half_patch,half_patch)):
    rotation = np.roll(lab_tmp_pad,i,axis=0)
    for j in range(-half_patch,half_patch):
        lab_idx += np.roll(rotation,j,axis=1)

lab_idx = (lab_idx >= patch_label_nb_thres) * lab_idx
lab_idx = lab_idx[half_patch:-half_patch,half_patch:-half_patch]

nb_patches = np.sum(lab_idx>0)
order_img_coord = np.unravel_index(np.argsort(lab_idx, axis=None), lab_idx.shape)
order_img_coord = np.transpose(np.stack((np.flip(order_img_coord[0]),np.flip(order_img_coord[1])),axis=0))
order_img_coord = order_img_coord[:nb_patches,:]
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

# find corresponding data patches
world_coord = uil.image_coord_2_world_coord(lab_img_coord, label_dir)
dat_img_coord = uil.world_coord_2_image_coord(world_coord, data_dir)

# cut data patches
#dat_patch, dat_in_boundary_index = uil.cut_patch_from_array(lab, dat_img_coord.astype(np.int32), patch_size)
dat_patch, dat_in_boundary_index = uil.cut_patch(data_dir, dat_img_coord.astype(np.int32), patch_size)
lab_patch = lab_patch[dat_in_boundary_index,:,:]

print('Shape of data patch: ')
print(dat_patch.shape)
print('Shape of label patch: ')
print(lab_patch.shape)


'''
save data patches
'''
city = data_dir.split('/')[-3]
file_name = city+'_Patch_'+str(patch_size)+'_LabPerc_'+str(int(patch_label_perc*100))+'.h5'

f = h5py.File(file_name,'w')
f.create_dataset("dat", data=dat_patch)
f.create_dataset("lab", data=lab_patch)
f.close()




