import os
import sys
sys.path.append(os.path.abspath("../src/uil"))
import h5py
import numpy as np
import geo_uil as uil
import glob
from osgeo import gdal
from tqdm import tqdm

'''
Input arguments
'''
# Directory to label data of a city
label_dir = sys.argv[1]

# List of directories to sentinel-2 data of a city
data_store = sys.argv[2]

# labeled percentage of a data patch
patch_label_perc_thres = float(sys.argv[3])

# data patch size
patch_size = int(float(sys.argv[4]))
half_patch = np.array(patch_size/2).astype(np.int8)

# overlaping rate of adjacent data patches
shift_perc = float(sys.argv[5])
shift_gap = np.round(patch_size * shift_perc).astype(np.int64)

'''
Load label data
'''
city = label_dir.split('/')[-1].split('.')[0]
print('Data of city: {}'.format(city))

f = gdal.Open(label_dir)
dat = f.ReadAsArray()
lab_geoInfoGrid = f.GetGeoTransform()
del f



'''
label:
1: commercial buildings
2: industrial buildings
3: residential buildings
4: others buildings

Operations on label data
'''
# assign a pixel with one class which has the maximum probability
dat = np.transpose(dat,[1,2,0])
lab = np.argmax(dat,axis=2)+1
# set the pixels without label as the class 0
mask = np.sum(dat,axis=2)>0
lab[mask==0]=0

# setting overlaping rate of adjacent data patches
shift_gap = np.round(patch_size * shift_perc).astype(np.int64)

# calculating the labeling percentage of each data patch
lab_perc = uil.patch_labeling_percentage(mask, patch_size)
lab_idx = lab_perc >= patch_label_perc_thres

# find the image coordinate in label grid
nb_patches = np.sum(lab_idx>0)
order_img_coord = np.unravel_index(np.argsort(lab_idx, axis=None), lab_idx.shape)
order_img_coord = np.transpose(np.stack((np.flip(order_img_coord[0]),np.flip(order_img_coord[1])),axis=0))
order_img_coord = order_img_coord[:nb_patches,:]


# get rid of label patches that are too closed to each other
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
cut label patches and find world coordinates of label patches
'''
# cut label patches
_, lab_in_boundary_index = uil.cut_patch(label_dir, order_img_coord, patch_size)
lab_img_coord = order_img_coord[lab_in_boundary_index,:]
lab_patch, _ = uil.cut_patch_from_array(lab, lab_img_coord.astype(np.int32), patch_size)

# find corresponding data patches
world_coord = uil.image_coord_2_world_coord(lab_img_coord, label_dir)



'''
Find data files
'''
city_code = label_dir.split('/')[-1].split('.')[0].split('_')[1]
data_dir = glob.glob(data_store+'*_'+city_code+'_*/*/*.tif')
for i in range(len(data_dir)-1,-1,-1):
    if 'LCZ_results' in data_dir[i]:
        data_dir.remove(data_dir[i])



'''
Find corresponding data using world cooordinates of label patches
'''
for i in range(len(data_dir)):
    season = data_dir[i].split('/')[-1].split('.')[0].split('_')[-1]
    print('Paring data patches from: {}'.format(data_dir[i]))
    dat_img_coord = uil.world_coord_2_image_coord(world_coord, data_dir[i])
    # cut data patches
    #dat_patch, dat_in_boundary_index = uil.cut_patch_from_array(lab, dat_img_coord.astype(np.int32), patch_size)
    dat_patch_out, dat_in_boundary_index = uil.cut_patch(data_dir[i], dat_img_coord.astype(np.int32), patch_size)
    lab_patch_out = lab_patch[dat_in_boundary_index,:,:]

    print('Shape of data patch: {}'.format(dat_patch_out.shape))
    print('Shape of label patch: {}'.format(lab_patch_out.shape))

    # save data patches
    out_directory = '../data/s2_data_patch/'+city
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    file_name = out_directory+'/PatchSz'+str(patch_size)+'_LabPerc'+str(int(patch_label_perc_thres*100))+'_'+season+'.h5'

    f = h5py.File(file_name,'w')
    f.create_dataset("dat", data=dat_patch_out)
    f.create_dataset("lab", data=lab_patch_out)
    f.close()




