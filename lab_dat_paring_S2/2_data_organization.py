import glob
import os
import h5py
import numpy as np

'''
Merge autumn data of cultural 10 cities as the training data, singapore and new york as the testing data
'''
# cities
cities = 'cultural-10' # Currently Cultural-10 

# Season selection
season = 'all'
#season = 'spring'
#season = 'summer'
#season = 'autumn'
#season = 'winter'

# Mask our unknown class
# if mask_out_unknown_class = True, the values equal 0 for those pixels labeled as unknown class
mask_out_unknown_class = False

# out_directory
if mask_out_unknown_class:
    out_directory = cities+'-season-'+season+'-exclude-unknown'
else:
    out_directory = cities+'-season-'+season+'-include-unknown'


# Season selection
if season == 'all':
    h5_files_selection = '../data/s2_data_patch/*/*.h5'         # all seasons
elif season == 'spring':
    h5_files_selection = '../data/s2_data_patch/*/*_spring.h5'  # spring
elif season == 'summer':
    h5_files_selection = '../data/s2_data_patch/*/*_summer.h5'  # summer
elif season == 'autumn':
    h5_files_selection = '../data/s2_data_patch/*/*_autumn.h5' 	# autumn
elif season == 'winter':
    h5_files_selection = '../data/s2_data_patch/*/*_winter.h5'  # winter

h5_files = glob.glob(h5_files_selection)


dat_list = []
lab_list = []

for idx in range(len(h5_files)):
    if '_22447_' in h5_files[idx] or '_23083_' in h5_files[idx] or '_RGB.h5' in h5_files[idx]:
        # skip Singapore and New York
        continue
    f = h5py.File(h5_files[idx],'r')
    print('{}: {}'.format(h5_files[idx], f['dat'].shape))
    dat_list.append(np.array(f['dat']))
    lab_list.append(np.array(f['lab']))
    f.close()


dat = np.concatenate(dat_list[:],axis=0)
lab = np.concatenate(lab_list[:],axis=0)

if mask_out_unknown_class:
    for i in range(dat.shape[3]):
        tmp = dat[:,:,:,i]
        tmp[lab==0]=0
        dat[:,:,:,i] = tmp


if not os.path.exists(out_directory):
    os.makedirs(out_directory)

file_name = out_directory+'/train.h5'
f = h5py.File(file_name,'w')
f.create_dataset("dat", data=dat)
f.create_dataset("lab", data=lab)
f.close()


