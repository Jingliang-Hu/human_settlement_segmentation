import glob
import h5py
import numpy as np

'''
Merge all season data of cultural 10 cities as the training data, singapore and new york as the testing data
'''
h5_autumn_files = glob.glob('../data/data_patch/*/*_autumn.h5')

dat_list = []
lab_list = []

for idx in range(len(h5_autumn_files)):
    if '_22447_' in h5_autumn_files[idx] or '_23083_' in h5_autumn_files[idx] or '_RGB.h5' in h5_autumn_files[idx]:
        # skip Singapore and New York
        continue
    f = h5py.File(h5_autumn_files[idx],'r')
    print('{}: {}'.format(h5_autumn_files[idx], f['dat'].shape))
    dat_list.append(np.array(f['dat']))
    lab_list.append(np.array(f['lab']))
    f.close()


dat = np.concatenate(dat_list[:],axis=0)
lab = np.concatenate(lab_list[:],axis=0)




