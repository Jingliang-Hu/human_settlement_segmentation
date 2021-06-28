import glob
from PIL import Image
import numpy as np
import h5py
import sys


city = '00005_21206_Mumbai'
city = sys.argv[1]

h5py_file = '../data/data_patch/'+city+'/*.h5'
h5py_file = glob.glob('../data/data_patch/'+city+'/*.h5')[0]
out_dirs = '../data/data_patch/'+city+'/'

f = h5py.File(h5py_file,'r')
dat = np.array(f['dat'])
lab = np.array(f['lab'])
f.close()

nb_patch = dat.shape[0]
rgb = np.stack((dat[:,:,:,4],dat[:,:,:,3],dat[:,:,:,2]),axis=3)
del dat

lab_img = np.zeros((nb_patch,32,32,3),dtype=np.uint8)
# red --> commectial
lab_img[:,:,:,0] = 255 * (lab == 1)
# green --> residential
lab_img[:,:,:,1] = 255 * (lab == 3)
# blue --> industrial
lab_img[:,:,:,2] = 255 * (lab == 2)


# rgb normlaization to 0 - 255
for idx_bnd in range(3):
    mmin = np.min(rgb[:,:,:,idx_bnd])
    mmax = np.max(rgb[:,:,:,idx_bnd])
    rgb[:,:,:,idx_bnd] = 255 * ((rgb[:,:,:,idx_bnd]-mmin)/(mmax-mmin))





for i in range(nb_patch):
    PIL_dat_image = Image.fromarray(np.uint8(rgb[i,:,:,:])).convert('RGB')
    PIL_dat_image.save(out_dirs+str(i+1)+"_img.png")

    PIL_lab_image = Image.fromarray(np.uint8(lab_img[i,:,:,:])).convert('RGB')
    PIL_lab_image.save(out_dirs+str(i+1)+"_lab.png")

