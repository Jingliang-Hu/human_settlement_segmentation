from tqdm import tqdm
from osgeo import gdal, osr, ogr
import numpy as np
from tqdm import tqdm

def image_coord_2_world_coord(img_coord, geo_file):
    '''
    This function transforms image coordinates in rows and columns to latitude and longitude
    
    Input:
    	- img_coord 		-- N by 2 array, N is the number of pixels; [row, column]
    	- geo_file 		-- directory to geo-reference grid file that can be load by gdal
    Output:
    	- world_coord 		-- N by 2 array, N is the number of pixels; [latitude, longitude]
    '''
    # longitude latitude ESPG
    out_EPSG = 4326
    # target geo file
    f = gdal.Open(geo_file)
    # The upper left corner of the upper left pixel is at position (geoInfoGrid[0],geoInfoGrid[3]).
    xoffset, px_w, rot1, yoffset, rot2, px_h = f.GetGeoTransform()
    # in EPSG
    proj = osr.SpatialReference(wkt=f.GetProjection())
    in_EPSG = proj.GetAttrValue('AUTHORITY',1)

    del f

    # calculate the world coordinate
    posX = xoffset + px_w * img_coord[:,1] + rot1 * img_coord[:,0]
    posY = yoffset + rot2 * img_coord[:,1] + px_h * img_coord[:,0]
    # shift to the center of the pixel
    posX -= px_w / 2.0
    posY -= px_h / 2.0

    # set coordinate transformation function
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(int(in_EPSG))

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(out_EPSG)

    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # initial output
    world_coord = np.zeros((img_coord.shape[0],2))

    # transform point
    for i in range(0,world_coord.shape[0]):
        p = ogr.Geometry(ogr.wkbPoint)
        p.AddPoint(posX[i], posY[i])
        p.Transform(coordTransform)
        world_coord[i][0] = p.GetX()
        world_coord[i][1] = p.GetY()

    return world_coord



def world_coord_2_image_coord(world_coord, geo_file):
    # WGS 84 / UTM zone
    in_EPSG = 4326
    # target geo file
    f = gdal.Open(geo_file)
    # The upper left corner of the upper left pixel is at position (geoInfoGrid[0],geoInfoGrid[3]).
    xoffset, px_w, rot1, yoffset, rot2, px_h = f.GetGeoTransform()
    # in EPSG
    proj = osr.SpatialReference(wkt=f.GetProjection())
    out_EPSG = proj.GetAttrValue('AUTHORITY',1)
    del f

    # set coordinate transformation function
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(in_EPSG)

    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(int(out_EPSG))

    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # initial output
    img_coord = np.zeros((world_coord.shape))
    pos = np.zeros(world_coord.shape)
    # transform point
    for i in range(0,world_coord.shape[0]):
        p = ogr.Geometry(ogr.wkbPoint)
        p.AddPoint(world_coord[i,0], world_coord[i,1])
        p.Transform(coordTransform)
        pos[i,0] = p.GetX()
        pos[i,1] = p.GetY()

    img_coord[:,1] = np.ceil((pos[:,0] - xoffset) / px_w)
    img_coord[:,0] = np.ceil((pos[:,1] - yoffset) / px_h)

    return img_coord


def cut_patch_from_array(dat, img_coord, patch_size):

    if dat.ndim == 2:
        [row, col] = dat.shape
        bnd = 1
    elif dat.ndim == 3:
        [row, col, bnd] = dat.shape

    # half patch size for cutting patches
    half_patch_size = np.floor(patch_size/2).astype(np.int32)
    # delete the incomplete data patches on the boundary due to the patch size.
    upper_out = (img_coord[:,0] - half_patch_size)<0
    print("%d samples on the upper boundary are deleted" %(np.sum(upper_out)))
    bottom_out = (img_coord[:,0] + half_patch_size)>=row
    print("%d samples on the bottom boundary are deleted" %(np.sum(bottom_out)))
    left_out = (img_coord[:,1] - half_patch_size)<0
    print("%d samples on the left boundary are deleted" %(np.sum(left_out)))
    right_out = (img_coord[:,1] + half_patch_size)>=col
    print("%d samples on the right boundary are deleted" %(np.sum(right_out)))

    in_boundary_index = np.logical_not(np.logical_or(right_out, np.logical_or(left_out, np.logical_or(bottom_out, upper_out))))
    img_coord = img_coord[in_boundary_index,:]

    if bnd == 1:
        data_patches = np.zeros((img_coord.shape[0],patch_size,patch_size))
        for i in tqdm(range(0, img_coord.shape[0])):
            data_patches[i,:,:] = dat[img_coord[i,0]-half_patch_size+1:img_coord[i,0]+half_patch_size+1,img_coord[i,1]-half_patch_size+1:img_coord[i,1]+half_patch_size+1]
    else:
        data_patches = np.zeros((img_coord.shape[0],patch_size,patch_size,bnd))
        for i in tqdm(range(0, img_coord.shape[0])):
            data_patches[i,:,:,:] = dat[img_coord[i,0]-half_patch_size+1:img_coord[i,0]+half_patch_size+1,img_coord[i,1]-half_patch_size+1:img_coord[i,1]+half_patch_size+1,:]


    return data_patches, in_boundary_index




def cut_patch(geo_file, img_coord, patch_size):

    # load data or label
    f = gdal.Open(geo_file)
    dat = f.ReadAsArray()
    row = f.RasterYSize
    col = f.RasterXSize
    bnd = f.RasterCount
    del f
    if bnd>1:
        dat = np.transpose(dat,[1,2,0])

    # half patch size for cutting patches
    half_patch_size = np.floor(patch_size/2).astype(np.int32)
    # initial data patches output
    data_patches = np.zeros((img_coord.shape[0],patch_size,patch_size,bnd))

    # delete the incomplete data patches on the boundary due to the patch size.
    upper_out = (img_coord[:,0] - half_patch_size)<0
    print("%d samples on the upper boundary are deleted" %(np.sum(upper_out)))
    bottom_out = (img_coord[:,0] + half_patch_size)>=row
    print("%d samples on the bottom boundary are deleted" %(np.sum(bottom_out)))
    left_out = (img_coord[:,1] - half_patch_size)<0
    print("%d samples on the left boundary are deleted" %(np.sum(left_out)))
    right_out = (img_coord[:,1] + half_patch_size)>=col
    print("%d samples on the right boundary are deleted" %(np.sum(right_out)))

    in_boundary_index = np.logical_not(np.logical_or(right_out, np.logical_or(left_out, np.logical_or(bottom_out, upper_out))))
    img_coord = img_coord[in_boundary_index,:]

    for i in tqdm(range(0, img_coord.shape[0])):
        data_patches[i,:,:,:] = dat[img_coord[i,0]-half_patch_size+1:img_coord[i,0]+half_patch_size+1,img_coord[i,1]-half_patch_size+1:img_coord[i,1]+half_patch_size+1,:]

    return data_patches, in_boundary_index



def patch_labeling_percentage(label_mask, patch_size):
    '''
    This function calculate the labeling percentage of a data patch with the given patch size
    Input:
        - label_mask        -- a 2D binary array indicating where labels are
        - patch_size        -- the size of a data patch
    Output:
        - label_perc        -- a 2D binary array indicating labeling percentage of a data patch
    '''
    # indication of where are the labels for the first three classes
    lab_tmp = label_mask
    half_patch = np.array(patch_size/2).astype(np.int8)
    # find the center points of patches whose labeled contents occupy a percentage higher than setting
    lab_tmp_pad = np.pad(lab_tmp,((half_patch,half_patch),(half_patch,half_patch))).astype(np.int32)
    lab_idx = np.zeros(lab_tmp_pad.shape).astype(np.int32)
    print('Calculating the patch label percentage:')
    for i in tqdm(range(-half_patch,half_patch)):
        rotation = np.roll(lab_tmp_pad,i,axis=0)
        for j in range(-half_patch,half_patch):
            lab_idx += np.roll(rotation,j,axis=1)

    lab_idx = lab_idx[half_patch:-half_patch,half_patch:-half_patch]
    label_perc = lab_idx/(patch_size*patch_size)
    return label_perc


class dataAndLabel():
    def __init__(self, dir2label, dir2data, patchsize, interval):
        if os.path.exists(dir2label):
            self.dir2label = dir2label
        else:
            print('The given label file does not exist')
            return 1

        if os.path.exists(dir2data):
            self.dir2data = dir2data
        else:
            print('The given data file does not exist')
            return 1

        self.patchsize = patchsize
        self.interval = interval
        return 0

    def getDataGSD():
        f = gdal.Open(self.dir2data)
        gMatrix = f.GetGeoTransform()
        f.close()
        x_gsd = gMatrix[1]
        y_gsd = gMatrix[5]
        if y_gsd!=-x_gsd:
            print('Ground sampling distances are different on x and y directions, GSD on x direction is returned.')
        return x_gsd

    def getLabelGSD():
        f = gdal.Open(self.dir2label)
        gMatrix = f.GetGeoTransform()
        f.close()
        x_gsd = gMatrix[1]
        y_gsd = gMatrix[5]
        if y_gsd!=-x_gsd:
            print('Ground sampling distances are different on x and y directions, GSD on x direction is returned.')
        return x_gsd

    def dataUpsampling():
        return('to be constructed')


    def cutData():
        f = gdal.Open(self.dir2label)
        lab = f.ReadAsArray()
        [x0,x_res,_,y0,_,y_res] = f.GetGeoTransform()

        f_dat = gdal.Open(self.dir2data)
        dat = f_dat.ReadAsArray()
        [xd0,xd_res,_,yd0,_,yd_res] = f_dat.GetGeoTransform()
        print('Label information: tie point: {}, {}; resolution: {}, {}; image size: {}, {}'.format(x0,y0,x_res,y_res,f.RasterXSize, f.RasterYSize))
        print(' Data information: tie point: {}, {}; resolution: {}, {}; image size: {}, {}'.format(xd0,yd0,xd_res,yd_res,f_dat.RasterXSize, f_dat.RasterYSize))
        # for x_idx in range(0,f.RasterXSize-self.patchsize,self.interval):
           # for y_idx in range(0,f.RasterYSize-self.patchsize,self.inverval):
        return('to be constructed')

