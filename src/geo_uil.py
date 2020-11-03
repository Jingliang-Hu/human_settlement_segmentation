from tqdm import tqdm
from osgeo import gdal, osr, ogr
import numpy as np

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
            data_patches[i,:,:] = dat[img_coord[i,0]-half_patch_size:img_coord[i,0]+half_patch_size,img_coord[i,1]-half_patch_size:img_coord[i,1]+half_patch_size]
    else:
        data_patches = np.zeros((img_coord.shape[0],patch_size,patch_size,bnd))
        for i in tqdm(range(0, img_coord.shape[0])):
            data_patches[i,:,:,:] = dat[img_coord[i,0]-half_patch_size:img_coord[i,0]+half_patch_size,img_coord[i,1]-half_patch_size:img_coord[i,1]+half_patch_size,:]


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
        data_patches[i,:,:,:] = dat[img_coord[i,0]-half_patch_size:img_coord[i,0]+half_patch_size,img_coord[i,1]-half_patch_size:img_coord[i,1]+half_patch_size,:]

    return data_patches, in_boundary_index






