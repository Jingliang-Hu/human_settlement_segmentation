import numpy as np
from tqdm import tqdm
from osgeo import gdal

def intial_geotiff_segmentation_prob_map(in_data_directory, out_map_directory):
    f = gdal.Open(in_data_directory)
    n_row = f.RasterYSize
    n_col = f.RasterXSize
    proj = f.GetProjection()
    geoInfo = f.GetGeoTransform()
    data = f.ReadAsArray()
    mask = (np.sum(np.transpose(data,(1,2,0)),axis=2)>0).astype(np.uint8)
    del data
    del f

    f = gdal.GetDriverByName('GTiff')
    out_file = f.Create(out_map_directory, n_col, n_row, 3, gdal.GDT_Float32)
    out_file.SetProjection(proj)
    out_file.SetGeoTransform(geoInfo)
    for i in range(3):
        outBand = out_file.GetRasterBand(i+1)
        outBand.WriteArray(np.zeros((n_row,n_col),dtype = np.uint8))
        outBand.FlushCache()
    del f
    del outBand
    del out_file

    out_mask_dir = out_map_directory.replace('.tif','_mask.tif')
    f = gdal.GetDriverByName('GTiff')
    out_file = f.Create(out_mask_dir, n_col, n_row, 1, gdal.GDT_Byte)
    out_file.SetProjection(proj)
    out_file.SetGeoTransform(geoInfo)
    outBand = out_file.GetRasterBand(1)
    outBand.WriteArray(mask)
    outBand.FlushCache()
    del f
    return 0




def intial_geotiff_segmentation_map(in_data_directory, out_map_directory):
    f = gdal.Open(in_data_directory)
    n_row = f.RasterYSize
    n_col = f.RasterXSize
    proj = f.GetProjection()
    geoInfo = f.GetGeoTransform()
    data = f.ReadAsArray()
    mask = (np.sum(np.transpose(data,(1,2,0)),axis=2)>0).astype(np.uint8)
    del data
    del f

    f = gdal.GetDriverByName('GTiff')
    out_file = f.Create(out_map_directory, n_col, n_row, 1, gdal.GDT_Byte)
    out_file.SetProjection(proj)
    out_file.SetGeoTransform(geoInfo)
    outBand = out_file.GetRasterBand(1)
    outBand.WriteArray(np.zeros((n_row,n_col),dtype = np.uint8))
    outBand.FlushCache()
    del f
    del outBand
    del out_file
    
    out_mask_dir = out_map_directory.replace('.tif','_mask.tif')
    f = gdal.GetDriverByName('GTiff')
    out_file = f.Create(out_mask_dir, n_col, n_row, 1, gdal.GDT_Byte)
    out_file.SetProjection(proj)
    out_file.SetGeoTransform(geoInfo)
    outBand = out_file.GetRasterBand(1)
    outBand.WriteArray(mask)
    outBand.FlushCache()
    del f
    return 0


def cut_patch_4_inference(in_data_directory, patch_size):
    f = gdal.Open(in_data_directory)
    n_row = f.RasterYSize
    n_col = f.RasterXSize
    n_bnd = f.RasterCount
    data = f.ReadAsArray()
    data = np.transpose(data,(1,2,0))
    del f

    row = np.arange(0, n_row, patch_size)
    col = np.arange(0, n_col, patch_size)
    n_patches = (len(row)-1)*(len(col)-1)
    data_patches = np.zeros((n_patches, patch_size, patch_size, n_bnd))

    nb_row_blocks = len(row)-1
    nb_col_blocks = len(col)-1
    for idx_row in tqdm(range(nb_row_blocks)):
        for idx_col in range(nb_col_blocks):
            data_patches[idx_row*nb_col_blocks+idx_col,:,:,:] = data[row[idx_row]:row[idx_row+1],col[idx_col]:col[idx_col+1],:]

    img_coord = []
    img_coord.append(row)
    img_coord.append(col) 

    return data_patches, img_coord

def save_regression_inference(out_geotiff, prediction, img_coord):
    f = gdal.Open(out_geotiff)
    n_row = f.RasterYSize
    n_col = f.RasterXSize
    proj = f.GetProjection()
    geoInfo = f.GetGeoTransform()
    del f

    row = img_coord[0]
    col = img_coord[1]
    n_patches = (len(row)-1)*(len(col)-1)

    nb_row_blocks = len(row)-1
    nb_col_blocks = len(col)-1

    pred = np.zeros((3,n_row,n_col))
    for idx_row in tqdm(range(nb_row_blocks)):
        for idx_col in range(nb_col_blocks):
            pred[:,row[idx_row]:row[idx_row+1],col[idx_col]:col[idx_col+1]] = prediction[idx_row*nb_col_blocks+idx_col,:,:,:]

    f = gdal.GetDriverByName('GTiff')
    driver = f.Create(out_geotiff, n_col, n_row, 3, gdal.GDT_Float32)
    driver.SetProjection(proj)
    driver.SetGeoTransform(geoInfo)
    for i in range(3):
        outBand = driver.GetRasterBand(i+1)
        outBand.WriteArray(pred[i,:,:])
        outBand.FlushCache()
    del(outBand)



def save_prediction_inference(out_geotiff, prediction, img_coord):
    f = gdal.Open(out_geotiff)
    n_row = f.RasterYSize
    n_col = f.RasterXSize
    proj = f.GetProjection()
    geoInfo = f.GetGeoTransform()
    del f

    row = img_coord[0]
    col = img_coord[1]
    n_patches = (len(row)-1)*(len(col)-1)
    
    nb_row_blocks = len(row)-1
    nb_col_blocks = len(col)-1

    pred = np.zeros((n_row,n_col),dtype=np.uint8)
    for idx_row in tqdm(range(nb_row_blocks)):
        for idx_col in range(nb_col_blocks):
            pred[row[idx_row]:row[idx_row+1],col[idx_col]:col[idx_col+1]] = prediction[idx_row*nb_col_blocks+idx_col,:,:]


    f = gdal.GetDriverByName('GTiff')
    driver = f.Create(out_geotiff, n_col, n_row, 1, gdal.GDT_Byte)
    driver.SetProjection(proj)
    driver.SetGeoTransform(geoInfo)

    outBand = driver.GetRasterBand(1)
    outBand.WriteArray(pred)
    outBand.FlushCache()
    del(outBand)







