import random
from osgeo import gdal_array
from utils import *


datasetName = 'DFC2018'
#datasetName = 'Vaihingen'

cropSize = 256
all_rgb, all_dsm, all_sem, all_hsi = collect_tilenames("train", datasetName)
path = './train_data'
if not os.path.exists(path):
    os.mkdir(path)

for idx in range(len(all_rgb)):
    rgb_tile = np.array(Image.open(all_rgb[idx]))
    sem_tile = np.array(Image.open(all_sem[idx]))

    hsi_tile = gdal_array.LoadFile(all_hsi[idx], buf_xsize=1192, buf_ysize=1202)
    hsi_tile = hsi_tile.transpose([1, 2, 0])
    hsi_tile = hsi_tile.astype(np.float32)

    dsm_tile = np.array(Image.open(all_dsm[2 * idx]))
    dem_tile = np.array(Image.open(all_dsm[2 * idx + 1]))
    dsm_tile = correctTile(dsm_tile)
    dem_tile = correctTile(dem_tile)
    dsm_tile = dsm_tile - dem_tile
    norm_tile = genNormals(dsm_tile)

    h = hsi_tile.shape[0]
    w = hsi_tile.shape[1]
    r_list = random.sample(range(0, h - cropSize + 1), 500)
    c_list = random.sample(range(0, w - cropSize + 1), 500)

    for i in range(1, 501):
        r = r_list[i-1]
        c = c_list[i-1]

        rgb = rgb_tile[r:r + cropSize, c:c + cropSize]
        rgb.astype(np.uint8)
        rgb = Image.fromarray(rgb)
        rgb.save(os.path.join(path, 'rgb_' + str(i + idx * 500) + '.png'))

        sem = sem_tile[r:r + cropSize, c:c + cropSize]
        sem.astype(np.uint8)
        sem = Image.fromarray(sem)
        sem.save(os.path.join(path, 'sem_' + str(i + idx * 500) + '.png'))

        hsi = hsi_tile[r:r+cropSize, c:c+cropSize]
        np.save(os.path.join(path, 'hsi_' + str(i + idx * 500) + '.npy'), hsi)

        dsm = dsm_tile[r:r + cropSize, c:c + cropSize]
        np.save(os.path.join(path, 'dsm_' + str(i + idx * 500) + '.npy'), dsm)

        norm = norm_tile[r:r + cropSize, c:c + cropSize]
        np.save(os.path.join(path, 'norm_' + str(i + idx * 500) + '.npy'), norm)










