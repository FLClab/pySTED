from matplotlib import pyplot as plt
import numpy as np
import tifffile
import tqdm

path = "D:/SCHOOL/Maitrise/H2021/Recherche/Data/Ca2+/stream1.tif"
theresa_data = tifffile.imread(path)
start = 450
end = 490
len_sequence = theresa_data.shape[0] - start
col_start, col_end = 140, 160
row_start, row_end = 100, 125

# je veux que mon affichage soit normalisé pendant le vidéo, donc je trouve le max de tout et j'ajoute une bordure
# à mes frames or something :)
max_photons = np.max(theresa_data[start: end, row_start: row_end, col_start: col_end])

plt.ion()
fig, axes = plt.subplots(1, 1)
for i in tqdm.trange(start, end):
    theresa_data[i, row_start, col_start] = max_photons
    axes.clear()
    axes.set_title(f"frame {i + 1}")
    # axes.imshow(theresa_data[i], cmap="inferno")   # whole
    flash_imshow = axes.imshow(theresa_data[i, row_start: row_end, col_start: col_end], cmap="inferno")   # crop
    plt.pause(0.1)
plt.show()
