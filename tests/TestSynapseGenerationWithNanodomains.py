from matplotlib import pyplot as plt
from pysted import exp_data_gen as dg


# generating the molecules dispositions
shroom = dg.Synapse(5, mode="mushroom", seed=42)

shroom_no_nd = dg.Synapse(5, mode="mushroom", seed=42)

n_molecs_in_domain = 10
min_dist = 100
# valid_pos_shroom = shroom.filter_valid_nanodomain_pos()
shroom.add_nanodomains(40, min_dist_nm=min_dist, n_molecs_in_domain=n_molecs_in_domain, seed=42, valid_thickness=3)
# shroom.fatten_nanodomains()

plt.imshow(shroom.frame)
plt.show()