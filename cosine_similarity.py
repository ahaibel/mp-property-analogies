from scipy.spatial.distance import cdist
import numpy as np

# EuAl3Cu2
ref_a = 5.419676971280253
ref_b = 5.419676971280253
ref_c = 4.07009126
ref_v = 103.53366382224992

tes_a = 5.5652949
tes_b = 5.5652949
tes_c = 3.7621771
tes_v = 100.9127948

del_a = ref_a - tes_a
del_b = ref_b - tes_b
del_c = ref_c - tes_c
del_v = ref_v - tes_v

curr_deltas = [[]]

dists = cdist(curr_deltas, batch, "euclidean")
