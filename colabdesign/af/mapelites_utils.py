import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from scipy.spatial import distance
from tqdm import tqdm as track

from colabdesign.af.alphafold.common import residue_constants


### AminoAcid level archive

AA_mapping = {
    "A": "hydrophob",
    "V": "hydrophob",
    "I": "hydrophob",
    "L": "hydrophob",
    "M": "hydrophob",
    "F": "hydrophob",
    "Y": "hydrophob",
    "W": "hydrophob",
    "S": "polar",
    "T": "polar",
    "N": "polar",
    "Q": "polar",
    "H": "charged",
    "K": "charged",
    "R": "charged",
    "D": "charged",
    "E": "charged",
    "C": "other",
    "G": "other",
    "P": "other"
}


AA_idx = {
    i: AA for i, AA in enumerate(residue_constants.restypes)
}  # this order should match the order of the alphabet in the model

archive_dims = {
    "seq_len": 0,
    "perc_hydrophob": 1,
    "perc_polar": 2,
    "perc_charged": 3,
    "perc_other": 4,
}

### Utilities for MAP-Elites archive

@dataclass
class Sequence:
  seq: list
  perc_hydrophob: float
  perc_polar: float
  perc_charged: float
  perc_other: float
  seq_len: int
  loss: float = 0.0
  feature_vector: np.ndarray = None
  centroid: np.ndarray = None
  fitness: float = 0.0
  aux: dict = None
  min_len: int = 20
  max_len: int = 50
  niche_id: int = None
  
  def __post_init__(self):
    self.feature_vector = np.array([(self.seq_len - self.min_len)/ (self.max_len - self.min_len), self.perc_hydrophob, self.perc_polar, self.perc_charged, self.perc_other])
    self.aa_seq = "".join([residue_constants.restypes[x] for x in self.seq])
    
def sample_length(n, min_len=20, max_len=50):
    return (np.random.choice(np.arange(min_len, max_len+1), n) - min_len) / (max_len - min_len)

# def cvt(k, dim, samples, min_len=20, max_len=50):
#     x = np.random.rand(samples, dim - 1)
#     x = np.hstack((sample_length(samples, min_len=min_len, max_len=max_len).reshape(-1, 1), x))
#     print("CVT input shape:", x.shape)
#     k_means = KMeans(init='k-means++', n_clusters=k, n_init="auto", verbose=0)
#     k_means.fit(x)
#     return k_means.cluster_centers_

def cvt(k, dim, samples, min_len=20, max_len=50):
    n_cuts = dim - 2

    u = np.random.rand(samples, n_cuts)
    u.sort(axis=1)
    
    u_with_bounds = np.hstack((np.zeros((samples, 1)), u, np.ones((samples, 1))))
    
    percentages = np.diff(u_with_bounds, axis=1)

    length_dim = sample_length(samples, min_len=min_len, max_len=max_len).reshape(-1, 1)

    x = np.hstack((length_dim, percentages))
    print("CVT input shape:", x.shape) 
    k_means = KMeans(init='k-means++', n_clusters=k, n_init="auto", verbose=0)
    k_means.fit(x)
    return k_means.cluster_centers_

def create_cvt(n_niches, dim_map, samples, min_len=20, max_len=50):
    c = cvt(n_niches, dim_map, samples, min_len=min_len, max_len=max_len)
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    return c, kdt

class Archive:
  def __init__(self, archive_dims, niches=500, samples=75_000, min_len=20, max_len=50):
    self.archive = {}
    self.archive_dims = archive_dims
    self.c, self.kdt = create_cvt(niches, archive_dims, samples, min_len=min_len, max_len=max_len)

  def add_to_archive(self, seq: Sequence):
    added = False
    seq_key = tuple(seq.seq)
    if seq_key in self.archive:
      return added
    
    niche_dist = distance.cdist(self.c, [seq.feature_vector], 'euclidean')
    niche_id = np.argmin(niche_dist)
    niche = self.c[niche_id]
    centroid = self.kdt.query([niche])[1][0][0]
    seq.niche_id = niche_id
    
    if centroid not in self.archive:
      self.archive[centroid] = seq
      added = True
    elif seq.fitness > self.archive[centroid].fitness:
      self.archive[centroid] = seq
      added = True
    return added
  
  def fits_in_archive(self, seq: Sequence):
    
    seq_key = tuple(seq.seq)
    if seq_key in self.archive:
      return -1
    
    print("C: ", self.c.shape, "Features :", seq.feature_vector)
    niche_dist = distance.cdist(self.c, [seq.feature_vector], 'euclidean')
    niche_id = np.argmin(niche_dist)
    niche = self.c[niche_id]
    centroid = self.kdt.query([niche])[1][0][0]
    
    if centroid not in self.archive:
      return niche_id
    return -1
  
  def __contains__(self, key: tuple):
    if len(self.archive) == 0:
      return False
    elite_sequences = [tuple(x.seq) for x in self.archive.values()]
    return key in elite_sequences
  
  def __len__(self):
    return len(self.archive)
  
  @property
  def __getitem__(self, key):
    return self.archive[key]
  
  @property
  def elite_sequences(self):
    return [x.seq for x in self.archive.values()]

  @property
  def elites(self):
    return list(self.archive.values())



### SAMPLING

def sample_aas_by_category(n, aa_to_cat, category_probs=None, rng=None, return_indices=True):
    """
    Sample n amino acids where each category has equal probability (by default),
    and AAs within a category are sampled uniformly. Works with any aa->category
    mapping whose keys are one-letter codes or AF indices.

    - aa_to_cat: dict like {"A":"hydrophob", ...} or {idx:"hydrophob", ...}
    - category_probs: optional dict {category: prob} or array aligned to discovered categories
    - return_indices: if True, return AF indices per residue_constants.restypes; else return list of letters
    """
    letters, cats = [], []
    for k, v in aa_to_cat.items():
        aa = k
        if isinstance(k, int):
            if 0 <= k < len(residue_constants.restypes):
                aa = residue_constants.restypes[k]
            else:
                continue
        if aa not in residue_constants.restypes:
            continue
        letters.append(aa)
        cats.append(v)

    # build category -> list of AAs
    cat_to_aas = {}
    for aa, cat in zip(letters, cats):
        cat_to_aas.setdefault(cat, []).append(aa)

    # keep only non-empty categories
    cat_names = [c for c, lst in cat_to_aas.items() if lst]
    if not cat_names:
        raise ValueError("No valid amino acids found in mapping.")

    C = len(cat_names)
    if category_probs is None:
        probs = np.full(C, 1.0 / C)
    else:
        if isinstance(category_probs, dict):
            probs = np.array([float(category_probs.get(c, 0.0)) for c in cat_names], dtype=float)
        else:
            probs = np.array(category_probs, dtype=float)
            assert probs.shape[0] == C, "category_probs length must match number of categories"
        s = probs.sum()
        assert s > 0, "category_probs must sum > 0"
        probs = probs / s

    if rng is None:
        rng = np.random.default_rng()

    # sample categories then AAs within each category
    chosen_cats_idx = rng.choice(len(cat_names), size=int(n), p=probs)
    seq_letters = [rng.choice(cat_to_aas[cat_names[i]]) for i in chosen_cats_idx]

    if return_indices:
        return np.array([residue_constants.restypes.index(a) for a in seq_letters], dtype=int)
    return seq_letters