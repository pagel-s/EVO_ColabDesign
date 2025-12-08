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
  MIN_LEN: int = 20
  MAX_LEN: int = 50
  
  def __post_init__(self):
    self.feature_vector = np.array([(self.seq_len - self.MIN_LEN)/ (self.MAX_LEN - self.MIN_LEN), self.perc_hydrophob, self.perc_polar, self.perc_charged, self.perc_other])
    self.aa_seq = "".join([residue_constants.restypes[x] for x in self.seq])
    
def sample_length(n, min_len=20, max_len=50):
  return (np.random.choice(np.arange(min_len, max_len+1), n) - min_len) / (max_len - min_len)

def cvt(k, dim, samples, min_len=20, max_len=50):
    x = np.random.rand(samples, dim -1)
    x = np.hstack((sample_length(samples, min_len=min_len, max_len=max_len).reshape(-1,1), x))
    print("CVT input shape:", x.shape)
    k_means = KMeans(init='k-means++', n_clusters=k,
                     n_init="auto", verbose=0)
    k_means.fit(x)
    return k_means.cluster_centers_

def create_cvt(n_niches, dim_map, samples, min_len=20, max_len=50):
    c = cvt(n_niches, dim_map,
              samples, min_len=min_len, max_len=max_len)
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    return c, kdt

class Archive:
  def __init__(self, archive_dims, niches=500, samples=25_000, min_len=20, max_len=50):
    self.archive = {}
    self.archive_dims = archive_dims
    self.c, self.kdt = create_cvt(niches, archive_dims, samples, min_len=min_len, max_len=max_len)

  def add_to_archive(self, seq: Sequence):
    seq_key = tuple(seq.seq)
    if seq_key in self.archive:
      return
    
    niche_dist = distance.cdist(self.c, [seq.feature_vector], 'euclidean')
    niche_id = np.argmin(niche_dist)
    niche = self.c[niche_id]
    centroid = self.kdt.query([niche])[1][0][0]
    
    if centroid not in self.archive:
      self.archive[centroid] = seq
    elif seq.fitness > self.archive[centroid].fitness:
      self.archive[centroid] = seq
    return None
  
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
