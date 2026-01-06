# EVO ColabDesign

This code implements a Quality-Diversity Algorithm for binder design with ColabDesign.

## Implementation Details

### Archive Structure
The implementation uses a **GridArchive** with discrete binning optimized for protein sequences:
- **Sequence length**: One bin per integer length (categorical variable)
- **Amino acid composition**: 3D binning over hydrophobic, polar, and charged percentages (other percentage derived as 1 - sum)
- **Niche count**: `n_lengths × (category_bins)³` — typical configurations yield 150-700 niches

### Quality-Diversity Optimization
- **Fitness**: Negative AlphaFold loss (lower loss = higher fitness)
- **Diversity**: Maintained through amino acid category composition and sequence length
- **Operators**: Mutation (position-wise AA substitution) and crossover (chunk-based recombination with pLDDT-guided selection)

### Key Implementation Notes
- All loss terms are length-normalized to prevent bias toward short/long sequences
- Model state (weights, inputs) is preserved across sequence evaluations
- Per-niche elite structures are automatically saved and updated when improved
- Full per-term loss breakdown logged in JSONL format for post-hoc analysis

## Usage


### Running MAP-Elites

```python
af_model.design_mapelites(
    iters=100,
    num_elites=700,  # higher count enables finer category resolution
    mutation_rate=5,
    num_sequences=200,
    min_len=20,
    max_len=40,
    experiment_name="my_experiment",
    init_sampling_strategy="stratified"  # or "random"
)
```

Output:
- `experiment_name/all_seq.jsonl`: Full loss breakdown per evaluated sequence
- `experiment_name/elite_structure/`: Best PDB structure per niche (auto-updated)
- `experiment_name/final_elites.jsonl`: Final archive with complete metrics

### OpenBLAS Configuration
If you encounter OpenBLAS errors:

```fish
set -x OPENBLAS_NUM_THREADS 1
set -x OMP_NUM_THREADS 1
set -x MKL_NUM_THREADS 1
```

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

---

<details>
  <summary>Original ColabDesign README</summary>
  
# ColabDesign
### Making Protein Design accessible to all via Google Colab! 
- P(structure | sequence)
  - [TrDesign](/tr) - using TrRosetta for design
  - [AfDesign](/af) - using AlphaFold for design
  - [WIP] [RfDesign](https://github.com/RosettaCommons/RFDesign) - using RoseTTAFold for design
- P(sequence | structure)
  - [ProteinMPNN](/mpnn)
  - [WIP] TrMRF
- P(sequence)
  - [WIP] [MSA_transformer](/esm_msa)
  - [WIP] [SEQ](/seq) - (GREMLIN, mfDCA, arDCA, plmDCA, bmDCA, etc)
- P(structure)
  - [Rfdiffusion](/rf)

### Where can I chat with other ColabDesign users?
  - See our [Discord](https://discord.gg/gna8maru7d) channel!


### Presentations
[Slides](https://docs.google.com/presentation/d/1Zy7lf_LBK0_G3e7YQLSPP5aj_-AR5I131fTsxJrLdg4/)
[Talk](https://www.youtube.com/watch?v=2HmXwlKWMVs)

### Contributors:
- Sergey Ovchinnikov [@sokrypton](https://github.com/sokrypton)
- Shihao Feng [@JeffSHF](https://github.com/JeffSHF)
- Justas Dauparas [@dauparas](https://github.com/dauparas)
- Weikun.Wu [@guyujun](https://github.com/guyujun) (from [Levinthal.bio](http://levinthal.bio/en/))
- Christopher Frank [@chris-kafka](https://github.com/chris-kafka)

</details>