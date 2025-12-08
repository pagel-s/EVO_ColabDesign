#!/usr/bin/env python3
"""
ColabDesign CLI - Protein Design from Command Line

Supports multiple design protocols:
  - binder: binder design
"""

import os
import sys
import argparse
import warnings
import time
import json
import math

warnings.simplefilter(action="ignore", category=FutureWarning)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import jax
import jax.numpy as jnp

try:
    from google.colab import files
except ImportError:
    files = None

from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from colabdesign.af.alphafold.common import residue_constants

##### BINDCRAFT loss functions ######

def add_rg_loss(self, weight=0.1):
    '''add radius of gyration loss'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg":rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight

# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}
    
    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight

# add helicity loss
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):  
      if "offset" in inputs:
        offset = inputs["offset"]
      else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:,None] - idx[None,:]

      # define distogram
      dgram = outputs["distogram"]["logits"]
      dgram_bins = get_dgram_bins(outputs)
      mask_2d = np.outer(np.append(np.zeros(self._target_len), np.ones(self._binder_len)), np.append(np.zeros(self._target_len), np.ones(self._binder_len)))

      x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
      if offset is None:
        if mask_2d is None:
          helix_loss = jnp.diagonal(x,3).mean()
        else:
          helix_loss = jnp.diagonal(x * mask_2d,3).sum() + (jnp.diagonal(mask_2d,3).sum() + 1e-8)
      else:
        mask = offset == 3
        if mask_2d is not None:
          mask = jnp.where(mask_2d,mask,0)
        helix_loss = jnp.where(mask,x,0.0).sum() / (mask.sum() + 1e-8)

      return {"helix":helix_loss}
    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight

# add N- and C-terminus distance loss
def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    '''Add loss penalizing the distance between N and C termini'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight

#####################################



def prepare_inputs(
    starting_pdb=None,
    chain="A",
    length=100,
    target_hotspot_residues=None,
    seed=0,
    advanced_settings={},
):
    assert starting_pdb is not None or length is not None, (
        "Either starting_pdb or length must be provided."
    )
    assert chain is not None, "Chain must be specified for fixed backbone design."
    assert length is None or length > 0, "Length must be a positive integer."

    # Make advanced_settings robust and set defaults
    advanced_settings = advanced_settings or {}

    inputs_prep = {
        "pdb_filename": starting_pdb if starting_pdb else None,
        "chain": chain or "A",
        "binder_len": int(length) if length is not None else 100,
        "hotspot": target_hotspot_residues if target_hotspot_residues else None,
        "seed": int(seed) if seed is not None else 0,
        "rm_aa": advanced_settings.get("omit_AAs", None),
        "rm_target_seq": advanced_settings.get("rm_template_seq_design", False),
        "rm_target_sc": advanced_settings.get("rm_template_sc_design", False),
    }
    return inputs_prep


def get_pdb(pdb_code=""):
    """
    Fetch a PDB file from local file, RCSB PDB, or AlphaFold.
    In Colab, can also upload a file interactively.
    """
    # Handle Colab file upload
    if (pdb_code is None or pdb_code == "") and files is not None:
        upload_dict = files.upload()
        pdb_string = upload_dict[list(upload_dict.keys())[0]]
        with open("tmp.pdb", "wb") as out:
            out.write(pdb_string)
        return "tmp.pdb"

    # Local file
    elif os.path.isfile(pdb_code):
        return pdb_code

    # RCSB PDB code (4-letter)
    elif len(pdb_code) == 4:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"

    # AlphaFold PDB code
    else:
        os.system(
            f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb"
        )
        return f"AF-{pdb_code}-F1-model_v3.pdb"


def run_binder_design(
    pdb_file,
    chain="A",
    length=50,
    iters=100,
    design_method="3stage",
    soft_iters=50,
    temp_iters=50,
    hard_iters=10,
    verbose=1,
    mcmc_steps=1000,
    mcmc_half_life=200,
    mcmc_temp_init=0.01,
    mutation_rate=5,
    semigreedy_tries=10,
    mapelites_iters=100,
    num_elites=100,
    num_mutants=1,
    num_sequences=200,
    min_len=10,
    max_len=20,
    contrastive_pdb=None,
    contrastive_chain="A",
    advanced_settings={},
    seed=0,
    contrastive_target_hotspot_residues=None,
    target_hotspot_residues=None,
    experiment_folder="binder_mapelites",
    helicity_value=0.5,
):
    """Binder design"""
    print("\n" + "=" * 60)
    print("BINDER DESIGN")
    print("=" * 60)

    clear_mem()
    af_model_contrastive = None
    contrastive_inputs = None
    
    if target_hotspot_residues == "":
        target_hotspot_residues = None
    if contrastive_target_hotspot_residues == "":
        contrastive_target_hotspot_residues = None

    af_model = mk_afdesign_model(
        protocol="binder",
        debug=False,
        data_dir=advanced_settings.get("af_params_dir", None),
        use_multimer=advanced_settings.get("use_multimer_design", True),
        num_recycles=advanced_settings.get("num_recycles", 3),
        best_metric="loss",
    )
    af_model.prep_inputs(
        pdb_filename=pdb_file,
        chain=chain,
        binder_len=length,
        hotspot=target_hotspot_residues,
        seed=seed,
        rm_aa=advanced_settings["omit_AAs"],
        rm_target_seq=advanced_settings["rm_template_seq_design"],
        rm_target_sc=advanced_settings["rm_template_sc_design"],
    )

    inputs_prep = prepare_inputs(
        starting_pdb=pdb_file,
        chain=chain,
        length=length,
        target_hotspot_residues=target_hotspot_residues,
        seed=seed,
        advanced_settings=advanced_settings,
    )    

    if contrastive_pdb is not None:
        af_model_contrastive = mk_afdesign_model(
            protocol="binder",
            debug=False,
            data_dir=advanced_settings.get("af_params_dir", None),
            use_multimer=advanced_settings.get("use_multimer_design", True),
            num_recycles=advanced_settings.get("num_recycles", 3),
            best_metric="loss",
        )

        contrastive_inputs = prepare_inputs(
            starting_pdb=contrastive_pdb,
            chain=contrastive_chain,
            length=length,
            target_hotspot_residues=contrastive_target_hotspot_residues,
            seed=seed,
            advanced_settings=advanced_settings,
        )

        af_model_contrastive.prep_inputs(
            pdb_filename=contrastive_pdb,
            chain=contrastive_chain,
            binder_len=length,
            hotspot=contrastive_target_hotspot_residues,
            seed=seed,
            rm_aa=advanced_settings["omit_AAs"],
            rm_target_seq=advanced_settings["rm_template_seq_design"],
            rm_target_sc=advanced_settings["rm_template_sc_design"],
        )


        ### additional loss functions
    if advanced_settings.pop("use_rg_loss", True):
        # radius of gyration loss
        weights_rg = advanced_settings.pop("weights_rg", 0.3)
        add_rg_loss(af_model, weights_rg)
        if af_model_contrastive is not None:
            add_rg_loss(af_model_contrastive, weights_rg)


    if advanced_settings.pop("use_i_ptm_loss", True):
        # interface pTM loss
        weights_iptm = advanced_settings.pop("weights_iptm", 0.05)
        add_i_ptm_loss(af_model, weights_iptm)
        if af_model_contrastive is not None:
            add_i_ptm_loss(af_model_contrastive, weights_iptm)

    if advanced_settings.pop("use_termini_distance_loss", False):
        # termini distance loss
        weights_termini_loss = advanced_settings.pop("weights_termini_loss", 0.1)
        add_termini_distance_loss(af_model, weights_termini_loss)
        if af_model_contrastive is not None:
            add_termini_distance_loss(af_model_contrastive, weights_termini_loss)
    # add the helicity loss
    add_helix_loss(af_model, helicity_value)
    if af_model_contrastive is not None:
        add_helix_loss(af_model_contrastive, helicity_value)


    print(f"Sequence length: {af_model._len}")
    print(f"Weights: {af_model.opt['weights']}")

    af_model.restart()

    # Choose design method
    if design_method == "3stage":
        print(
            f"\nRunning 3-stage design: soft({soft_iters}) -> temp({temp_iters}) -> hard({hard_iters})"
        )
        af_model.design_3stage(
            soft_iters=soft_iters,
            temp_iters=temp_iters,
            hard_iters=hard_iters,
            verbose=verbose,
        )
    elif design_method == "hard":
        print(f"\nRunning hard design for {iters} iterations")
        af_model.design_hard(iters=iters, verbose=verbose)
    elif design_method == "soft":
        print(f"\nRunning soft design for {iters} iterations")
        af_model.design_soft(iters=iters, verbose=verbose)
    elif design_method == "logits":
        print(f"\nRunning logits design for {iters} iterations")
        af_model.design_logits(iters=iters, verbose=verbose)
    elif design_method == "mcmc":
        print(
            f"\nRunning MCMC design: steps={mcmc_steps}, half_life={mcmc_half_life}, temp_init={mcmc_temp_init}"
        )
        af_model._design_mcmc(
            steps=mcmc_steps,
            half_life=mcmc_half_life,
            T_init=mcmc_temp_init,
            mutation_rate=mutation_rate,
            save_best=True,
            verbose=verbose,
        )
    elif design_method == "semigreedy":
        print(
            f"\nRunning semigreedy design for {iters} iterations, tries={semigreedy_tries}"
        )
        af_model.design_semigreedy(
            iters=iters, tries=semigreedy_tries, save_best=True, verbose=verbose
        )
    elif design_method == "mapelites":
        print(
            f"\nRunning MAP-Elites: gens={mapelites_iters}, elites={num_elites}, max_len={max_len}"
        )
        af_model.design_mapelites(
            iters=mapelites_iters,
            num_elites=num_elites,
            num_sequences=num_sequences,
            min_len=min_len,
            max_len=max_len,
            save_best=True,
            inputs_prep=inputs_prep,
            experiment_name=experiment_folder,
            verbose=verbose,
            negative_model=af_model_contrastive,
            negative_inputs=contrastive_inputs,
            mutation_rate=mutation_rate, # will be recalculated inside function for each binder of varying length
        )
    else:
        raise ValueError(f"Unknown design method: {design_method}")

    # Save results
    save_file_name = f"{os.path.basename(pdb_file).split('.')[0]}_{chain}"
    af_model.plot_traj(dpi=150, save_path=f"{save_file_name}_{design_method}_traj.png")
    af_model.save_pdb(f"{af_model.protocol}_{design_method}_design.pdb")
    af_model.plot_pdb()

    print(f"\nSaved PDB: {af_model.protocol}_{design_method}_design.pdb")
    return af_model


def main():
    parser = argparse.ArgumentParser(
        description="ColabDesign CLI - Protein Design Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fixed backbone design with 3-stage
  python design_script.py --protocol fixbb --pdb 1TEN.pdb --chain A --method 3stage
  
  # Hallucination with 3-stage
  python design_script.py --protocol hallucination --length 100 --method 3stage
  
  # Hard design for 200 iterations
  python design_script.py --protocol fixbb --pdb 1TEN.pdb --method hard --iters 200
  
  # MCMC design with custom parameters
  python design_script.py --protocol fixbb --pdb 1TEN.pdb --method mcmc --mcmc-steps 2000 --mcmc-half-life 300
  
  # Semigreedy design
  python design_script.py --protocol fixbb --pdb 1TEN.pdb --method semigreedy --iters 100 --semigreedy-tries 20
  
  # MAP-Elites design (binder hallucination)
  python design_script.py --protocol hallucination --length 50 --method mapelites --mapelites-iters 50 --num-elites 100 --max-len 50
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file with design parameters.",
    )

    # parser.add_argument(
    #     "--protocol",
    #     type=str,
    #     default="binder",
    #     choices=["binder", "hallucination"],
    #     help="Design protocol to use (default: binder)"
    # )

    # parser.add_argument(
    #     "--pdb",
    #     type=str,
    #     default="1TEN",
    #     help="PDB file or PDB code (4-letter RCSB code or AlphaFold ID). Required for fixbb. (default: 1TEN)"
    # )

    # parser.add_argument(
    #     "--chain",
    #     type=str,
    #     default="A",
    #     help="Chain to design (fixbb only, default: A)"
    # )

    # parser.add_argument(
    #     "--length",
    #     type=int,
    #     default=100,
    #     help="Target sequence length for hallucination (default: 100)"
    # )

    # parser.add_argument(
    #     "--method",
    #     type=str,
    #     default="3stage",
    #     choices=["3stage", "hard", "soft", "logits", "mcmc", "mapelites", "semigreedy"],
    #     help="Design method (default: 3stage)"
    # )

    # parser.add_argument(
    #     "--iters",
    #     type=int,
    #     default=100,
    #     help="Number of iterations for single-stage designs (default: 100)"
    # )

    # parser.add_argument(
    #     "--soft-iters",
    #     type=int,
    #     default=50,
    #     help="Iterations for soft stage in 3-stage design (default: 50)"
    # )

    # parser.add_argument(
    #     "--temp-iters",
    #     type=int,
    #     default=50,
    #     help="Iterations for temperature stage in 3-stage design (default: 50)"
    # )

    # parser.add_argument(
    #     "--hard-iters",
    #     type=int,
    #     default=10,
    #     help="Iterations for hard stage in 3-stage design (default: 10)"
    # )

    # parser.add_argument(
    #     "--verbose",
    #     type=int,
    #     default=1,
    #     help="Verbosity level (default: 1)"
    # )

    # # MCMC parameters
    # parser.add_argument(
    #     "--mcmc-steps",
    #     type=int,
    #     default=1000,
    #     help="Number of MCMC steps (default: 1000)"
    # )

    # parser.add_argument(
    #     "--mcmc-half-life",
    #     type=int,
    #     default=200,
    #     help="Half-life for temperature decay in MCMC (default: 200)"
    # )

    # parser.add_argument(
    #     "--mcmc-temp-init",
    #     type=float,
    #     default=0.01,
    #     help="Initial temperature for MCMC annealing (default: 0.01)"
    # )

    # parser.add_argument(
    #     "--mutation-rate",
    #     type=int,
    #     default=1,
    #     help="Mutations per MCMC step (default: 1)"
    # )

    # # MAP-Elites parameters
    # parser.add_argument(
    #     "--mapelites-iters",
    #     type=int,
    #     default=100,
    #     help="Number of MAP-Elites generations (default: 100)"
    # )

    # parser.add_argument(
    #     "--num-elites",
    #     type=int,
    #     default=100,
    #     help="Number of elites in MAP-Elites archive (default: 100)"
    # )

    # parser.add_argument(
    #     "--num-mutants",
    #     type=int,
    #     default=1,
    #     help="Mutations per elite in MAP-Elites (default: 1)"
    # )

    # parser.add_argument(
    #     "--num-sequences",
    #     type=int,
    #     default=200,
    #     help="Initial sequences for MAP-Elites (default: 200)"
    # )

    # parser.add_argument(
    #     "--min-len",
    #     type=int,
    #     default=10,
    #     help="Minimum sequence length for MAP-Elites (default: 10)"
    # )

    # parser.add_argument(
    #     "--max-len",
    #     type=int,
    #     default=20,
    #     help="Maximum sequence length for MAP-Elites (default: 20)"
    # )

    # # Semigreedy parameters
    # parser.add_argument(
    #     "--semigreedy-tries",
    #     type=int,
    #     default=10,
    #     help="Number of tries per semigreedy iteration (default: 10)"
    # )

    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=0,
    #     help="Random seed (default: 0)"
    # )

    # parser.add_argument(
    #     "--contrastive-pdb",
    #     type=str,
    #     default=None,
    #     help="PDB file or code for contrastive binder design (default: None)"
    # )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)

    outdir = config.get("output_dir", "design_output")
    exp_name = config.get("experiment_name", "colabdesign_exp")
    outdir = os.path.join(outdir, exp_name)
    outdir_pdbs = os.path.join(outdir, "pdb")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir_pdbs, exist_ok=True)
    # save config for reference
    with open(os.path.join(outdir, "config_used.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    
    start_time = time.time()


    pdb_file = get_pdb(config["pdb_file"])
    if contrastive_pdb := config.get("contrastive_pdb", None):
        contrastive_file = get_pdb(contrastive_pdb)
    else:
        contrastive_file = None

    run_binder_design(
        pdb_file,
        chain=config.pop("chain", "A"),
        length=config.pop("length", 50),
        iters=config.pop("iters", 100),
        design_method=config.pop("method", "mapelites"),
        soft_iters=config.pop("soft_iterations", 10),
        temp_iters=config.pop("temporary_iterations", 10),
        hard_iters=config.pop("hard_iterations", 10),
        verbose=config.pop("verbose", False),
        mcmc_steps=config.pop("mcmc_steps", 1000),
        mcmc_half_life=config.pop("mcmc_half_life", 200),
        mcmc_temp_init=config.pop("mcmc_temp_init", 0.01),
        semigreedy_tries=config.pop("semigreedy_tries", 10),
        mapelites_iters=config.pop("mapelites_iters", 100),
        num_elites=config.pop("num_elites", 100),
        num_mutants=config.pop("num_mutants", 1),
        num_sequences=config.pop("num_sequences", 200),
        min_len=config.pop("min_len", 20),
        max_len=config.pop("max_len", 40),
        contrastive_pdb=contrastive_file,
        contrastive_chain=config.pop("contrastive_chain", "A"),
        advanced_settings=config.pop("advanced_settings", config),
        seed=config.pop("seed", 0),
        contrastive_target_hotspot_residues=config.pop(
            "contrastive_target_hotspot_residues", None
        ),
        target_hotspot_residues=config.pop("target_hotspot_residues", None),
        experiment_folder=outdir,
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Design completed in {elapsed:.2f} seconds")
    print(f"{'=' * 60}\n")

    # except Exception as e:
    #     print(f"\nERROR: {e}", file=sys.stderr)
    #     sys.exit(1)


if __name__ == "__main__":
    main()
