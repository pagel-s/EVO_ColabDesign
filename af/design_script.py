import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from IPython.display import HTML, display

print("ColabDesign setup complete.")
try:
    from google.colab import files
except ImportError:
    files = None  # will handle file uploads differently if not in Colab

from colabdesign import mk_afdesign_model, clear_mem

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
        os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb")
        return f"AF-{pdb_code}-F1-model_v3.pdb"

# Example usage (timing)
import time
start_time = time.time()
pdb_file = get_pdb("1a3n")  # replace with your PDB code or leave blank to upload
print(f"PDB file obtained: {pdb_file}")
print(f"Elapsed time: {time.time() - start_time:.2f} seconds")



clear_mem()
af_model = mk_afdesign_model(protocol="fixbb")
af_model.prep_inputs(pdb_filename=get_pdb("1TEN"), chain="A")

print("length",  af_model._len)
print("weights", af_model.opt["weights"])


af_model.restart()
af_model.design_3stage()

af_model.plot_traj()  


af_model.save_pdb(f"{af_model.protocol}.pdb")
af_model.plot_pdb()