
from colabdesign.af.loss import add_losses


def prepare_model(model, settings):
    model.opt["weights"].update({"pae":settings["weights"]["pae"],
                                "plddt":settings["weights"]["plddt"],
                                "i_pae":settings["weights"]["i_pae"],
                                "con":settings["weights"]["con"],
                                "i_con":settings["weights"]["i_con"],
                                    })

    model.opt["con"].update({"num":settings["opt"]["con"]["num"],"cutoff":settings["opt"]["con"]["cutoff"],"binary":False,"seqsep":9})
    model.opt["i_con"].update({"num":settings["opt"]["i_con"]["num"],"cutoff":settings["opt"]["i_con"]["cutoff"],"binary":False})
    return model


def add_inputs(model, settings):
    model.prep_inputs(
        pdb_filename=settings["pdb_filename"],
        chain=settings["chain"],
        binder_len=settings["binder_len"],
        hotspot=settings["hotspot"],
        seed=settings["seed"],
        rm_aa=settings["rm_aa"],
        rm_target_seq=settings["rm_target_seq"],
        rm_target_sc=settings["rm_target_sc"],
    )

def set_up_model(model, settings):
    add_inputs(model, settings)
    prepare_model(model, settings["model_prep"])
    add_losses(model, settings)