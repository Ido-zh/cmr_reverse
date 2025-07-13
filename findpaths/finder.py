import os
import yaml
from pathlib import Path

with open("./paths.yaml", 'r') as rstream:
    paths = yaml.safe_load(rstream)


def get_diffusion_dump_base(create_new=False):
    base = Path(paths["diffusion_dump_base"]).absolute()
    if not base.exists():
        if create_new:
            base.mkdir(exist_ok=True, parents=True)
        else:
            raise FileNotFoundError(f"Cannot locate dump base {base.as_posix()}.")
    return base


def get_acdc_training_base(nnunet_task_name: str = "Task900_ACDC_Phys"):
    nnunet_raw_base = Path(os.environ.get("nnUNet_raw_data_base"))
    if not nnunet_raw_base.exists():
        raise FileNotFoundError(f"`nnUNet_raw_data_base` is not correctly set! Current value: {nnunet_raw_base.as_posix()}.")
    base = nnunet_raw_base / "nnUNet_raw_data" / nnunet_task_name
    if not base.exists():
        raise FileNotFoundError(f"Cannot locate dump base {base.as_posix()}.")
    return base


def get_reverse_imaging_base(create_new=False):
    base = Path(paths["reverse_imaging_save_base"]).absolute()
    if not base.exists():
        if create_new:
            base.mkdir(exist_ok=True, parents=True)
        else:
            raise FileNotFoundError(f"Cannot locate dump base {base.as_posix()}.")
    return base
