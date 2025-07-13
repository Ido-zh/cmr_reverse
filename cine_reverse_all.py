import torch
import numpy as np
from pathlib import Path
from cond_diffusion.ssfp import bSSFPImagingCondition
from cond_diffusion.DDPMConditionScheduler import DDPMConditionalScheduler
from cond_diffusion.DDPMCondPipeline import DDPMCondPipeline
from cond_diffusion.utils import NCC
from data.niftiio import NiftiImage
from findpaths import finder as pathfinder

# 0. load dataset
diffusion_base = pathfinder.get_diffusion_dump_base(create_new=False)
pipeline = DDPMCondPipeline.from_pretrained(diffusion_base / "phys-acdc-128").to('cuda')
scheduler = DDPMConditionalScheduler.from_pretrained(diffusion_base / "phys-acdc-128/scheduler")
pipeline.scheduler = scheduler

# 1. ACDC dataset
source_base = pathfinder.get_acdc_training_base() / "imagesTr"
nifti_files = sorted(list(source_base.glob("*.nii.gz")))
save_base = pathfinder.get_reverse_imaging_base(create_new=True) / "acdc-phys-128"
save_base.mkdir(exist_ok=True, parents=True)

# 2. Reverse all!
for nft_file in nifti_files:
    image = NiftiImage(nft_file)
    print(image.identifier)

    # MOLLI cond
    conditioner = bSSFPImagingCondition(rho_rescale=0.6, t1_rescale=3200.,
                                        t2_rescale=300., sigma=0.1,
                                        likelihood='gaussian')
    """
    Be careful with the seeds!
    """
    good_estimate = np.zeros((image.slices, ), dtype=bool)
    posterior = []
    t1_sim, t2_sim = [], []
    for sl in range(image.slices):
        # go through all slices
        cond_data = image.get_condition(sl, cond_size=128, fa=45., in_scale=0.3)
        hidden = pipeline(
            batch_size=5,
            num_inference_steps=200,
            condition_input=cond_data,
            conditioner=conditioner,
            generator=torch.Generator(device='cpu').manual_seed(1),
            step_size=400,
        ).images
        hidden_orig = image.convert_hidden(hidden)
        hidden_orig = torch.clamp((hidden_orig+1)*0.5, 0., 1.)

        # sanity check
        ncc = NCC()
        reference = cond_data["ssfp"]
        t1 = hidden[:, [1], ]
        t2 = hidden[:, [2], ]
        ncc_t1 = ncc.loss(reference.to(t1), t1).squeeze().detach().cpu().numpy()
        ncc_t2 = ncc.loss(reference.to(t2), t2).squeeze().detach().cpu().numpy()
        ind = np.argmin(ncc_t1 + ncc_t2)
        hidden_orig = hidden_orig.detach().cpu().numpy()
        posterior.append(hidden_orig[ind])
        t1_sim.append(ncc_t1[ind])
        t2_sim.append(ncc_t2[ind])
        if ncc_t1[ind] > - 0.45 or ncc_t2[ind] > -0.53:
            print(f"Abnormal phys: {image.identifier}, slice: {sl}, NCC: {ncc_t1[ind]} (T1), {ncc_t2[ind]} (T2)\n")
            print(image.identifier, "Slice ind:", sl, ncc_t1[ind], ncc_t2[ind])
    t1_sim = np.stack(t1_sim, axis=0)
    t2_sim = np.stack(t2_sim, axis=0)

    # rescale PD/T1/T2
    posterior = np.stack(posterior, axis=0)
    posterior[:, 0] /= 0.6
    posterior[:, 1] *= 3200
    posterior[:, 0] *= 300

    np.savez(save_base / f"{image.identifier}.npz", posterior=posterior,
             t1_sim=t1_sim, t2_sim=t2_sim)

