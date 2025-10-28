# Reverse Imaging in Cardiac Magnetic Resonance Imaging
*[Yidong Zhao](https://www.tudelft.nl/staff/y.zhao.8/), Department of Imaging Physics, Delft University of Technology, The Netherlands*

In MRI, images acquired with different RF pulses are manifestations of the same underlying tissue properties including magnetization strength $\mathrm{M}_0$, $\mathrm{T}_1$, $\mathrm{T}_2$. ***Reverse imaging*** estimates the underlying physical properties of tissues that have caused the observed images, from qualitative images such as bSSFP cine. The physical properties can be used for physics-grounded cross-sequence synthesis and data augmentation in training segmentation models.
![Reverse imaging](assests/git-demo.png)
*We note that only the 128x128 resolution reverse imaging has been published. High resolution reverse imaging models and results as shown in the figure will be open-sourced later.*


## Citation
[MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/2605_paper.pdf), **Young Scientist Award**.
```bibtex
@inproceedings{zhao2025reverse,
  title={Reverse Imaging for Wide-Spectrum Generalization of Cardiac MRI Segmentation},
  author={Zhao, Yidong and Kellman, Peter and Xue, Hui and Yang, Tongyun and Zhang, Yi and Han, Yuchi and Simonetti, Orlando and Tao, Qian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={555--565},
  year={2025},
  organization={Springer}
}
```

## 0. Installation 
1. Clone the repository recursively. 
```bash
git clone --recurse-submodules https://github.com/Ido-zh/cmr_reverse.git
```

2. Create a venv using the requirements.txt
```bash 
python3 -m venv revimg
source revimg/bin/activate
pip install -r requirements.txt
```

3. Install the [modified nnUNet](https://github.com/Ido-zh/nnUNet-phys-seg) based on [nnUNetV1](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1).
```bash 
cd nnunet
pip install -e .
```
Set the environmental variables required:

```bash
export nnUNet_raw_data_base="/path/to/nnunetdata/raw"
export nnUNet_preprocessed="/path/to/nnunetdata/preprocessed"
export RESULTS_FOLDER="/path/to/nnunetdata/results"
```


## 1. Download weights and required data.
Please make sure the environmental variables in nnUNet have been properly set. 
```bash
bash download_ddpm.sh
bash download_nnunet_trainingdata.sh
bash download_nnunet_weights.sh
```

## 2. Cookbook
### 1.1 Perform reverse imaging on the ACDC training split.
```bash
python cine_reverse_all.py  
```

### 1.2 Segmentation inference with the trained nnUNets.
We follow the nnUNet tradition of data organization. Any test case must be named as `[case_identifier]_0000.nii.gz`. The following command will perform inference on test CMR short-axis images.
```bash
nnUNet_predict -i /path/to/images\
       -o /path/to/predictions\
       -t Task900_ACDC_Phys \
       -tr nnUNetTrainerV2_InvGreAug\
        -m 2d -f 0 1 2 3 4 \
        --overwrite
```

