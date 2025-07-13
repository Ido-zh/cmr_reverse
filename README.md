# Reverse Imaging in Cardiac Magnetic Resonance Imaging

In MRI, images acquired with different RF pulses are manefestations of the same underlying tissue properties including magnetization strength $\mathrm{M}_0$, $\mathrm{T}_1$, $\mathrm{T}_2$. ***Reverse imaging*** estimates the underlying physical properties of tissues that have caused the observed images, from qualitative images such as bSSFP cine. The physical properties can be used for physics-grounded cross-sequence synthesis and data augmentation in training segmentation models.


## 0. Installation 
1. Clone the repository recursively. 
```bash
git clone --recurse-submodules https://github.com/Ido-zh/cmr_reverse.git
```
2. Create a venv using the requirements.txt
