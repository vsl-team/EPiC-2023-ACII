# EPiC-2023-ACII
Solution for [EPiC 2023 competition](https://github.com/Emognition/EPiC-2023-competition) by VSL team - Pattern Recognition Lab, Chonnam Nat'l Univ.

1. Create conda environment
```
conda create -n epic_vsl python=3.9
conda activate epic_vsl
pip install -r requirements.txt
```
2. To train the model, edit OUT_DIR and DATA_DIR in `config/base_cfg.yaml`, or directly in `scripts/*_train.sh`. The scripts for training our 3 attempts are in scripts. The pretrained weights, prediction, completed configs of our 3 attempts could be found in [this link](https://ejnu-my.sharepoint.com/:f:/g/personal/vthuynh_jnu_ac_kr/EsYrz5b6DPJChfpnvZLZbwkBnlh5HhKaPgfc4telwsjuuQ?e=gHMI1z).

Our repo are trained on Debian 11.5 with CUDA 12.1 and CUDNN 8.
