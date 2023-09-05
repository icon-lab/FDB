# FDB

Official PyTorch implementation of FDB as described in the [paper](https://arxiv.org/abs/2308.01096)

Muhammad U. Mirza, Onat Dalmaz, Hasan A. Bedel, Gokberk Elmas, Yilmaz Korkmaz, Alper Gungor, Salman UH Dar, Tolga Çukur, "Learning Fourier-Constrained Diffusion Bridges for MRI Reconstruction", arXiv 2023.

<img src="./figures/ddpm_vs_fdb.png" width="600px">

## Dependencies

```
python==3.8.13
blobfile==2.0.2
h5py==3.9.0
imageio==2.22.1
mpi4py==3.1.4
numpy==1.24.4
Pillow==10.0.0
torch==2.0.1
```

## Installation
- Clone this repo:
```
git clone https://github.com/icon-lab/FDB
cd FDB
```

## Train

<br />

For Single-Coil
```
python train.py --data_dir /path_to_data/ --log_interval 5000 --save_dir 'model_singlecoil' --save_interval 5000 --image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --noise_schedule cosine --lr 1e-4 --batch_size 1 --lr_anneal_steps 100000 --undersampling_rate 2 --data_type 'singlecoil'
```
For Multi-Coil
```
python train.py --data_dir /path_to_data/ --log_interval 5000 --save_dir 'model_multicoil' --save_interval 5000 --image_size 384 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --noise_schedule cosine --lr 1e-4 --batch_size 1 --lr_anneal_steps 15000 --undersampling_rate 8 --data_type 'multicoil'
```
<br />

## Inference

<br />

For Single-Coil
```
python sample.py --model_path model_singlecoil/ema_0.9999_100000.pt --data_path /path_to_data/ --image_size 256 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1500 --noise_schedule cosine --save_path results_singlecoil --num_samples 1 --batch_size 1 --data_type 'singlecoil' --R 5 --contrast 'T1'
```
For Multi-Coil
```
python sample.py --model_path model_multicoil/ema_0.9999_015000.pt --data_path /path_to_data/ --image_size 384 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3 --diffusion_steps 1000 --noise_schedule cosine --save_path results_multicoil --num_samples 1 --batch_size 1 --data_type 'multicoil' --R 8 --contrast 'FLAIR'
```

<br />
<br />

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@misc{mirza2023learning,
      title={Learning Fourier-Constrained Diffusion Bridges for MRI Reconstruction}, 
      author={Muhammad U. Mirza and Onat Dalmaz and Hasan A. Bedel and Gokberk Elmas and Yilmaz Korkmaz and Alper Gungor and Salman UH Dar and Tolga Çukur},
      year={2023},
      eprint={2308.01096},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
For any questions, comments and contributions, please contact Usama Mirza (usama.mirza.819[at]gmail.com ) <br />

(c) ICON Lab 2023
