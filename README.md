## Introduction

This github provides the implementation used for the results in "Lipschitz Continuous Autoencoders in Application to Anomaly Detection" accepted for AISTATS 2020 [[paper]](http://proceedings.mlr.press/v108/kim20c/kim20c.pdf)[[supplementary]](http://proceedings.mlr.press/v108/kim20c/kim20c-supp.pdf).

## Directory tree
<pre>
.
├── data_celeba
│   └── readme_celeba.md
├── Datapreprocessing_CelebA.ipynb
├── README.md
└── src
    ├── architecture.py
    ├── base.py
    ├── configuration.py
    ├── dataset
    │   ├── celeba.py
    │   ├── fmnist.py
    │   ├── kdd99.py
    │   ├── main.py
    │   └── mnist.py
    ├── LCAE_trainer.py
    ├── main.py
    ├── other_methods
    │   ├── alad
    │   │   ├── alad_trainer.py
    │   │   ├── architecture.py
    │   │   └── run_alad.py
    │   └── svdd
    │       ├── architecture.py
    │       ├── deepSVDD_trainer.py
    │       └── run_svdd.py
    ├── run_lcae.py
    └── utils.py
</pre>
## Commands to run experiments

We provide commands to run experiments in manuscript and supplementary material. Explanations about arguments are attached in 'Explanation about argments' in this document. Before running experiments on CelebA, please download and pre-process dataset using 'Datapreprocessing_CelebA.ipynb'.

(1) Proposed method

1. (KDD99)
To run experiments on Table 4 in the manuscript, please execute the following command at the src folder.
```bash
# KDD99 with uncontaminated training dataset (0%)
python3 main.py 'lcae' 'kdd99' -n 20 -e 200 -m 0.0 -l 10.0 -C 0.7 -c 0 -g 0
# KDD99 with contaminated training dataset (5%)
python3 main.py 'lcae' 'kdd99' -n 20 -e 200 -m 0.0 -l 10.0 -C 0.7 -c 5 -g 0
```
To run experiments on Tables 2 and 3 in the manuscript and Tables 5 and 6 in the supplementary material, please execute the above command after setting n to be 10 and varying m, l, and C.

2. (MNIST and Fashion-MNIST)
To run experiments on Tables 7 and 8 in the supplementary material, please execute the following command at the src folder.
```bash
# MNIST with uncontaminated training dataset (0%)
python3 main.py 'lcae' 'mnist'  -n 10 -e 50 -m 2.0 -l 2.0 -C 0.8 -c 0 -g 0 -N 0
# MNIST with contaminated training dataset (5%)
python3 main.py 'lcae' 'mnist' -n 10 -e 50 -m 2.0 -l 2.0 -C 0.8 -c 5 -g 0 -N 0
# Fashion-MNIST with uncontaminated training dataset (0%)
python3 main.py 'lcae' 'fmnist' -n 10 -e 50 -m 2.0 -l 2.0 -C 0.8 -c 0 -g 0 -N 0
# Fashion-MNIST with contaminated training dataset (5%)
python3 main.py 'lcae' 'fmnist' -n 10 -e 50 -m 2.0 -l 2.0 -C 0.8 -c 5 -g 0 -N 0
```
  
3. (CelebA)
To run experiments on Table 4 in the manuscript, please execute the following command at the src folder.
```bash
# CelebA (glasses) with uncontaminated training dataset (0%)
python3 main.py 'lcae' 'celeba' -n 5 -e 100 -m 2.0 -l 2.0 -C 0.8 -c 0 -g 0 -N -1 -a 15
# CelebA (glasses) with contaminated training dataset (5%)
python3 main.py 'lcae' 'celeba' -n 5 -e 100 -m 2.0 -l 2.0 -C 0.8 -c 5 -g 0 -N -1 -a 15
```
(2) Other methods

1. (KDD99)
To run experiments on Table 4 in the manuscript, please execute the following command at the src folder.
```bash
# Deep SVDD: KDD99 with uncontaminated training dataset (0%)
python3 main.py 'svdd' 'kdd99' -n 20 -e 200 -c 0 -g 0
# Deep SVDD: KDD99 with contaminated training dataset (5%)
python3 main.py 'svdd' 'kdd99' -n 20 -e 200 -c 5 -g 0
# ALAD: KDD99 with uncontaminated training dataset (0%)
python3 main.py 'alad' 'kdd99' -n 20 -e 200 -c 0 -g 0
# ALAD: KDD99 with contaminated training dataset (5%)
python3 main.py 'alad' 'kdd99' -n 20 -e 200 -c 5 -g 0
```

2. (MNIST and Fashion-MNIST)
To run experiments on Tables 7 and 8 in the supplementary material, please execute the following command at the src folder.
```bash
# Deep SVDD: MNIST with uncontaminated training dataset (0%)
python3 main.py 'svdd' 'mnist' -n 10 -e 50 -c 0 -g 0 -N 0
# Deep SVDD: MNIST with contaminated training dataset (5%)
python3 main.py 'svdd' 'mnist' -n 10 -e 50 -c 5 -g 0 -N 0
# ALAD: MNIST with uncontaminated training dataset (0%)
python3 main.py 'alad' 'mnist' -n 10 -e 50 -c 0 -g 0 -N 0
# ALAD: MNIST with contaminated training dataset (5%)
python3 main.py 'alad' 'mnist' -n 10 -e 50 -c 5 -g 0 -N 0
# Deep SVDD: Fashion-MNIST with uncontaminated training dataset (0%) 
python3 main.py 'svdd' 'fmnist' -n 10 -e 50 -c 0 -g 0 -N 0
# Deep SVDD: Fashion-MNIST with contaminated training dataset (5%)
python3 main.py 'svdd' 'fmnist' -n 10 -e 50 -c 5 -g 0 -N 0
# ALAD: Fashion-MNIST with uncontaminated training dataset (0%)
python3 main.py 'alad' 'fmnist' -n 10 -e 50 -c 0 -g 0 -N 0
# ALAD: Fashion-MNIST with contaminated training dataset (5%)
python3 main.py 'alad' 'fmnist' -n 10 -e 50 -c 5 -g 0 -N 0
```
3. (CelebA)
To run experiments on Table 4 in the manuscript, please execute the following command at the src folder.
```bash
# Deep SVDD: CelebA (glasses) with uncontaminated training dataset (0%)
python3 main.py 'svdd' 'celeba' -n 5 -e 100 -c 0 -g 0 -N -1 --a 15
# Deep SVDD: CelebA (glasses) with contaminated training dataset (5%)
python3 main.py 'svdd' 'celeba' -n 5 -e 100 -c 5 -g 0 -N -1 --a 15
# ALAD: CelebA (glasses) with uncontaminated training dataset (0%)
python3 main.py 'alad' 'celeba' -n 5 -e 100 -c 0 -g 0 -N -1 --a 15
# ALAD: CelebA (glasses) with contaminated training dataset (5%)
python3 main.py 'alad' 'celeba' -n 5 -e 100 -c 5 -g 0 -N -1 --a 15
```

## Explanations about argments

a (or attribute): attribute class in experiments with CelebA. Default value is 0.<br/>
C (or lipschitzconstant): Lipschitz constant ('K' in the equation (2) in the manuscript) in experiments with LCAE. Default value is 0.95.<br/>
c (or contamratio): contaminated ratio in training set. Default value is 0. Possible choices are 0 and 5.<br/>
d (or decide): number or list of experiment. Possible choices are 'int' and 'list'.<br/>
dataset: the name of the dataset you want to run the experiment on. Possible choices are 'kdd99', 'mnist', 'fmnist', and 'celeba'.<br/>
degree: degree of L-p norm in experiments with ALAD. Default value is 1.<br/>
e (or epoch): number of epochs. Default value is 50.<br/>
enable_dzz: enable dzz discriminator in experiments with ALAD. Default value is True.<br/>
enable_early_stop: enable early_stopping in experiments with ALAD. Default value is True.<br/>
enable_sm: enable TF summaries in experiments with ALAD. Default value is True.<br/>
g (or gpu): which gpu to use. Default value is 0.<br/>
m: mode/method for discriminator loss in experiments with ALAD. Possible choices are 'cross-e' and 'fm'. Default value is 'fm'.<br/>
m (or mmdweight): mmd loss weight ('lambda' in the equation (2) in the manuscript) in experiments with LCAE. Default value is 0.0.<br/>
model: the model name of the example you want to run. Possible choices are 'lcae', 'svdd', and 'alad'.<br/>
N (or normalclass): normal class in experiments on MNIST, Fashion-MNIST, and CelebA. Possible choices are 0, 1, ..., 9 in MNIST and Fashion-MNIST, and-1 and 1 in CelebA. Default value is 0.<br/>
n (or number): number or list of experiment. Default value is 10.<br/>
sn: enable spectral_norm in experiments with ALAD. Default value is True.<br/>
t (or tsne): tsne for original/reconstructed data in experiments with LCAE. Default value is False.<br/>
l (or lipschitzweight): Lipschitz loss weight ('phi' in the equation (2) in the manuscript) in experiments with LCAE. Default value is 0.0.<br/>
w: weight in experiments with ALAD. Default value is 0.1.<br/>

## Reproducibility

The results of the proposed method and Deep SVDD are reproducible while that of ALAD is not. For Deep SVDD and ALAD, we adopt and modified codes published by [1] and [2], respectively. Both codes were not reproducible when GPU is used, so we revised the code to guarantee reproducibility. The modified Deep SVDD implementation gives consistent results in the system environment we elaborated in 'Dependency'. In contrast, the code from ALAD work is built in TensorFlow of low version, so reproducibility can not be achieved. We tried to reduce oscilations as small as possible. For details about the reproducibility issue in TensorFlow, please refer https://github.com/NVIDIA/tensorflow-determinism.

## Dependency

argparse                      1.1<br/>
imageio                       2.3.0<br/>
ipykernel                     4.8.2<br/>
json                          2.0.9<br/>
logging                       0.5.1.2<br/>
matplotlib                    3.1.2<br/>
numpy                         1.16.1<br/>
pandas                        0.25.3<br/>
pickleshare                   0.7.4<br/>
PIL                           5.1.0<br/>
python                        3.6.5<br/>
skimage                       0.16.2<br/>
sklearn                       0.20.2<br/>
tensorflow                    1.9.0<br/>
torch                         1.0.1.post2<br/>
torchvision                   0.2.2<br/> 
urllib3                       1.22<br/>

## References for the implementation

[1] Ruff, Lukas, et al. "Deep one-class classification." International conference on machine learning. 2018. URL: https://github.com/lukasruff/Deep-SVDD-PyTorch<br/>
[2] Zenati, Houssam, et al. "Adversarially learned anomaly detection." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018. URL: https://github.com/houssamzenati/Adversarially-Learned-Anomaly-Detection
