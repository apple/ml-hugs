# HUGS: Human Gaussian Splats

This repository is a reference implementation for HUGS. HUGS reconstructs both the background scene and an animatable human from a single video using neural radiance fields.

[[Paper](https://arxiv.org/abs/2311.17910)] | [[Project Page](https://machinelearning.apple.com/research/hugs)]

> [**HUGS: Human Gaussian Splats**](https://arxiv.org/abs/2311.17910),            
> [Muhammed Kocabas](https://ps.is.tuebingen.mpg.de/person/mkocabas), 
> [Jen-Hao Rick Chang](https://rick-chang.github.io/), 
> [James Gabriel](https://www.linkedin.com/in/jamescgabriel/), 
> [Oncel Tuzel](https://www.onceltuzel.net/), 
> [Anurag Ranjan](https://anuragranj.github.io/)       
> *IEEE Computer Vision and Pattern Recognition (CVPR) 2024* 

<p float="center">
  <img src="assets/hugs_teaser.png" width="100%" />
</p>

# Getting Started

We tested our system with Ubuntu 22.04.3 using a CUDA 11.7 compatible GPU.

- Clone our repo:
```
git clone --recursive git@github.com:apple/ml-hugs.git
```

- Run the setup script to create a conda environment and install the required packages.
```
source scripts/conda_setup.sh
```

# Preparing the datasets and models

## Datasets
- Download the SMPL neutral body model
    - Register to [SMPL](https://smpl.is.tue.mpg.de/index.html) website.
    - Download v1.1.0 and SMPL UV obj file from the [download](https://smpl.is.tue.mpg.de/download.php) page.
    - Extract the files and rename `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` to `SMPL_NEUTRAL.pkl`.
    - Put the files into `./data/smpl/` folder with the following structure:

        ```
        data/smpl/
        ├── SMPL_NEUTRAL.pkl
        └── smpl_uv.obj
        ```

- Download NeuMan dataset and pretrained models:
    - Data ([download](https://docs-assets.developer.apple.com/ml-research/models/hugs/neuman_data.zip))
    - Pretrained models ([download](https://docs-assets.developer.apple.com/ml-research/models/hugs/hugs_pretrained_models.zip))

    Alternately, run the following script to set up data and pretrained models.
    ```
    source scripts/prepare_data_models.sh
    ```

- Download AMASS dataset for novel animation rendering:
  - AMASS dataset is used for rendering novel poses.
  - We used SFU mocap(SMPL+H G) and MPI_mosh (SMPL+H G) subsets, please download from [AMASS](https://amass.is.tue.mpg.de/download.php).
  - Put the downloaded mocap data in to `./data/` folder.

After following the above steps, you should obtain a folder structure similar to this:

```
data/
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   ├── SMPL_NEUTRAL.pkl
│   ├── smpl_uv.obj
├── neuman
│   └── dataset
│       ├── bike
│       ├── citron
│       ├── jogging
│       ├── lab
│       ├── parkinglot
│       └── seattle
├── MPI_mosh
│   ├── 00008
│   ├── 00031
│   ├── ...
│   └── 50027
└── SFU
    ├── 0005
    ├── 0007
    ├── ...
    └── 0018
```


# Training

To train HUGS on NeuMan dataset, there are three different modes you can choose from: 1. joint human and scene 2. human only, 3. scene only. 

1. Joint human and scene training

    This is the original HUGS setup where jointly optimize human Gaussians and scene Gaussians. 
    ```
    python main.py --cfg_file cfg_files/release/neuman/hugs_human_scene.yaml dataset.seq=lab
    ```

2. Human only training

    This mode only optimizes the Triplane+MLP model introduced in HUGS.

    ```
    python main.py --cfg_file cfg_files/release/neuman/hugs_human.yaml dataset.seq=lab
    ```


3. Scene only training

    This setup is identical to original 3DGS paper. Here we provide the script to run it on the NeuMan dataset

    ```
    python main.py --cfg_file cfg_files/release/neuman/hugs_scene.yaml dataset.seq=lab
    ```

`cfg_files/release` directory contains the final configuration files we used to train HUGS. Please refer to the [config.py](hugs/cfg/config.py) file to see different config parameters and their meanings.

**Note**: Expect to see slight differences compared to the pretrained models. This is due to the inherent randomness in the rendering process, which makes achieving deterministic results across multiple runs challenging, even when proper seeding is applied. So it is expected to obtain results slightly different than what is reported in the paper.

# Evaluation and Animation

Here we show how to perform evaluation with the pretrained models on the NeuMan dataset.

```
python scripts/evaluate.py -o <<path to the output directory>>
```

This command will print out the PSNR, SSIM, and LPIPS metrics for a given pretrained model.

# Citation
```
@inproceedings{
    kocabas2024hugs,
    title={{HUGS}: Human Gaussian Splatting},
    author={Kocabas, Muhammed and Chang, Jen-Hao Rick and Gabriel, James and Tuzel, Oncel and Ranjan, Anurag},
    booktitle = {2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2024},
    url={https://arxiv.org/abs/2311.17910}
}
```

# License
The code is released under the [LICENSE](LICENSE) terms.
