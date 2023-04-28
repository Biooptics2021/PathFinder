# PathFinder: AI bsaed pathological biomarker finder
<img src="https://github.com/Biooptics2021/PathFinder/blob/main/Docs/figure1.png?raw=true" width="700">

### [Project Page](https://github.com/Biooptics2021/PathFinder/)  |  [Paper](https://www.nature.com/articles/s42256-023-00635-3)  |  [Research Highlight](https://www.nature.com/articles/s44222-023-00069-x)
**Note:** Try [Quick Discovery](#quick-discovery) to implement PathFinder with the pre-trained network.

© This code is made available for non-commercial academic purposes. 

## Contents
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Pre-requisites and Environment](#pre-requisites-and-environment)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Biomarker Discovery](#biomarker-discovery)
- [Acknowledgements](#acknowledgements)

## Overview
Tissue biomarkers are crucial for cancer diagnosis, prognosis assessment and treatment planning. However, there are few known biomarkers that are robust enough to show true analytical and clinical value. Deep learning (DL)-based computational pathology can be used as a strategy to predict survival, but the limited interpretability and generalizability prevent acceptance in clinical practice. Here we present an interpretable human-centric DL-guided framework called PathFinder (Pathological-biomarker-finder) that can help pathologists to discover new tissue biomarkers from well-performing DL models. By combining sparse multi-class tissue spatial distribution information of whole slide images with attribution methods, PathFinder can achieve localization, characterization and verification of potential biomarkers, while guaranteeing state-of-the-art prognostic performance. Using PathFinder, we discovered that spatial distribution of necrosis in liver cancer, a long-neglected factor, has a strong relationship with patient prognosis. We therefore proposed two clinically independent indicators, including necrosis area fraction and tumour necrosis distribution, for practical prognosis, and verified their potential in clinical prognosis according to criteria derived from the Reporting Recommendations for Tumor Marker Prognostic Studies. Our work demonstrates a successful example of introducing DL into clinical practice in a knowledge discovery way, and the approach may be adopted in identifying biomarkers in various cancer types and modalities.


For more details, please see our paper: 
["*Deep learning supported discovery of biomarkers for clinical prognosis of liver cancer (2023)*"](https://www.nature.com/articles/s42256-023-00635-3).

### Framework of PathFinder
<center><img src="https://github.com/Biooptics2021/PathFinder/blob/main/Docs/Figure1-framework.png?raw=true" width="900" align="middle" /></center>

## Directory Structure

```bash
PathFinder
    └──WSI_decoupling
          ├── decoupling.py
          ├── inference.py
          ├── visualization.py
          └── PaSegNet
                ├── train.py
                └── test.py
    ├── Prognosis
          ├── data_loaders.py
          ├── train_TCGA_CV.py
          ├── train_TCGA_test_QHCG.py
          ├── train_test.py
          ├── utils.py
          ├── Data_prepare
                ├── cut_heatmap.py
                └── Generate_prognostic_patches.py
          └── Networks
                ├── M2M_network.py
                ├── Macro_networks.py
                └── Micro_networks.py
    ├── Discovery
          ├── data_loaders.py
          ├── networks.py
          ├── attribution.ipynb
          ├── verification.ipynb
          ├── ckpt
                └── trained_model.pth
          └── segmap
                └── segmap_example.npy
    └──Data
          └── WSIs and clinical information
```

* **WSI_decoupling**: *Get the macro mode (3D-numpy-array of multi-class tissue probability heatmaps) of WSIs.*
* **Prognosis**: *Train prognostic deep neural networks (MacroNet, MicroNet and M2MNet) for cancer prognosis.*
* **Discovery**: *Use attribution methods to find important features for further survival analyses and biomarker discovery.*
* **Data**: *WSI and clinical information source data.*


## Pre-requisites and Environment

### Our Environment
* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 3090 x 4)
* Python (3.7.9), PyTorch (version 1.8.0), Lifelines (version 0.25.11), NumPy (version 1.19.2), Pandas (version 1.2.2), Albumentations (version 0.5.2), OpenCV (version 4.5.1), Pillow (version 7.2.0), OpenSlide (version 1.1.2), Captum (version 0.2.0), SciPy (version 1.4.1), Seaborn (version 0.9.0), Matplotlib (version 3.1.1).
### Environment Configuration
1. Create a virtual environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).
   ```
   $ conda create -n pathfinder python=3.7.9
   $ conda activate pathfinder
   $ pip install torch==1.8.0+cu110 torchvision==0.8.2+cu110 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```
      *Note:  `pip install` command is required for Pytorch installation.*
      
2. To try out the Python code and set up environment, please activate the `pathfinder` environment first:

   ``` shell
   $ conda activate pathfinder
   $ cd PathFinder/
   ```
3. For ease of use, you can just set up the environment and run the following:
   ``` shell
   $ pip install -r requirements.txt
   ```

## Data Preparation

### Data Format
* WSIs and clinical information of patients are used in this project. Raw WSIs are stored as ```.svs```, ```.mrxs``` or ```.tiff``` files. Clinical information are stored as ```.csv``` files. 

### Generate Macro Mode

* WSIs are first processed by PaSegNet to get multi-class tissue probability heatmaps (macro mode), which sorted as ```.npy``` files.
The macro mode of WSIs can be by generated by calling:
 
    ``` shell
    $ cd ./WSI_decoupling
    $ python decoupling.py
    ```
* To cut the empty area of macro mode and get square input for training, call:
    ``` shell
    $ cd ./Prognosis/Data_prepare
    $ python cut_heatmap.py
    ```

### Generate Micro Mode

* Tumor pathces (micro mode) are extracted based on macro mode and WSIs, and stored as ```.tif``` files. 
The micro mode of WSIs can be by generated by calling:
 
    ``` shell
    $ cd ./Prognosis/Data_prepare
    $ python Generate_prognostic_patches.py
    ```
    
### Data Usage

* Macro mode and clinical information is used to train ```MacroNet```, micro mode and clinical information is used to train ```MicroNet```, both macro mode, micro mode and clinical information is used to train ```M2MNet```. 

### Data Distribution

```bash
DATA_ROOT_DIR/
    └──DATASET_DIR/
         ├── clinical_information                       + + + 
                ├── Hospital_1.csv                          +
                ├── Hospital_2.csv                          +
                └── ...                                     +
         ├── WSI_data                                       +
                ├── Hospital_1                              +
                       ├── slide_1.svs                      +
                       ├── slide_2.svs                Source Data
                       └── ...                              +
                ├── Hospital_2                              +
                       ├── slide_1.svs                      +
                       ├── slide_2.svs                      +
                       └── ...                              +
                └── ...                                 + + +
         ├── macro_mode                                 + + +
                ├── Hospital_1                              +
                       ├── slide_1_heatmaps.npy             +
                       ├── slide_2_heatmaps.npy             +
                       └── ...                              +
                ├── Hospital_2                              +
                       ├── slide_1_heatmaps.npy             +
                       ├── slide_2_heatmaps.npy             +
                       └── ...                              +
                └── ...                                     +
         └── micro_mode                            Processed Data
                ├── Hospital_1                              +
                       ├── slide_1                          +
                              ├── patch_1.tif               +
                              ├── patch_2.tif               +
                              └── ...                       +
                       ├── slide_2                          +
                              ├── patch_1.tif               +
                              ├── patch_2.tif               +
                              └── ...                       +
                       └── ...                              +
                └── ...                                 + + +             
```
DATA_ROOT_DIR is the base directory of all datasets (e.g. the directory to your SSD). DATASET_DIR is the name of the folder containing data specific to one experiment.


## Training and Evaluation

### K-fold Cross Validation
After data preparation, MacroNet can be trained and tested on TCGA data in a k-fold cross-validation by calling:
``` shell
$ cd ./Prognosis
$ python train_TCGA_CV.py
```

### Independent Hospital Test
The generalization ability of MacroNet can be tested by calling:
``` shell
$ cd ./Prognosis
$ python train_TCGA_test_QHCG.py
```

### Training and Evaluation of MicroNet and M2MNet
To train and evaluate MicroNet and M2MNet, import corresponding data loader and network architecture in ```./Prognosis/train_test.py```. Data loaders can be found in ```./Prognosis/data_loaders.py```, network architectures can be found in ```./Prognosis/Networks```.


## Biomarker Discovery
### Quick Discovery 
* We provide simple and user-friendly Jupyter notebook ```./Discovery/attribution.ipynb``` as a quick discovery demo to visualize attribution and characterize potential biomarkers.
* Please click here to download [pre-trained MacroNet](https://cloud.tsinghua.edu.cn/f/5b240f9c55844142b54a/?dl=1) and place the file in ```./Discovery/ckpt```. Also, the macro mode of WSIs ```./Discovery/segmap``` are provided as source data.
* Before you launch the notebooks, please configure an environment following the instruction in [Environment Configuration](#environment-configuration). 
* Then, you can launch the notebook and find new biomarkers with the inspiration of AI:

<center><img src="https://github.com/Biooptics2021/PathFinder/blob/main/Docs/attribution.png?raw=true" width="900" align="middle" /></center>

### Verification
Biomarker verification according to REMARK and survival analyses are performed in ```./Discovery/verification.ipynb```


## Acknowledgements
- Prognosis training and test code base structure was inspired by [Pathomic Fusion](https://github.com/mahmoodlab/PathomicFusion).
- Tumor necrosis distribution score (TND) was inspired by [TILAb-Score](https://github.com/TissueImageAnalytics/TILAb-Score).
- Readme structure was inspired by [DeepCAD-RT](https://github.com/cabooster/DeepCAD-RT).

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our paper.

- Liang, J., Zhang, W., Yang, J. et al. Deep learning supported discovery of biomarkers for clinical prognosis of liver cancer. Nat Mach Intell (2023). [https://doi.org/10.1038/s42256-023-00635-3](https://doi.org/10.1038/s42256-023-00635-3)
```
@article{liang2023deep,
  title={Deep learning supported discovery of biomarkers for clinical prognosis of liver cancer},
  author={Liang, Junhao and Zhang, Weisheng and Yang, Jianghui and Wu, Meilong and Dai, Qionghai and Yin, Hongfang and Xiao, Ying and Kong, Lingjie},
  journal={Nature Machine Intelligence},
  pages={1--13},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
