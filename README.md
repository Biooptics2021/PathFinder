# PathFinder: AI bsaed pathological biomarker finder
© This code is made available for non-commercial academic purposes. 

## Contents
- [Directory Structure](#directory-structure)
- [Pre-requisites and Environment](#pre-requisites-and-environment)
- [Data Preparation](#data-preparation)
- [Training and Evaluation](#training-and-evaluation)
- [Biomarker Discovery](#biomarker-discovery)
- [Characterization and Verification](#characterization-and-verification)
- [Acknowledgements](#acknowledgements)

## Directory Structure
* **WSI_decoupling**: *Get the macro mode (3D-numpy-array of multi-class tissue probability heatmaps) of WSIs.*
* **Prognosis**: *Train prognostic deep neural networks (MacroNet, MicroNet and M2MNet) for cancer prognosis.*
* **Discovery**: *Use attribution methods to find important features for further survival analyses and biomarker discovery.*
* **Data**: *WSI and clinical information source data.*

## Pre-requisites and Environment

* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on Nvidia GeForce RTX 3090 x 4)
* Python (3.7.9),PyTorch (version 1.8.0), Lifelines (version 0.25.11), NumPy (version 1.19.2), Pandas (version 1.2.2), Albumentations (version 0.5.2), OpenCV (version 4.5.1), Pillow (version 7.2.0), OpenSlide (version 1.1.2), Captum (version 0.2.0), SciPy (version 1.4.1), Seaborn (version 0.9.0), Matplotlib (version 3.1.1).
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
* WSIs and clinical information of patients are used in this project. Raw WSIs are stored as ```.svs```, ```.mrxs``` or ```.tiff``` files. Clinical information are stored as ```.csv``` files. 
* WSIs are first processed by PaSegNet to get multi-class tissue probability heatmaps (macro mode), which sorted as ```.npy``` files.
The macro mode of WSIs can be by generated by calling:
 
    ``` shell
    $ cd ./WSI_decoupling
    $ python decoupling.py
    ```
    To cut the empty area of macro mode and get square input for training, call:
    ``` shell
    $ cd ./Prognosis/Data_prepare
    $ python cut_heatmap.py
    ```
* Tumor pathces (micro mode) are extracted based on macro mode and WSIs, and stored as ```.tif``` files. 
The micro mode of WSIs can be by generated by calling:
 
    ``` shell
    $ cd ./Prognosis/Data_prepare
    $ python Generate_prognostic_patches.py
    ```
* Macro mode and clinical information is used to train ```MacroNet```, micro mode and clinical information is used to train ```MicroNet```, both macro mode, micro mode and clinical information is used to train ```M2MNet```. 
The Data distribution is like:
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
After data preparation, MacroNet can be trained and tested on TCGA data in a cross-validation by calling:
``` shell
$ cd ./Prognosis
$ python train_TCGA_CV.py
```
The generalization ability of MacroNet can be tested by calling:
``` shell
$ cd ./Prognosis
$ python train_TCGA_test_QHCG.py
```
To train and evaluate MicroNet and M2MNet, import corresponding data loader and network architecture in ```./Prognosis/train_test.py```. Data loaders can be found in ```./Prognosis/data_loaders.py```, network architectures can be found in ```./Prognosis/Networks```.


## Biomarker Discovery
To explore the features networks focus on, run ```./Discovery/attribution.ipynb```.

## Characterization and Verification

## Acknowledgements
- Prognosis training and test code base structure was inspired by [Pathomic Fusion](https://github.com/mahmoodlab/PathomicFusion).
- Tumor necrosis distribution score (TND) was inspired by [TILAb-Score](https://github.com/TissueImageAnalytics/TILAb-Score).

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our paper.
