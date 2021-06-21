# Effect of large-scale pre-training on full and few-shot transfer learning for natural and medical images
*by Mehdi Cherti, Jenia Jitsev* [\[arXiv:2106.00116\]](https://arxiv.org/abs/2106.00116)

## Introduction

In this repository, we provide the code for reproducing the experiments on large-scale pre-training and transfer learning for the paper *"Effect of large-scale pre-training on full and few-shot transfer learning for natural and medical images"* ([arXiv:2106.00116](https://arxiv.org/abs/2106.00116)).

We provide instructions on how to download the different datasets used in the paper.
We provide the pre-trained models, the instructions to fine-tune a pre-trained model on one of the datasets considered in the paper, as well as new datasets.

Organization
------------

    ├── LICENSE            <- MIT License
    ├── README.md          <- Main doc README on reproducing the experiments
    ├── requirements.txt   <- The requirements file for reproducing the experiments environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    ├── setup.py           <- makes `transfer learning` package installable so it can be imported
    ├── pretrain.py        <- code for pre-training 
    ├── finetune.py        <- code for fine-tuning 
    ├── transfer_learning  <- Source code
    │   ├── dataloaders    <- Dataset loaders
    │   ├── datasets       <- Datasets
    │   ├── finetuning     <- utilities used for Bit-HyperRule
    │   ├── lr_scheduler   <- Learning rate schedulers
    │   ├── models         <- Neural network architecture definitions
    │   ├── optim          <- Optimizers
    ├── datasets           <- Folder where datasets are stored
    ├── pretrained_models  <- Folder where pre-trained models are stored

--------

## Installation

Steps to install the package:

- `pip install -r requirements.txt`
- `python setup.py develop`

## Obtaining Data

The folder `datasets` will be used to store the datasets. In each subfolder of `datasets`, there will be one dataset.
Following are in the instructions to download each dataset considered in the paper.

### Obtaining source datasets for pre-training

#### CheXpert v1.0

1. Fill the form and download the dataset from <https://stanfordmlgroup.github.io/competitions/chexpert/>, and extract the archive
2. Put the folder `CheXpert-v1.0` in `datasets`

#### MIMIC-CXR v2.0

1. Follow the instructions at <https://physionet.org/content/mimic-cxr-jpg/2.0.0/> (section **Files**) and extract the archive
2. Put the folder `mimic-cxr-jpg` in `datasets`

#### NIH Chest Xray-14

1. Download all the files at <https://nihcc.app.box.com/v/ChestXray-NIHCC>, and extract all the archives in `images/`
2. Create a folder `NIH-ChestXRay-14` inside `datasets`, and all the files and the folder `images` in `NIH-ChestXRay-14`

#### PadChest

1. Download the complete dataset from <https://bimcv.cipf.es/bimcv-projects/padchest/> after filling the form and extract the archive
2. Unzip all the zip files `0.zip`, `1.zip`,...,`55.zip` inside `BIMCV-PadChest-FULL`
3. Put the folder `BIMCV-PadChest-FULL` in `datasets` and rename it to `PadChest`

### Obtaining target datasets for transfer

#### Oxford Flowers-102

1. Download the dataset from <https://bit.ly/3xBF9XZ> and extract the archive
2. Put the folder `oxford-102-flowers` in `datasets`

If you would like to re-create the dataset from the original version, follow these steps:

1. Download the dataset from <https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz> and extract the archive
2. `cat valid.txt train.txt > mod_test.txt`
3. `cat test.txt > mod_train.txt`
5. `wget https://raw.githubusercontent.com/SLAMPAI/large-scale-pretraining-transfer/master/scripts/datasets/flowers_to_image_folder.py;python flowers_to_image_folder.py`, this will create a folder `mod_train` and a folder `mod_test`
6.  Move the folder to `oxford-102-flowers` to `datasets`

#### Oxford-III Pets

1. Download the dataset from <https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz> and extract the archive
2. Put the folder `oxford-iiit-pet` in `datasets`

#### COVIDx

1. Download the dataset from <https://www.kaggle.com/andyczhao/covidx-cxr2/version/3> and extract the archive inside a new folder `COVIDx-CXR2`
2. Put the folder `COVIDx-CXR2`  in `datasets`

#### Tuberculosis

1. Download the dataset from <https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities/version/1> and extract the archive inside a new folder `Tuberculosis_dataset`
2. Put the folder `Tuberculosis_dataset` in `datasets`

## How to run

### Pre-training experiments

Note that you need Horovod to execute pre-training experiments.
Check <https://horovod.readthedocs.io/en/stable/running_include.html> or <https://horovod.readthedocs.io/en/stable/mpi.html> to see how to run Horovod,
depending on your setup.

For instance, here is how you can run pre-training with Horovod, using 4 GPUs:

`horovodrun -np 4 python pretrain.py --config-file configs/chexpert_mimic_nih_padchest_bit50x1.yaml`

This will run pre-training for a ResNet-50x1 BiT model, on the concatenation of CheXpert, MIMIC-CXR, NIH Chest-Xray and PadChest.
You can check the other config files in `configs/` for other pre-training experiments, and run them in the same manner.

### Pre-trained models

We provide models with pre-trained weights different network sizes (ResNet-50x1, ResNet-152x4) and on various source datasets of different type and size.
You can download all the models from <https://bit.ly/34MYsBc>.

Each model has its own folder, named following the template `<DATASET>_<MODEL>`,
e.g., `chexpert_mimic_nih_padchest_bit152x4` is a ResNet152x4 pre-trained on
the concatenation of CheXpert, MIMIC-CXR, NIH Chest-Xray and PadChest.
You can select one or several folders and download them
directly. Once the archive is downloaded please extract it in the folder
`pretrained_models`.

### Fine-tuning transfer experiments

#### CIFAR-10 example

`python finetune.py --pretrain-config-file configs/imagenet1k_bit50x1.yaml --finetune-config-file configs/finetune/cifar10.yaml --logdir cifar10_finetuning`

This will fine-tune an R50x1 (pre-trained on ImageNet-1k) on CIFAR-10.
The file `configs/finetune/cifar10.yaml` contains the hyper-parmeters used in fine-tuning.

Inside the log directory `cifar10_finetuning`, you will find a log file and a tensorboard file
that you can use to visualize the learning curve with different metrics.


#### Tuberculosis example

`python finetune.py --pretrain-config-file configs/chexpert_mimic_nih_padchest_bit50x1.yaml --finetune-config-file configs/finetune/tuberculosis_full.yaml --logdir tuberculosis_finetuning`

This will fine-tune on an R50x1 (pre-trained on the concatenation of CheXpert, MIMIC-CXR, NIH Chest-Xray and PadChest) on the Tuberculosis dataset.
The file `configs/finetune/tuberculosis_full.yaml` contains the hyper-parameters used in fine-tuning.

Inside the log directory `tuberculosis_finetuning`, you will find a log file and a tensorboard file
that you can use to visualize the learning curve with different metrics.

#### New dataset?

You can also fine-tune one of the pre-trained models on a new dataset.
You might need a different data loader depending on your dataset structure.
The easiest would be to use an image folder compatible with [TorchVision's ImageFolder](https://pytorch.org/vision/stable/datasets.html#imagefolder), where
each subfolder of the image folder contains the images belonging to one of the classes.

Following are the steps to fine-tune on a dataset with an image folder structure.

1. `cp configs/finetune/template_image_folder.yaml configs/finetune/your_new_dataset.yaml`
2. change `train_dir` by the training directory
3. change `val_dir` by the val or test directory
4. change `nb_classes` by the number of classes
5. Train, using for instance `python finetune.py --pretrain-config-file configs/chexpert_mimic_nih_padchest_bit50x1.yaml --finetune-config-file configs/finetune/your_new_dataset.yaml --logdir your_new_dataset_finetuning`

## Plot results

We provide all the results as a set of CSV files in the folder `results`.
You can use the notebook in `notebooks/plots.ipynb` to regenerate the figures from the paper.

## Citation

If you find this work helpful, please cite our paper:
```
@article{cherti2021effect,
  title={Effect of large-scale pre-training on full and few-shot transfer learning for natural and medical images},
  author={Cherti, Mehdi and Jitsev, Jenia},
  journal={arXiv preprint arXiv:2106.00116},
  year={2021}
}
```

## Acknowledgements

- Thanks to BiT authors <https://github.com/google-research/big_transfer>. We used BiT architecture, training procedures and BiT-HyperRule from their code.
- Thanks to TorchXrayVision authors <https://github.com/mlmed/torchxrayvision>. We used their dataset classes for medical data (CheXpert, MIMIC-CXR, NIH Chest-Xray, PadChest).
- Thanks to Horovod authors <https://github.com/horovod/horovod>, which was used for distributed training. The skeleton of pre-training code is based on <https://github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_imagenet_resnet50.py> from Horovod official examples.
- Thanks to <https://github.com/rwightman/pytorch-image-models>. The structure of the code was inspired from it.
- Thanks to creators and maintainers of openly available X-Ray medical imaging datasets (CheXpert, MIMIC-CXR, NIH Chest-Xray, PadChest, COVIDx, Tuberculosis) that enabled our research
- The authors gratefully acknowledge the Gauss Centre for Supercomputing e.V. (www.gauss-centre.eu) for funding this work by providing computing time through the John von Neumann Institute for Computing (NIC) on the GCS Supercomputers JUWELS, JUWELS Booster at Jülich Supercomputing Centre (JSC). We also acknowledge computing resources from the Helmholtz Data Federation and further computing time provided on supercomputer JUSUF in frame of offer for epidemiology research on COVID-19 by JSC.
