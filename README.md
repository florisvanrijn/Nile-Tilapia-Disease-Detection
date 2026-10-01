# Nile tilapia disease detection with deep learning

My MSc Computer Science thesis at the University of Bath (2023). It is also where my work on
tilapia started: [TilapAI](https://tilap.ai) and its fish-weighing system, Niloscale, grew out of it.

## The problem

Streptococcosis and Tilapia Lake Virus (TiLV) kill farmed Nile tilapia and cost the industry tens
to hundreds of millions of dollars a year in lost production. Spotting them is still a manual job.
Farms are also reluctant to admit to disease, so photos of sick fish are hard to come by and there
was no dataset to train a model on.

## What I did

- Modelled Nile tilapia in 3D, printed them and hand-painted them with the visible symptoms of both
  diseases.
- Took the models to Gårdsfisk, a Nile tilapia farm in Sweden, and moved them among the live fish
  on a clear plastic rod while an iPhone in a waterproof case shot timelapses inside the tanks.
  That produced about 11,000 images of healthy and sick red-strain Nile tilapia.
- Trained a CNN from scratch in Keras and fine-tuned a ResNet-18 in PyTorch, with data augmentation
  (brightness, cut-out, zoom, saturation, flips, Gaussian noise) and a hyperparameter search, at
  1120 x 630 px on a V100.

## Results

| Model | Test accuracy | Weights file |
| --- | --- | --- |
| Self-trained CNN, ID 13 | 99.67% | `Self-trained models/ID_13_Self_Trained_CNN_Weights.h5` |
| ResNet-18 fine-tuned, ID 2 | 96.74% | `ResNet-18 models/ID_2_ResNet18_CNN_Weights.pth` |

Scores are on held-out images from the collected dataset, where the sick fish are painted models.
Model IDs match Tables 9 and 10 in the thesis.

## What's in this repository

- `notebook.ipynb`: all the code with commentary,
  written for Google Colab.
- `thesis.md`: the thesis, readable here on GitHub (figures in `thesis_files/`).
- `thesis.pdf`: the thesis as submitted, with the code appendix.

The large files are in the
[v1.0 release](https://github.com/florisvanrijn/Nile-Tilapia-Disease-Detection/releases/tag/v1.0):

- `Trained_Model_Weights.zip` (800 MB): all 21 weight files. Self-trained CNNs in Keras `.h5`
  (IDs 1 to 14, plus two shorter-training variants) and ResNet-18s in PyTorch `.pth`
  (IDs 1, 2, 5, 6 and 7).
- `Nile_Tilapia_3D_models_STL.zip` (500 MB): printable adult tilapia, dorsal fin raised and lowered.
- `Nile_Tilapia_3D_models_FBX.zip` (990 MB): the same adults plus a juvenile, for rendering or
  editing.
- `Tilapia_Deep_Learning_Presentation.mp4`: a short presentation of the project.

## Dataset

About 11,000 images (3 GB) in two classes, `Healthy_Fish` and `Sick_Fish`, on
[Kaggle](https://kaggle.com/datasets/a7372f357af9e3faaaf8df6b83e94cd0664a081fe912e782c217a100e03d7a95).

## Running the notebook

1. Download `Healthy_Fish.zip` and `Sick_Fish.zip` from the Kaggle dataset.
2. Put them in your Google Drive under `MyDrive/Tilapia data/`.
3. Open the notebook in Google Colab. Training needs a V100 or better with high RAM. To evaluate
   only, load the weights from the release.

## Citation

van Rijn, F. (2023). *Nile Tilapia Disease Detection Using Deep Learning*. MSc thesis, University
of Bath.

## License

The code, model weights and 3D models are MIT licensed (see `LICENSE`): anyone can use, change and
share them, commercially too, as long as they keep the copyright notice. The thesis PDF and the
presentation stay under my copyright because they include figures from other papers; cite them as
above.
