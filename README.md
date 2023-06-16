# Adaptive Sharpness and Generalization
Code for CS439 Optimization for Machine Learning (2023), developed by Hadi Hammoud, Léo Nicollier, and Orfeas Liossatos. 
The goal of the project is to study the relationship between adaptive worst-case sharpness and test loss on LeNet-5, FCNN, and GAT.

## Top-Level Directory Structure
Three python notebooks are available. Each notebook trains 50 models from scratch on a dataset and runs Projected Gradient Ascent to estimate sharpness, producing plots comparing sharpness to test loss. The file structure is the following.
+ **/Datasets** contains the Abalone dataset for FCNN
+ **/Plots** contains the results of our experiments
+ **GAT_model.ipynb** is a notebook that runs the experiment on a graph attentional network
+ **LeNet5_model.ipynb** is a notebook that runs the experiment on LeNet-5
+ **FCNN_model.ipynb** is a notebook that runs the experiment on a fully connected neural network

## Requirements (TODO)
- Python 3.9.6
- PyTorch
- Torch Geometric
- SciPy
- NumPy
- Torch-lr-finder
- matplot-lib
- torchvision
- sklearn
- pickle
- shutil
- tqdm
- pandas

## Usage
To run the project locally, install the requirements with pip and execute the ipynb files.
