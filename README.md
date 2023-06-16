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

## Requirements
- Python 3.9.6
- torch 2.0.1
- torch-geometric 2.3.1
- scipy 1.10.1
- numpy 1.22.4
- torch-lr-finder 0.2.1
- matplotlib 3.7.1
- torchvision 0.15.2
- scikit-learn 1.2.2
- tqdm 4.65.0
- pandas 1.5.3

## Usage
To run the project locally, install the requirements with pip and execute the ipynb files.
