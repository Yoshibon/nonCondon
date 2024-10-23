# Modeling Infrared Spectroscopy of Nucleic Acids

This repository contains a `PyTorch` implementation of the model proposed in the paper:   **"Modeling Infrared Spectroscopy of Nucleic Acids: Integrating Vibrational Non-Condon Effects with Machine Learning Schemes."** 

## Overview 

Infrared spectroscopy is a key technique for studying the structural dynamics of nucleic acids. Our model incorporates vibrational non-Condon effects into machine learning frameworks to enhance the accuracy of spectral predictions. 

## Project Structure 

- `./nonCondon/`: Contains the core components of the model, including functions, models, and utilities. 
  - `./saved_model/`: Contains the best model we reported in the paper
  -  `./standardizer/`: Contains the standardizer we used in developing model. The new data **MUST** be processed by these standardizers for doing preditions.
  - `Make_prediction.py`: Defines the function for making predictions using the trained model.  
  - `Models.py`: Contains the implementation of the deep learning model used for spectroscopy.  
  - `utils.py`: Includes utility functions and the definition of the custom loss function used in training.
  
- `./demo_data/`: Contains demo datasets for illustrating how to apply our model
- `./predictions/`: Contains the predicted magnitudes of demo datasets
- `demo_example.ipynb`: A simple example of how to use our model making predictions

## Data Availability

The datasets used for training are not included in this repository due to size constraints. 

## Further Information

For more details on the implementation and theoretical background, please refer to our paper.

## Contact

If you have any questions or need further assistance, feel free to contact us via email.
