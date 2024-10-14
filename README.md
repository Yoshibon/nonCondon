# Modeling Infrared Spectroscopy of Nucleic Acids

This repository contains a `PyTorch` implementation of the model proposed in the paper:   **"Modeling Infrared Spectroscopy of Nucleic Acids: Integrating Vibrational Non-Condon Effects with Machine Learning Schemes."** 

## Overview 

Infrared spectroscopy is a key technique for studying the structural dynamics of nucleic acids. Our model incorporates vibrational non-Condon effects into machine learning frameworks to enhance the accuracy of spectral predictions. 

## Project Structure 

- `./nonCondon/`: Contains the core components of the model, including functions, models, and utilities. 
  - `Make_prediction.py`: Defines the function for making predictions using the trained model.  
  - `Models.py`: Contains the implementation of the machine learning model used for spectroscopy.  
  - `simply_A.py`: Calculates the spectroscopy of G5C5 using the provided magnitudes. 
  - `simply_B.py`: Calculates the spectroscopy of GC8 using the provided magnitudes.
  - `utils.py`: Includes utility functions and the definition of the custom loss function used in training.
- `./picked_model/`: Contains the final trained version of the model, which was selected for evaluation and analysis. 
- `train.py`: The script that handles the training procedure for the model.

## Data Availability

The datasets used for training are not included in this repository due to size constraints. To access the datasets, please visit [this link](#).

## Further Information

For more details on the implementation and theoretical background, please refer to our paper.

## Contact

If you have any questions or need further assistance, feel free to contact us via email.
