# Lung Cancer Prediction using Artificial Neural Networks (ANN)

Lung cancer is the primary cause of cancer-related deaths globally. This project aims to predict the risk of lung cancer by categorizing cases into three risk levels: high, medium, and low. We used an Artificial Neural Network (ANN) model and compared its performance with other machine learning and deep learning models. The ANN model achieved a high accuracy of 94.9% using the Adam optimizer, making it an effective tool for early lung cancer prediction.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

Lung cancer is a significant health issue, with smoking and air pollution being major contributing factors. In this study, we used a dataset with 26 attributes to discern lung cancer and applied deep learning techniques for accurate prediction. The ANN model was compared against other machine learning models to ensure the best performance in terms of accuracy.

## Dataset

The dataset used in this project includes patient data with 26 attributes related to lung cancer diagnosis. This data is crucial for training and evaluating the machine learning models used in this study.

- *Attributes:* Age, Gender, Smoking Habits, Family History, Exposure to Air Pollution, etc.
- *Number of Records:* [Number of records in your dataset]
- *Source:* [Mention the source of the dataset if available]

## Installation

To get started with this project, follow these steps:

1. Clone the repository:
    bash
    git clone https://github.com/vatsalmehta163/Lung-cancer-prediction-using-ANN.git
    cd Lung-cancer-prediction-using-ANN
    

2. Install the required dependencies:
    bash
    pip install -r requirements.txt
    

3. Launch the Jupyter Notebook:
    bash
    jupyter notebook DLproject.ipynb
    

## Usage

To use the model for predicting lung cancer risk, you can either run the provided Jupyter Notebook DLproject.ipynb or integrate the model into your own project. The notebook contains all the necessary code for data preprocessing, model training, and evaluation.

## Modeling Approach

### Data Preprocessing

The data preprocessing steps include:

- Handling missing values
- Normalization of features
- Splitting the dataset into training and test sets

### Model Architecture

The ANN model used in this project has the following architecture:

- *Input Layer:* [Number of input neurons]
- *Hidden Layers:* [Number of hidden layers and neurons]
- *Activation Functions:* ReLU for hidden layers, Softmax for the output layer
- *Optimizer:* Adam
- *Loss Function:* Categorical Crossentropy

### Training and Evaluation

The model was trained on the training set and evaluated on the test set using various metrics such as accuracy, precision, recall, and F1-score.

## Results

The ANN model achieved an accuracy of 94.9% on the test set, outperforming other machine learning models such as SVM, Random Forest, and Logistic Regression.

- *Accuracy:* 94.9%
- *Precision:* [Precision value]
- *Recall:* [Recall value]
- *F1-score:* [F1-score value]

These results demonstrate the effectiveness of the ANN model in predicting lung cancer risk.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Mention any individuals, organizations, or resources that contributed to your project]
- Special thanks to [Name of any collaborators or mentors] for their guidance and support.
