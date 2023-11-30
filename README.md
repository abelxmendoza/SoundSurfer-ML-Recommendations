# SoundSurfer-ML-Recommendations

---

![1699520963119](image/README/1699520963119.png)

## Overview

This project, developed as part of the Computer Science class CPSC 483 - Introduction to Machine Learning with Professor Yu Bai at California State University, Fullerton, focuses on predicting danceability in music using machine learning techniques. The dataset used for this project contains various features related to songs, and a RandomForestRegressor model is employed for predicting danceability.

## Authors

* Abel Mendoza ([@abelxmendoza](https://github.com/abelxmendoza))
* Akshat ([@jane-smith](https://github.com/jane-smith))
* Meng Yang ([@robert-johnson](https://github.com/robert-johnson))

## About

This project leverages machine learning concepts, specifically the RandomForestRegressor model, to predict the danceability of songs based on various features. The RandomForestRegressor is an ensemble learning method that builds a collection of decision trees and outputs the average prediction of the individual trees. The project demonstrates the process of loading and preprocessing data, splitting it into training and testing sets, training a model, evaluating its performance, and using it to make predictions for practical applications, such as generating music playlists.

## Getting Started

### Prerequisites

- Python
- Jupyter Notebook or Google Colab
- Libraries: NumPy, pandas, scikit-learn

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/music-prediction-project.git
   cd music-prediction-project
   ```
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the Jupyter Notebook or Google Colab containing the project code.

## Project Structure

* `data.csv`: Dataset containing song features.
* `ML_final_Project.ipynb`: Jupyter Notebook or Colab file with the project code.
* `README.md`: Project documentation.

## Code Explanation

The project code consists of the following key steps:

1. **Data Loading and Preprocessing** : Mounting Google Drive, reading the dataset, and cleaning unnecessary columns.
2. **Train-Test Split** : Splitting the data into training and testing sets.
3. **Model Training** : Using a RandomForestRegressor to train the model.
4. **Model Evaluation** : Calculating Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) for model performance.
5. **Playlist Generation** : Creating a playlist by predicting danceability for all songs and selecting the top 50.
6. **Display Playlist** : Printing the generated playlist.

## Random Forest Regressor

The Random Forest Regressor is an ensemble learning algorithm that operates by constructing a multitude of decision trees at training time and outputs the average prediction of the individual trees for regression tasks. This ensemble approach imparts robustness and improves generalization performance compared to individual trees.

### Key Features:

1. **Decision Trees** : The basic building blocks of a Random Forest are decision trees. Each tree is constructed by recursively splitting the data based on features to form leaf nodes that contain the final predictions.
2. **Random Subspace Sampling** : At each split in a decision tree, a random subset of features is considered, introducing diversity among the trees and reducing overfitting.
3. **Bootstrapped Sampling** : Each tree is trained on a bootstrapped sample, meaning that a random subset of the training data is sampled with replacement. This process introduces variability in the training sets for different trees.
4. **Voting/Averaging** : For regression tasks, the final prediction is the average of predictions from all individual trees. This aggregation helps to smooth out individual tree predictions and produce a more stable and accurate result.

### Advantages:

* **Robustness** : Random Forests are less prone to overfitting due to the combination of multiple trees and random feature selection.
* **Versatility** : Suitable for both classification and regression tasks.
* **Feature Importance** : Provides a measure of feature importance, aiding in feature selection.

## Usage in This Project

In this music prediction project, the Random Forest Regressor is employed to predict danceability based on various features of songs. The model is trained on a subset of the data and evaluated for its performance using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2).

## Usage

1. Open the Jupyter Notebook or Colab file.
2. Run each cell sequentially to execute the code.
3. Review the model evaluation metrics and the generated playlist.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your proposed changes.

## License

This project is licensed under the [MIT License](https://chat.openai.com/c/LICENSE).

## Acknowledgments

* The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/).
* This project is part of the coursework for CPSC 483 - Introduction to Machine Learning at California State University, Fullerton, under the guidance of Professor Yu Bai.

Special thanks to Professor Yu Bai for his guidance and support throughout the course.
