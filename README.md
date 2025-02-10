
Predicting Housing Prices Using Artificial Neural Networks

This project predicts housing prices based on various features using an Artificial Neural Network (ANN) model built with TensorFlow and Keras. The dataset is split into training and testing sets, and the model is trained and evaluated based on Mean Absolute Error (MAE) and loss values.

Project Overview
- Dataset: The dataset consists of housing data with various features, such as the date of the listing, location, size, and other details that can influence the price of a house. 
- Goal: To predict the price of a house using a neural network model.
- Tech Stack:
  - Python
  - Pandas (for data manipulation)
  - Matplotlib (for plotting)
  - Scikit-learn (for data preprocessing)
  - TensorFlow/Keras (for building and training the ANN)

Getting Started
To run this project, ensure you have the following dependencies installed:

pip install pandas matplotlib scikit-learn tensorflow

Files in this Project
- df_train.csv: The training dataset, containing features and the target variable (price).
- df_test.csv: The testing dataset, containing features used to evaluate the model.
- predictions.csv: The output file with the predicted house prices for the test set.
- assignment2.py: Python script that loads the data, preprocesses it, builds the neural network, and saves predictions.

Code Explanation

1. Data Preprocessing
The dataset is first loaded using Pandas, and the target variable (price) is separated from the feature set. The date column is processed as a numerical feature by converting it into a timestamp, and all input features are normalized using StandardScaler from scikit-learn.

2. Neural Network Model
A Sequential model is built using Keras, consisting of:
- Two hidden layers with 16 neurons and ReLU activation.
- One output layer with a single neuron and a linear activation function for regression.

The model is compiled using the Adam optimizer and Mean Squared Error (MSE) loss function.

3. Model Training and Evaluation
The model is trained for 30 epochs, with validation on the test set. The Mean Absolute Error (MAE) and loss are tracked during training, and the final model is evaluated on the test data.

4. Predictions and Results
Once the model is trained, predictions are made on the test dataset, and the predictions are denormalized. These predictions are added to the df_test dataframe and saved as predictions.csv.

5. Plotting
The code includes plots showing:
- Mean Absolute Error (MAE) over the epochs for both training and validation sets.
- Loss over the epochs for both training and validation sets.

How to Run
1. Make sure that your df_train.csv and df_test.csv files are in the same directory as assignment2.py.
2. Run the script:

python assignment2.py

The script will:
- Train the model
- Output the validation MAE and Loss
- Save the predicted house prices to predictions.csv
- Display plots for MAE and Loss

Example Output
After running the script, you will see the following printed output:

Validation Loss: <value>, Validation MAE: <value>
Predictions saved to 'predictions.csv'

Additionally, the MAE and Loss plots will be displayed.

