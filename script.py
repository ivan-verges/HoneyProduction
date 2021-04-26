import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import wget
from sklearn import linear_model

#Defines a Base Path for the DataSet File
file_path = os.path.abspath(os.path.dirname(__file__))

#Completes the File Path by adding "\\Data\\HoneyProduction.csv" for Windows, or "/Data/HoneyProduction.csv" for UNIX Like OS (Linux, Mac, etc)
if os.name == "nt":
    file_path += "\\Data\\HoneyProduction.csv"
else:
    file_path += "/Data/HoneyProduction.csv"

#If the File Path does Not Exists, Creates the Base Folder and Downloads the DataSet from Internet
if not os.path.exists(file_path):
    #Creates the Base Folder to Store the DataSet
    os.mkdir(os.path.abspath(os.path.dirname(file_path)))

    #DataSet URL:
    url = "https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv"

    #Downloads the DataSet to the Defined Path
    wget.download(url, file_path)

#Loads the Data from the CSV File into a Dataframe using Pandas
df = pd.read_csv(file_path)

#Groups the DataFrame Data by Year and the Mean of the other columns
prod_per_year = df.groupby(["year"]).mean().reset_index()

#Creates the Features (Year) from data and formats it to be in the form of array ([[0, 1, 2]])
X = prod_per_year["year"]
X = X.values.reshape(-1, 1)

#Creates the Labels (Total Production) for the Features and formats it to be in the form of array ([[0, 1, 2]])
y = prod_per_year["totalprod"]
y = y.values.reshape(-1, 1)

#Generates a years time-lapse to predict and formats it to be in the form of array ([[0, 1, 2]])
X_future = np.array(range(2013, 2026))
X_future = X_future.reshape(-1, 1)

#Creates a Linear Regression model to train
regr = linear_model.LinearRegression()

#Trains the Model using the Linear Regression class
regr.fit(X, y)

#Predicts the production for each year in X
y_predict = regr.predict(X)

#Predicts the production for each year in our time-lapse series
future_predict = regr.predict(X_future)

#Prints the Coeficient (Weight) of our model
print("Model Weight: " + str(regr.coef_))

#Prints the Intercept (Bias) of our model
print("Model Bias: " + str(regr.intercept_))

#Avoids the usage of Scientific Notation in the Graphs
plt.ticklabel_format(style='plain')

#Sets Axis Labels
plt.xlabel("Year")
plt.ylabel("Honey Production")

#Set the graph to plot our data and our future predictions
plt.scatter(df["year"], df["totalprod"], color='skyblue', linewidth=4, label="Productions by Year")
plt.scatter(X, y, color='red', linewidth=6, label="Avg. Productions by Year")
plt.plot(X, y_predict, color='blue', linewidth=4, label="Actual Predictions")
plt.plot(X_future, future_predict, color='orange', linewidth=4, linestyle='dashed', label="Future Predictions")

#Show Legends
plt.legend()

#Show the plot graph
plt.show()