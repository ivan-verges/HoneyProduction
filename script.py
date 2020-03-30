import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

#Reads data into a Dataframe
df = pd.read_csv("https://s3.amazonaws.com/codecademy-content/programs/data-science-path/linear_regression/honeyproduction.csv")

#Group dataframe data about mean production by year
prod_per_year = df.groupby(["totalprod"]).mean().reset_index()

#Creates the Features from data and format it
X = prod_per_year["year"]
X = X.values.reshape(-1, 1)

#Creates the Labels for the Features
y = prod_per_year["totalprod"]

#Generates a years time-lapse to predict
X_future = np.array(range(2013, 2050))
X_future = X_future.reshape(-1, 1)

#Creates a Linear Regression model to train
regr = linear_model.LinearRegression()

#Trains the Model
regr.fit(X, y)

#Predicts the production for each year in X
y_predict  = regr.predict(X)

#Predicts the production for each year in our time-lapse series
future_predict  = regr.predict(X_future)

#Prints the Coeficient (Weight) of our model
print(regr.coef_)

#Prints the Intercept (Bias) of our model
print(regr.intercept_)

#Set the graph to plot our data and our future predictions
plt.scatter(X, y)
plt.plot(X, y_predict)
plt.plot(X_future, future_predict)

#Show the plot graph
plt.show()