# import packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#load data
bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")

#Train Data
bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']],bmi_life_data[['Life expectancy']])

#Predict Life Expectancy
laos_life_exp = bmi_life_model.predict(21.07931)
print("Life Expectancy of individual with BMI "+str(21.07931) + " is " '{}').format(laos_life_exp)

