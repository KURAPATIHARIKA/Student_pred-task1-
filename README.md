# Student_pred-task1-
all abt the Student score predection task....where the percentile of the studenets has to be predicted using ml with python

import pandas as pd
import matplotlib.pyplot as plt
# loading the dataset
url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df=pd.read_csv(url)
# printing first 5 rows
print(df.head())
# info abt the dataset
print(df.info())
# looking for null values
print(df.isnull().sum())
# data cleaning
df.drop_duplicates(inplace=True)
# plotting graph
df.plot.scatter(x='Hours', y='Scores')
plt.show()
# saving changes to another file
df.to_csv('grip-task1.csv', index=False)
# creating linear regressn model
model=LinearRegression()

# fitting the data to model
X = df[['Hours']]  # Input feature (independent variable)
y = df['Scores']   # Target variable (dependent variable)
model.fit(X, y)

# Making predictions
new_data = pd.DataFrame({'Hours': [9.25]})  # Create a DataFrame with feature names
predictions = model.predict(new_data)
print("The student who spend 9.25 hrs for study may get" ,predictions)

