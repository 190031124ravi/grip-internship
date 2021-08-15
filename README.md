# grip-internship
codes:
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline


url = "http://bit.ly/w-data"
data = pd.read_csv(url)


data.head()


data.plot(x='Scores', y='Hours', marker='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Percentage Score')  
plt.ylabel('Hours Studied')  
plt.show()


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


line = regressor.coef_*X+regressor.intercept_


plt.scatter(X, y)
plt.plot(X, line);
plt.show()



print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 

hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))



from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 

