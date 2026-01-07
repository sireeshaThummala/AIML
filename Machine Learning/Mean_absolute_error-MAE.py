
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,root_mean_squared_error

# Mean Absolute Error (MAE)
# Error means the difference between the ACtual value and the predicted value.

def mean_absolute_error(y_true, y_pred):
    """
    doc string : It is a Description about the function '''   '''
    Args : 
     y_true : "Actual Value" , int or flot Data type
     y_pred : "Predicted Value" List(int or float : ) int or float
    
    Return : 
     float : The Mean Absolute Error
    
    Raises Error :
    Absolute : means the whole number or int
    
    """
    return np.mean(np.abs(y_true - y_pred))

# Example data
y_true = np.array([3, -0.5 , 2, 7])
y_pred = np.array([2.5 , 0.0 , 2, 8])

mae = mean_absolute_error(y_true,y_pred)
print("Mearn Absolute Error", mae)

#Plotting the graph
plt.figure(figsize=(10,6))
plt.scatter(y_true , y_pred , color='blue', label='Actual Vs Predicted')
plt.plot([min(y_true), max(y_true)] , [min(y_true , max(y_true))] , color='red', linestyle='---', label='perfect Prediction')
plt.xlable('Actual Values')
plt.ylable('Predicted Values')
plt.title('Actual Vs Predicted')
plt.legend()
plt.show()
