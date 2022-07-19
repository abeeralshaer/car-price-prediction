import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns


save_model_directry = r'./models/'

# Function to manage user input data
def preprocess_input(inp, final_dummy):
    index_dict = dict(zip(final_dummy.iloc[:,4:].columns,range(final_dummy.shape[1])))
    new_vector = np.zeros(105)
    try:
        new_vector[index_dict['make_'+inp.make]] = 1
    except:
        pass
    try:
        new_vector[index_dict['vehicle_type_'+inp.vehicle_type]] = 1
    except:
        pass
    try:
        new_vector[index_dict['drivetrain_'+inp.drivetrain]] = 1
    except:
        pass
    try:
        new_vector[index_dict['transmission_'+inp.transmission]] = 1
    except:
        pass
    try:
        new_vector[index_dict['fuel_type_'+inp.fuel_type]] = 1
    except:
        pass
    try:
        new_vector[index_dict['engine_block_'+engine_block]] = 1
    except:
        pass
    arr = np.array([inp.miles, inp.year, inp.engine_size])
    res = np.concatenate((arr, new_vector))
    res = pd.DataFrame(res.reshape(1,-1))
    res.columns = final_dummy.iloc[:,1:].columns
    return(res)



###################################
###################################
###########
# load data

usa = pd.read_csv(r'./dataset/us-dealers-used.csv')

canada = pd.read_csv(r'./dataset/ca-dealers-used.csv')

print("usa raw dataset's shape: ",usa.shape)
print("Canada raw dataset's shape: ",canada.shape)
print("#################################### \n")
###################################
###################################
# Remove unnecessary features

usa = usa.drop(usa.loc[:,['id', 'vin', 'stock_no', 'model', 'trim', 'body_type', 'seller_name', 'street', 
    'city', 'state', 'zip']].columns, axis=1)
canada = canada.drop(canada.loc[:,['id', 'vin', 'stock_no', 'model', 'trim', 'body_type', 'seller_name', 'street', 
    'city', 'state', 'zip']].columns, axis=1)
print("usa dataset's shape after Removing unnecessary features: ",usa.shape)
print("Canada dataset's shape after Removing unnecessary features: ",canada.shape)
print("#################################### \n")
##############################
##############################
# Remove rows with NA values for the target feature
usa=usa.dropna(subset=['price'])
canada=canada.dropna(subset=['price'])
print("usa dataset's shape after Removing Na rows: ",usa.shape)
print("Canada dataset's shape after Removing Na rows: ",canada.shape)
print("#################################### \n")
##############################
##############################
# Merge in one big dataset
final = pd.concat([usa, canada])
print("merged dataset shape: ",final.shape)

##############################
##############################
#Clean the dataset from all rows with at least one NA
final = final.dropna(axis=0)
print("final dataset's shape: ",final.shape)
print("#################################### \n")

####
# preprocessing(Dummization of Categorical features & spliting the dataset)

final_dummy = pd.get_dummies(final)
y = final_dummy.price
x = final_dummy.drop('price',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

# creating the model (regresion tree).
model = LinearRegression()
model.fit(x_train, y_train)
# prediction
y_pred=model.predict(x_test)
print(y_pred)
# evaluation 
Accuracy=r2_score(y_test,y_pred)*100
print(" Accuracy of the model is %.2f" %Accuracy)

##############################################################################
# Code to check if it works (Spoiler: it seems so)
inp = final.iloc[0,1:]
inp_dummy = preprocess_input(inp, final_dummy)
print(model.predict(inp_dummy))


##########################
# SAVE-LOAD using pickle #
##########################
import pickle

# save

with open(save_model_directry+'model.pkl','wb') as f:
    pickle.dump(model,f)