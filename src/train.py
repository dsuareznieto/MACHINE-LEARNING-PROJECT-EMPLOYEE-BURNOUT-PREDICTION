import pandas as pd
import numpy as np
from sklearn import model_selection
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from utils.functions import label_encoder
import pickle
import warnings
warnings.filterwarnings("ignore")
import os 

os.chdir(os.path.dirname(__file__))

#Loading the dataset
df = pd.read_csv('data/raw/dataset_burnout.csv')

#Drop columns
data= df.drop(['Date_of_termination', 'Unnamed: 32', 'Date_of_Hire', 'StockOptionLevel'], axis=1)

#Rename columns
data.rename(columns={'Attrition':'Burnout', 'Leaves':'Permitted_Leaves_Taken'}, inplace=True)

#Mapping and Label Encoding
col         = 'BusinessTravel'
conditions  = [ data[col] == 'Non-Travel', data[col] == 'Travel_Rarely', data[col] == 'Travel_Frequently']
choices     = [ 0, 1, 2 ]
    
data['BusinessTravel'] = np.select(conditions, choices)
data['BusinessTravel'] = data['BusinessTravel']

col         = 'Higher_Education'
conditions  = [ data[col] == '12th', data[col] == 'Graduation', data[col] == 'Post-Graduation', data[col] == 'PHD' ]
choices     = [ 0, 1, 2, 3 ]
    
data['Higher_Education'] = np.select(conditions, choices)
data['Higher_Education'] = data['Higher_Education']

label_encoder(data=data,column='Burnout')
label_encoder(data=data,column='Department')
label_encoder(data=data,column='Gender')
label_encoder(data=data,column='JobRole')
label_encoder(data=data,column='MaritalStatus')
label_encoder(data=data,column='OverTime')
label_encoder(data=data,column='Status_of_leaving')
label_encoder(data=data,column='Mode_of_work')
label_encoder(data=data,column='Work_accident')
label_encoder(data=data,column='Source_of_Hire')
label_encoder(data=data,column='Job_mode')

#Loading another dataset
df_ibm = pd.read_csv('data/raw/IBM-HR-Employee-Attrition.csv')
data['EnvironmentSatisfaction'] = df_ibm['EnvironmentSatisfaction']

#Loading another dataset
df_hr = pd.read_csv('data/raw/HR.csv')
data['average_montly_hours'] = df_hr['average_montly_hours']

#Feature transformation
data['promedio'] = (data['TotalWorkingYears'] + data['Age'])/2

#Generating processed dataset
data.to_csv('data/processed/data_processed.csv')

X = data[['OverTime', 'MaritalStatus',  'DistanceFromHome', 'JobRole', 'JobLevel','EnvironmentSatisfaction', 'Work_accident', 'promedio', 'Permitted_Leaves_Taken']]
y = data['Burnout']

#Doing split
validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,
                                                                    y,
                                                                    test_size=validation_size,
                                                                    random_state=seed)

#Doing OverSampling
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train, Y_train)

#Defining the model 
estimator = DecisionTreeClassifier(max_depth=1)

ada_clf = AdaBoostClassifier(base_estimator=estimator,
                            n_estimators=50,
                            random_state=42)

#Training the model 
ada_clf.fit(X_train_ros, y_train_ros)

#Saving the model 

with open('model\selected_model\AdaBoost', "wb") as archivo_salida:
    pickle.dump(ada_clf, archivo_salida)

#New predictions 
X_new = pd.DataFrame({'OverTime': [1], 
                     'MaritalStatus': [0],
                     'DistanceFromHome': [25], 
                     'JobRole': [5], 
                     'JobLevel': [3],
                     'EnvironmentSatisfaction':[1],
                     'Work_accident':[1],
                     'promedio': [49.0],
                     'Permitted_Leaves_Taken':[5]})

ada_predictions = ada_clf.predict(X_new)
print("Predicción", ada_predictions)



