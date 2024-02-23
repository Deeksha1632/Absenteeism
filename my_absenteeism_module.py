#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all libraries needed
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Create the special class that we are going to use from here on to predict new data
class absenteeism_model():
        
        def __init__(self, model_file):
            # read the 'model' file which was saved
            with open('model','rb') as model_file:
                self.model = pickle.load(model_file)
                self.data = None
                
        # Take a data file (*.csv) and preprocess it in the same way as done in the training data
        def load_clean_and_scale_data(self, data_file):
            
            df = pd.read_csv(data_file, delimiter = ',')
            self.df_with_predictions = df.copy()
            df = df.drop(['ID'], axis = 1)
            # to preserve the code we have created in the previous sections, we will add a column with 'NaN' strings
            df['Absenteeism Time in Hours'] = 'NaN'
            # create a separate dataframe containing dummy values for ALL available reasons
            reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True)
            
            reason_type_1 = reason_columns.iloc[:,0:14].max(axis=1)
            reason_type_2 = reason_columns.iloc[:,14:17].max(axis=1)
            reason_type_3 = reason_columns.iloc[:,17:20].max(axis=1)
            reason_type_4 = reason_columns.iloc[:,20:].max(axis=1)
            
            df.drop(['Reason for Absence'],axis=1,inplace=True)
            
            df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
            
            # reorder the columns
            column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
            'Daily Work Load Average', 'Body Mass Index', 'Education',
            'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason Type 1', 'Reason Type 2', 'Reason Type 3', 'Reason Type 4']
            
            df.columns = column_names
            
            # Convert the 'Date' column into datetime
            df['Date'] = pd.to_datetime(df['Date'], format = '%d/%m/%Y')
            # Retrieve month and day of week from the 'Date' column
            list_months = []
            for i in range(df.shape[0]):
                list_months.append(df['Date'][i].month)
                
            df['Month Value'] = list_months
            df['Day of Week'] = df['Date'].apply(lambda x: x.weekday())
            
            df = df.drop(['Date'], axis =1)
            
            # reorder the columns
            column_names_updated = ['Reason Type 1', 'Reason Type 2', 'Reason Type 3', 'Reason Type 4','Month Value','Day of Week',
            'Transportation Expense', 'Distance to Work', 'Age',
            'Daily Work Load Average', 'Body Mass Index', 'Education',
            'Children', 'Pets', 'Absenteeism Time in Hours']
            df = df[column_names_updated]
            
            # Map 'Education' variables; the result is a dummy
            df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
            
            # Prepare data to scale
            data_to_scale_1 = df.iloc[:,4:11]
            data_to_scale_2 = df.iloc[:,-3:-1]
            data_to_scale = pd.concat([data_to_scale_1, data_to_scale_2], axis=1)
            # Scale the data
            absenteeism_scaler = StandardScaler()
            scaled_data = absenteeism_scaler.fit_transform(data_to_scale)
            
            column_names = ['Month Value', 'Day of Week', 'Transportation Expense',
            'Distance to Work', 'Age', 'Daily Work Load Average',
            'Body Mass Index', 'Children', 'Pets']
            scaled_inputs = pd.DataFrame(scaled_data, columns = column_names)
            
            # Merge Scaled data with the data with dummies
            dummy_data_reason = df.iloc[:,:4]
            dummy_data_education = df.iloc[:,-4:-3]
            target_data = df.iloc[:,-1:]
            df_pre_final = pd.concat([dummy_data_reason, scaled_inputs,dummy_data_education, target_data], axis =1)
            
            reordered_columns = ['Reason Type 1', 'Reason Type 2', 'Reason Type 3', 'Reason Type 4',
            'Month Value', 'Day of Week', 'Transportation Expense',
            'Distance to Work', 'Age', 'Daily Work Load Average',
            'Body Mass Index','Education', 'Children', 'Pets', 
            'Absenteeism Time in Hours']
            
            df_final = df_pre_final[reordered_columns] 
            df_final = df_final.fillna(value = 0)
            # drop the original absenteeism time and the variables we decided that we don't need
            df_final = df_final.drop(['Absenteeism Time in Hours', 'Day of Week','Daily Work Load Average', 'Distance to Work'], axis = 1)
            
            # To call the 'preprocessed data'
            self.preprocessed_data = df_final.copy()
            # We need the below line so we can use it in the next functions
            self.data = df_final
        
        # A function which outputs the probability of a data point to be 1
        
        def predicted_probability(self):
            if (self.data is not None):
                pred = self.model.predict_proba(self.data)[:,1]
                return pred
            
        # A function which outputs 0 or 1 based on the model
        
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.model.predict(self.data)
                return pred_outputs
            
        # Predict the outputs and the probabilities and
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.model.predict_proba(self.data)[:,1]
                self.preprocessed_data['Prediction'] = self.model.predict(self.data)
                return self.preprocessed_data
            

