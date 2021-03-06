from fastapi import FastAPI
import pickle
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import json



class client_data(BaseModel):
    data: dict

file_name='Imputer'
open_file = open(file_name, "rb")
imputer_loaded = pickle.load(open_file)
open_file.close()

file_name='Poly'
open_file = open(file_name, "rb")
poly_transformer_loaded = pickle.load(open_file)
open_file.close()

file_name='OneHotEncoder'
open_file = open(file_name, "rb")
OHE_loaded = pickle.load(open_file)
open_file.close()

file_name='Scaler'
open_file = open(file_name, "rb")
scaler_loaded = pickle.load(open_file)
open_file.close()

file_name='Model'
open_file = open(file_name, "rb")
model_loaded = pickle.load(open_file)
open_file.close()

file_name='NumCol'
open_file = open(file_name, "rb")
num_col_loaded = pickle.load(open_file)
open_file.close()
        
file_name='dtype'
open_file = open(file_name, "rb")
dtype_loaded = pickle.load(open_file)
open_file.close()   
        
app = FastAPI()


@app.post("/predict")
async def predict(client_to_predict: client_data):
    
    data=pd.DataFrame(client_to_predict.data, index=[0])
    
    data=data.astype(dtype_loaded.to_dict() )
    
    data[num_col_loaded]=imputer_loaded.transform(data[num_col_loaded])
    
    data['CREDIT_INCOME_PERCENT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
    data['ANNUITY_INCOME_PERCENT'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data['CREDIT_TERM'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
    data['DAYS_EMPLOYED_PERCENT'] = data['DAYS_EMPLOYED'] / data['DAYS_BIRTH']
    
    poly_features = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
    poly_features = poly_transformer_loaded.transform(poly_features)
    poly_features = pd.DataFrame(poly_features, 
                        columns = poly_transformer_loaded.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 
                                                                           'EXT_SOURCE_3', 'DAYS_BIRTH']))
    poly_features['SK_ID_CURR'] = data['SK_ID_CURR']
    data = data.merge(poly_features.drop(columns=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']), on = 'SK_ID_CURR', how = 'left')
    
    data_object = data.select_dtypes('object')
    codes = OHE_loaded.transform(data_object).toarray()
    feature_names = OHE_loaded.get_feature_names(data.select_dtypes('object').columns)
    data = pd.concat([data.select_dtypes(exclude='object'), 
               pd.DataFrame(codes,columns=feature_names).astype(int)], axis=1)


    data=scaler_loaded.transform(data)
        
    return json.dumps(model_loaded.predict_proba(data)[:,1].tolist())
