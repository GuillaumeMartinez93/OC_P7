import pickle
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import json
import streamlit.components.v1 as components
import dvc.api

st.set_page_config(layout="wide")

def load_data():

	file_name='Model'
    open_file = open(file_name, "rb")
    model_loaded = pickle.load(open_file)
    open_file.close()

	file_name='DataSet'
    open_file = open(file_name, "rb")
    df_test = pickle.load(open_file)
    open_file.close()

	explainer = shap.TreeExplainer(model_loaded)
	shap_values = explainer.shap_values(df_test)[1]
	exp_value=explainer.expected_value[1]

return df_test,shap_values,exp_value

def score_viz(df_test,client,exp_value,shap_values):

	st.title('Dashboard Pret à dépenser')
	st.subheader('Visualisation score')

    json_object = json.dumps(client,indent=4) 
    url = "https://apphomecredit.herokuapp.com/predict"
    headers = {"Content-Type":"application/json"}
    x = requests.post(url, headers=headers ,data = json_object )
	score=float(x.text.split("[")[1].split("]")[0])
    if score> :
        result='Approved'
    else :
        result='Rejected'

	fig = go.Figure(go.Indicator(
	mode = "gauge+number+delta",
    value = score,
    number = {'font':{'size':48}},
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': result.tolist(), 'font': {'size': 28, 'color':color(result)}},
    delta = {'reference': 0.48, 'increasing': {'color': "red"},'decreasing':{'color':'green'}},
    gauge = {
        'axis': {'range': [0,1], 'tickcolor': color(result)},
        'bar': {'color': color(result)},
        'steps': [
            {'range': [0,0.48], 'color': 'lightgreen'},
            {'range': [0.48,1], 'color': 'lightcoral'}],
        'threshold': {
            'line': {'color': "black", 'width': 5},
            'thickness': 1,
            'value': 0.48}}))


	st.plotly_chart(fig)

	st_shap(shap.force_plot(exp_value, shap_values[client], features = df_test.iloc[client], feature_names=df_test.columns, figsize=(12,5)))

def color(pred):
	if pred=='Approved':
		col='Green'
	else :
		col='Red'
	return col

def st_shap(plot, height=None):
	shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
	components.html(shap_html, height=height)


def tab_client(db_test):

	st.title('Dashboard Pret à dépenser')
	st.subheader('Tableau clientèle')
	row0_1,row0_spacer2,row0_2,row0_spacer3,row0_3,row0_spacer4,row_spacer5 = st.columns([1,.1,1,.1,1,.1,4])

	sex=row0_1.selectbox("Sexe",['All']+db_test['CODE_GENDER'].unique().tolist())
	age=row0_1.selectbox("Age",['All']+(np.sort(db_test['YEARS_BIRTH'].unique()).astype(str).tolist()))
	fam=row0_2.selectbox("Statut familial",['All']+db_test['NAME_FAMILY_STATUS'].unique().tolist())
	child=row0_2.selectbox("Enfants",['All']+(np.sort(db_test['CNT_CHILDREN'].unique()).astype(str).tolist()))
	pro=row0_3.selectbox("Statut pro.",['All']+db_test['NAME_INCOME_TYPE'].unique().tolist())
	stud=row0_3.selectbox("Niveau d'études",['All']+db_test['NAME_EDUCATION_TYPE'].unique().tolist())

	db_display=db_test[['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
	'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
	'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]
	db_display['YEARS_BIRTH']=db_display['YEARS_BIRTH'].astype(str)
	db_display['CNT_CHILDREN']=db_display['CNT_CHILDREN'].astype(str)
	db_display['AMT_INCOME_TOTAL']=db_test['AMT_INCOME_TOTAL'].apply(lambda x: int(x))
	db_display['AMT_CREDIT']=db_test['AMT_CREDIT'].apply(lambda x: int(x))
	db_display['AMT_ANNUITY']=db_test['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))

	db_display=filter(db_display,'CODE_GENDER',sex)
	db_display=filter(db_display,'YEARS_BIRTH',age)
	db_display=filter(db_display,'NAME_FAMILY_STATUS',fam)
	db_display=filter(db_display,'CNT_CHILDREN',child)
	db_display=filter(db_display,'NAME_INCOME_TYPE',pro)
	db_display=filter(db_display,'NAME_EDUCATION_TYPE',stud)

	st.dataframe(db_display)
	st.markdown("**Total clients correspondants: **"+str(len(db_display)))

def get_client(db_test):
	client=st.sidebar.selectbox('Client',db_test['SK_ID_CURR'])
	idx_client=db_test.index[db_test['SK_ID_CURR']==client][0]
	return client,idx_client

def infos_client(db_test,client,idx_client):
	"""Affichage des infos du client sélectionné dans la barre latérale"""
	st.sidebar.markdown("**ID client: **"+str(client))
	st.sidebar.markdown("**Sexe: **"+db_test.loc[idx_client,'CODE_GENDER'])
	st.sidebar.markdown("**Statut familial: **"+db_test.loc[idx_client,'NAME_FAMILY_STATUS'])
	st.sidebar.markdown("**Enfants: **"+str(db_test.loc[idx_client,'CNT_CHILDREN']))
	st.sidebar.markdown("**Age: **"+str(db_test.loc[idx_client,'YEARS_BIRTH']))	
	st.sidebar.markdown("**Statut pro.: **"+db_test.loc[idx_client,'NAME_INCOME_TYPE'])
	st.sidebar.markdown("**Niveau d'études: **"+db_test.loc[idx_client,'NAME_EDUCATION_TYPE'])

def main():
    df_test,shap_values,exp_value=load_data()
    PAGES = [
	    "Tableau clientèle",
	    "Visualisation score",
	    "Comparaison clientèle"
	]
    st.sidebar.write('')
	st.sidebar.write('')

    st.sidebar.title('Pages')
	selection = st.sidebar.radio("Go to", PAGES)

	if selection=="Tableau clientèle":
		tab_client()
	if selection=="Visualisation score":
		client,idx_client=get_client(db_test)
		infos_client(db_test,client,idx_client)
		score_viz(lgbm,df_test,idx_client,exp_value,shap_values)
	if selection=="Comparaison clientèle":
        pass

if __name__ == '__main__':
	main()
