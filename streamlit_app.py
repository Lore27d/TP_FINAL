import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import statsmodels
import pickle
import shelve
from datetime import date, time, datetime


st.title("Predicción de Temperatura")


# Levantamos los modelos
modelos = shelve.open("./modelos_y_data.db")
model_est = modelos["model_est"]
results_ARIMA = modelos["model_arima"]


# Levantamos el df_test con las predicciones de test
with open('./df_test.pkl', 'rb') as f_df_test:
        dataset_test = pickle.load(f_df_test)

# Levantamos el df para la funcion
with open('./df.pkl', 'rb') as f_df:
        df = pickle.load(f_df)


#A = st.date_input(
#    "Fecha de Inicio",
#    date(1909, 1, 1)
#    )
A = date(1909, 1, 1)

B = st.date_input(
    "Seleccione la fecha hasta la cual desea realizar su predicción",
    date(2025, 12, 1)
    )

# Funcion Prediccion
def prediccion_fecha():
    inicio = A.strftime("%Y") + '-' + A.strftime("%m")
    
    fin = B.strftime("%Y") + '-' + B.strftime("%m")



    df_pred = pd.DataFrame()
    df_pred.set_index = df.index
    df_pred['Year'] = df.index.year
    df_pred['Month'] = df.index.month
    #df_pred

    #fecha = datetime.strptime(fecha_pred, '%Y/%m')

    años = []
    año = 2022
    for i in range(0,(B.year-año)+1):
        años.append(año)
        año += 1
    
    for año in años:
        if(año != años[-1]):
            i = 12
        elif(año == años[-1]):
            i= B.month
        for i in range (0,i):
            df_pred.loc[df_pred.shape[0]] = [año,i+1]

    dummies_mes_pred = pd.get_dummies(df_pred["Month"], drop_first=True)
    dummies_pred=pd.DataFrame(dummies_mes_pred)
    dummies_pred=dummies_pred.rename(columns={2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'})

    dummies_pred.index = df_pred.index
    df_pred=pd.merge(df_pred, dummies_pred, left_index=True, right_index=True)

    df_pred['timeIndex']=df_pred.index
    df_pred['Fecha'] = str(df_pred['Year']) + '-' + str(df_pred['Month'])

    for i in range (0, df_pred.shape[0]):
        df_pred['Fecha'][i] = datetime.strptime(str(df_pred['Year'][i]) +'-'+ str(df_pred['Month'][i]), "%Y-%m")   

    pred_reg = model_est.predict(df_pred[['timeIndex','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']])
    pred_arima = results_ARIMA.get_prediction(start= inicio,end=fin)
    pred_arima = pred_arima.predicted_mean


    predicciones = [ pred_reg[x] + pred_arima[x] for x in range(0,len(pred_reg))] 
    prediccion = pd.DataFrame()
    prediccion['Fecha'] = df_pred['Fecha']

    prediccion['Temp'] = predicciones
    return prediccion.sort_values(by='Fecha', ascending=False).head(120)


st.write(prediccion_fecha())

fig = plt.figure(figsize=(20,8))

data = prediccion_fecha()
data.set_index('Fecha', inplace=True )
data['Year'] = data.index.year
data['Month'] = data.index.month


st.write("Prediccion anual")
plot1 = sns.lineplot(x='Year' ,y='Temp', data=data, markers=True, dashes=False)
plot1 = plot1.figure
st.pyplot(plot1)


#plt.yticks([5, 15, 20, 25])
st.write("Prediccion mes a mes")
plot0 = data.plot(kind = "line", y = ['Temp'])
plot0 = plot0.figure
st.pyplot(plot0)








#def RMSE(predicted, actual):
#    mse = (predicted - actual) ** 2
#    rmse = np.sqrt(mse.sum() / mse.count())
#    return rmse

#st.write("Error de ARIMA")
#st.write(RMSE(dataset_test['model_ARIMA'], dataset_test['temp_min']))
#st.write("OLS")
#st.write(RMSE(dataset_test['predict_est'], dataset_test['temp_min']))