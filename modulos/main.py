import pandas as pd
from procesamiento import procesamiento
from seleccion_modelos import modelos


df = pd.read_csv('C:/Users/franc/Documents/Repositorios/TT_DS_proyecto_7/data/users_behavior.csv')


lista_data_procesada = procesamiento(df)

features_train = lista_data_procesada[0] 
features_test  = lista_data_procesada[1] 
features_valid = lista_data_procesada[2] 
target_train = lista_data_procesada[3] 
target_test = lista_data_procesada[4] 
target_valid =lista_data_procesada[5]


best_model = modelos(features_train, target_train, features_valid, target_valid)

#df_predicciones = prediccioens(best_model, test)

