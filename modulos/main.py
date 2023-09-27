import pandas as pd

df = pd.read_csv('/datasets/users_behavior.csv')


list_data_procesada = procesamiento(df)

features_train = lista_data_procesada[0] 
features_test  = lista_data_procesada[1] 
features_valid = lista_data_procesada[2] 
target_train = lista_data_procesada[3] 
target_test = lista_data_procesada[4] 
target_valid =lista_data_procesada[5] 

best_model = modelamiento(train)

df_predicciones = prediccioens(best_model, test)

