import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def procesamiento(df):
    
    df_train, df_40 = train_test_split(df, test_size=0.40, random_state=12345)

    df_valid, df_test = train_test_split(df_40, test_size=0.50, random_state=12345)
    
    # Establecemos las características y objetivo del conjunto de datos para el entrenamiento
    features_train = df_train.drop('is_ultra', axis=1)
    target_train = df_train['is_ultra']

    # Establecemos las características y objetivo del conjunto de datos para la validación
    features_valid = df_valid.drop('is_ultra', axis=1)
    target_valid = df_valid['is_ultra']

    # Establecemos las características y objetivo del conjunto de datos para la prueba
    features_test = df_test.drop('is_ultra', axis=1)
    target_test = df_test['is_ultra']
        
    scaler = MinMaxScaler()
    scaler.fit(features_train)
    
    features_train = scaler.transform(features_train)
    features_test = scaler.transform(features_test)
    features_valid = scaler.transform(features_valid)
    
    return [features_train, features_test, features_valid, target_train, target_test, target_valid]