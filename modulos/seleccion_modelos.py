import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split



def modelos(features_train, target_train, features_valid, target_valid):
    best_score_rt = 0
    best_dept_rt = 0

    # Arbol de decisión
    for depth in range(1,11):
        
        tree_model = DecisionTreeClassifier(random_state=12345, max_depth=depth)
        
        tree_model.fit(features_train, target_train)
        predictions_valid = tree_model.predict(features_valid)
        
        score = accuracy_score(target_valid, predictions_valid)
        
        if score > best_score_rt:
            best_score_rt = score
            best_depth_rt = depth
            best_random_tree = tree_model

        
    # Bosque Aleatorio    
    best_score_rf = 0
    best_est_rf = 0
    best_depth_rf = 0

    for est in range(10,51,10):
        for depth in range(1,11):
            
            forest_model= RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth)
            forest_model.fit(features_train, target_train)
            
            score = forest_model.score(features_valid, target_valid)
            
            if score > best_score_rf:
                
                best_score_rf = score
                best_est_rf = est
                best_depth_rf = depth
                
                best_random_forest = forest_model    
    
    # Modelo de regresion
    regression_model = LogisticRegression(random_state=12345, solver='liblinear')
    regression_model.fit(features_train, target_train)
    score_lg = regression_model.score(features_valid, target_valid)

    if (score_lg >= best_score_rt) & (score_lg >= best_score_rf):
        best_model = regression_model
        best_score = score_lg
        model_name = 'regresion_logistica'
    elif best_score_rt >= best_score_rf:
        best_model = best_random_tree
        best_score = best_score_rt
        model_name = 'arbol de decision'
    else:
        best_model = best_random_forest
        best_score = best_score_rf
        model_name = 'random_foresct'
        
    print(f'El mejor modelo es {model_name} con un valor de accurary de {best_score}')
    
    
    return best_model