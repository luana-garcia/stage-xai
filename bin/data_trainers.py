from bias_measure_fcts import Cpt_DI,Cpt_EoO,Cpt_Suf
from simple_nn import SimpleNNclassifier
from math import sqrt

from xgboost import XGBClassifier
from skrub import tabular_pipeline

import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error,confusion_matrix,roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

class DataTrainer:
    def __init__(self, data_loader):
        self.loader = data_loader

    def phi(self, cm):
        n = len(str(np.max(cm)))
        cm_00 = cm[0,0]/10**n
        cm_01 = cm[0,1]/10**n
        cm_10 = cm[1,0]/10**n
        cm_11 = cm[1,1]/10**n
        return (cm_00*cm_11-cm_01*cm_10)/(sqrt((cm_00+cm_01)*(cm_01+cm_11)*(cm_11+cm_10)*(cm_10+cm_00)))
    
    def train_sex_diff_proportions(self, state_data, target_var):
        if state_data == "usa":
            features, label, _ = self.loader.get_data_usa()
        else:
            features, label, _ = self.loader.get_data_state(state_data)

        pred = []
        nb_it = 50
        for k in range(nb_it+1):
            p=k/nb_it
            target_proportions = {
                '1.0' : p, #Hommes
                '2.0' : 1-p  #Femmes
            }
            features_sample,label_sample = self.loader.sample_variable(features, label, target_var,target_proportions)

            X_train, X_test, Y_train, Y_test = self.loader.train_test_data(features_sample,label_sample,train_size=0.7)

            self.model.fit(X_train,Y_train.values.ravel())

            Y_sample_pred = self.model.predict(features_sample)
            cm = confusion_matrix(label_sample,Y_sample_pred)
            #self.comp_CM_per_state("sample","sample",model)
            pred.append([k,(cm[0,0]+cm[1,1])/sum(sum(cm))])

        plt.scatter([x[0] for x in pred],[x[1] for x in pred])
        plt.title(f"Confusion Matrix - {state_data}")
        plt.savefig(f"cm_states/{state_data}.png", dpi=180, bbox_inches='tight')
        plt.clf()

    def show_fairness_stats(self, features, label, Y_test_pred):
        print(self.phi(confusion_matrix(label, self.model.predict(features))))
        print(confusion_matrix(label, self.model.predict(features)),"\n")

        print("       - Disparate Impact =",Cpt_DI(2-features["SEX"].values,Y_test_pred.ravel()))
        print("       - Equality of Odds =",Cpt_EoO(2-features["SEX"].values,Y_test_pred.ravel(),label.values.ravel()))
        print("       - Sufficiency =",Cpt_Suf(2-features["SEX"].values,Y_test_pred.ravel(),label.values.ravel()),"\n")

    def comp_CM_per_state(self, state_data, state_model):
        if state_data == "usa":
            features, label, _ = self.loader.get_data_usa()
        else:
            features, label, _ = self.loader.get_data_state(state_data)

        self.train_state(state_model)
        X_test = self.loader.X_test
        Y_test = self.loader.Y_test

        # Si les 2 Etats sont identiques, on fait la CM uniquement sur les données de test
        # pour ne pas biaiser avec la proportion en données d'entraînement mieux entraînée
        if state_data == state_model:
            Y_test_pred = self.model.predict(X_test)

            print("Données :",state_data.upper(),"// Modèle fait sur :",state_model.upper())
            self.show_fairness_stats(X_test, Y_test, Y_test_pred)
        else:
            Y_test_pred = self.model.predict(features)

            print("Données :",state_data.upper(),"// Modèle fait sur :",state_model.upper())
            self.show_fairness_stats(features, label, Y_test_pred)

    def train_state(self, state_data):
        if state_data == "usa":
            features, label, _ = self.loader.get_data_usa()
        else:
            features, label, _ = self.loader.get_data_state(state_data)

        X_train, _, Y_train, _ = self.loader.train_test_data(features,label)

        # train the model
        if self.model_name == "NN":
            self.model.fit(
                X_train,
                Y_train.values.ravel(),
                epochs_nb=100,
                batch_size=300,
                optimizer='SGD',
                save=True,
                state=state_data
            )
        else:
            self.model.fit(
                X_train,
                Y_train.values.ravel()
            )

    def test_model(self):
        y_probs = self.model.predict(self.loader.X_test)
        return y_probs

    def set_logistic_regression(self):
        self.model_name = "lr"
        self.model = make_pipeline(
            StandardScaler().set_output(transform="pandas"),
            LogisticRegression(solver="lbfgs",max_iter=10000)
        )
    
    def set_xgbclassifier(self):
        self.model_name = "xgb"
        self.model = XGBClassifier(max_depth=6, n_estimators = 200, random_state=8)
    
    def set_skrub(self):
        self.model_name = "skrub"
        self.model=tabular_pipeline('classifier')
    
    def set_nn(self):
        self.model_name = "NN"
        self.model = SimpleNNclassifier(10)

    
    def show_roc_curve(self, save = False, state = ''):
        features_usa, label_usa, _ = self.loader.get_data_usa()

        if self.model_name == 'NN':
            y_probs = 1-self.model.predict_proba(features_usa).ravel()
        else:
            y_probs = self.model.predict_proba(features_usa)[:, 1]

        # Calcul des coordonnées ROC
        fpr, tpr, _ = roc_curve(label_usa, y_probs)

        # AUC
        auc_score = roc_auc_score(label_usa, y_probs)

        # Tracer la courbe ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.3f})", color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='grey')  # ligne diagonale
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f'Courbe ROC - {state} - {self.model_name}')
        plt.legend()
        plt.grid(True)

        if save:
            dir = './plots/roc_curve'
            os.makedirs(dir, exist_ok=True)
            save_path = os.path.join(dir, f'roc_curve_{state}_{self.model_name}.png')
            plt.savefig(save_path)
        else:
            plt.show()