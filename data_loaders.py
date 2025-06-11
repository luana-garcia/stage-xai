import folktables
from folktables import ACSDataSource, ACSIncome

from scipy.stats import ks_2samp
import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        # Load A file
        file_1 = pd.read_csv("csv_pus/psam_pusa.csv",sep=",")
        features_1,label_1,group_1 = ACSIncome.df_to_pandas(file_1)

        # Load B file
        file_2 = pd.read_csv("csv_pus/psam_pusb.csv",sep=",")
        features_2,label_2,group_2 = ACSIncome.df_to_pandas(file_2)

        # Concatenate data for the USA group
        self.features_usa = pd.concat([features_1, features_2], ignore_index=True)
        self.label_usa = pd.concat([label_1, label_2], ignore_index=True)
        self.group_usa = pd.concat([group_1,group_2], ignore_index=True)


        self.data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

    def get_data_state(self, state):
        acs_data = self.data_source.get_data(states=[state], download=True)
        features, label, group = ACSIncome.df_to_pandas(acs_data)
        return features, label, group
    
    def get_data_usa(self):
        return self.features_usa, self.label_usa, self.group_usa
    
    '''
    The two-sample Kolmogorov-Smirnov test is a nonparametric hypothesis test that 
    evaluates the difference between the cdfs of the distributions of the two sample 
    data vectors over the range of x in each data set.
    '''
    def run_ks_test_2sample(self, features1, features2):
        for col in features1.columns:
            print(col,":",ks_2samp(features1[col],features2[col])[1])

    def sample_variable(self, features, labels, variable,target_proportions,sample_size=100000):
        #p=0.5
        #target_proportions = {
        #    '1.0' : p, #Hommes
        #    '2.0' : 1-p  #Femmes
        #}

        #valeurs que l'on veut prendre par attribut dans le sample
        counts = {val: int(sample_size * prop) for val, prop in target_proportions.items()}

        features_sample = []
        label_sample = []

        for val, n in counts.items():
            df = features[features[variable].astype(str) == val].sample(n=n,replace=True)
            list_index = df.index.tolist()

            features_sample.append(df)

            label_sample.append(labels.iloc[list_index,:])
            

        features_sample = pd.concat(features_sample).sample(frac=1, random_state=0).reset_index(drop=True)
        label_sample = pd.concat(label_sample).sample(frac=1, random_state=0).reset_index(drop=True)
        return features_sample,label_sample