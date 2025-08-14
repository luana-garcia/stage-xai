from folktables import ACSDataSource, ACSIncome
import subprocess

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

import pandas as pd

class DataLoader:
    def __init__(self):
        self.data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')

    def set_data_usa(self):
        # Load A file
        file_1 = pd.read_csv("csv_pus/psam_pusa.csv",sep=",", low_memory=False)
        features_1,label_1,group_1 = ACSIncome.df_to_pandas(file_1)

        # # Load B file
        file_2 = pd.read_csv("csv_pus/psam_pusb.csv",sep=",", low_memory=False)
        features_2,label_2,group_2 = ACSIncome.df_to_pandas(file_2)

        # # Concatenate data for the USA group
        self.features_usa = pd.concat([features_1, features_2], ignore_index=True)
        self.label_usa = pd.concat([label_1, label_2], ignore_index=True)
        self.group_usa = pd.concat([group_1,group_2], ignore_index=True)

    def get_data_state(self, state):
        try:
            acs_data = self.data_source.get_data(states=[state], download=True)
        except Exception as e:
            print(f"Error downloading data with folktables: {e}")
            print(f"Trying to manually download data of {state}...")

            # Execute external script to download and extract data
            try:
                subprocess.run(["bash", "upload_state.sh", state], check=True)
            except subprocess.CalledProcessError as e2:
                print(f"Failed to execute download script: {e2}")
                raise e2  # Reraise the error if the script also fails

            # After manual download, try to charge again the data without `download=True`
            acs_data = self.data_source.get_data(states=[state], download=False)

        features, label, group = ACSIncome.df_to_pandas(acs_data)

        self.set_feature_names(features)

        return features, label, group
    
    def get_data_usa(self):
        if not hasattr(self, 'features_usa'):
            self.set_data_usa()
        return self.features_usa, self.label_usa, self.group_usa
    
    def set_feature_names(self, features):
        self.feature_names = list(features.columns)
    
    def train_test_data(self, features, label):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(features,label,train_size=0.9)
        return self.X_train, self.X_test, self.Y_train, self.Y_test
    
    '''
    The two-sample Kolmogorov-Smirnov test is a nonparametric hypothesis test that 
    evaluates the difference between the cdfs of the distributions of the two sample 
    data vectors over the range of x in each data set.
    '''
    def run_ks_test_2sample(self, state1, state2, alpha=0.05):
        features1, _, _ = self.get_data_state(state1)
        features2, _, _ = self.get_data_state(state2)

        print(f'\nKolmogorov-Smirnov test: {state1} vs {state2}')
        print(f"{'Variable':<10} {'p-value':<10} Significant?")
        print("-"*30)
        
        for col in features1.columns:
            _, p_value = ks_2samp(features1[col], features2[col])
            sig = "YES" if p_value < alpha else "NO"
            print(f"{col:<10} {p_value:.4f}    {sig}")

    def sample_variable(self, features, labels, variable,target_proportions,sample_size=100000):
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