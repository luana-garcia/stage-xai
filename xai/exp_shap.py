import shap
import joblib as jbl
import pandas as pd
import numpy as np

class ExpSHAP:
    def __init__(self, ml_model, model_name, model_type, feature_names, X_train, shap_file_explainer = 'shap_explainer'):
        self.feature_names = feature_names
        self.ml_model = ml_model
        self.model_name = model_name
        self.model_type = model_type
        self.train_data = X_train
        self.shap_file = shap_file_explainer

        self.set_shap_instance()

    def set_shap_instance(self):
        shap_file_name = f'xai/{self.shap_file}_{self.model_name}'
        # Setting up SHAP
        try: 
            with open(shap_file_name, 'rb') as f:
                self.shap = jbl.load(f)
        except:
            if self.model_type == 'tree':
                self.shap = shap.TreeExplainer(self.ml_model)
            else:
                self.shap = shap.LinearExplainer(self.ml_model.named_steps['logisticregression'], 
                shap.sample(self.ml_model.named_steps['standardscaler'].transform(self.train_data), 100))
            print("Generating shap explainer...")
            with open(shap_file_name, 'wb') as f:
                jbl.dump(self.shap, f)

    def set_model(self, new_model, model_name, model_type):
        self.ml_model = new_model
        self.model_name = model_name
        self.model_type = model_type

        self.set_shap_instance()

    def run_shap(self, row):
        shap_values = self.shap.shap_values(row)
        return shap_values
    
    def format_row(self, row):
        row_num = row.name if isinstance(row, pd.Series) else row.index[0]

        # Convert input to proper DataFrame with feature names
        if isinstance(row, pd.Series):
            row = pd.DataFrame([row], columns=self.feature_names)
        elif isinstance(row, np.ndarray):
            row = pd.DataFrame(row.reshape(1, -1), columns=self.feature_names)
        else:
            row = row[self.feature_names]  # For DataFrame input

        return row, row_num
    
    # function to help sorting features
    def compare_shap_feature_weights(self, feature_value):
        if self.prediction_bool:
            # highest weights first
            return -feature_value['feature_weight']
        else:
            return feature_value['feature_weight']
    
    # export shap features' explanations
    def export_shap_exp(self, row):
        row_df, row_num = self.format_row(row)
        shap_values = self.run_shap(row_df)

        predict = self.ml_model.predict_proba(row_df)[0]
        self.prediction_bool = 1 if predict[1] >= 0.5 else 0
        
        # get shap values to refactor instance
        shap_output = dict()

        shap_output['row_num'] = row_num
        shap_output['prediction'] = self.prediction_bool
        shap_output['precision'] = predict[self.prediction_bool]
        
        features_exp = []
        for i in range(0, len(self.feature_names)):
            if len(shap_values) > 1:
                shap_values_to_refactor = shap_values[self.prediction_bool][i]
            else:
                shap_values_to_refactor = shap_values[0][i]

            condition = shap_values_to_refactor > 0 if self.prediction_bool else shap_values_to_refactor < 0
            # shap_values has positive values (to refactor) and negative ones (not to refactor)
            if condition:
                f = dict()
                feature_name = self.feature_names[i]
                f['feature_name'] = feature_name
                f['feature_value'] = int(row[feature_name])
                f['feature_weight'] = shap_values_to_refactor
                f['feature_ranges'] = None
                features_exp.append(f)
        # sort features by feature weight
        sorted_features_exp = sorted(features_exp, key=self.compare_shap_feature_weights)
        # append feature ranking after sort
        rank = 1
        for f in sorted_features_exp:
            f['feature_rank'] = rank
            rank += 1
        shap_output['features'] = sorted_features_exp
        return shap_output