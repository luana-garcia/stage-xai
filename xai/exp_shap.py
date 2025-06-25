import shap
import joblib as jbl

# function to help sorting features
def compare_shap_feature_weights(feature_value):
    # highest weights first
    return -feature_value['feature_weight']

class ExpSHAP:
    def __init__(self, ml_model, feature_names, shap_file_explainer):
        self.feature_names = feature_names
        # Setting up SHAP
        try: 
            with open(shap_file_explainer, 'rb') as f:
                self.shap = jbl.load(f)
        except:
            self.shap = shap.TreeExplainer(ml_model)
            print("Generating shap explainer...")
            with open(shap_file_explainer, 'wb') as f:
                jbl.dump(self.shap, f)

    def run_shap(self, row):
        shap_values = self.shap.shap_values(row)
        return shap_values
    
    # export shap features' explanations
    def export_shap_exp(self, row, prediction_bool):
        shap_values = self.run_shap(row)
        
        # get shap values to refactor instance
        shap_output = dict()
        
        features_exp = []
        features_weights = []
        for i in range(0, len(self.feature_names)):
            shap_values_to_refactor = shap_values[i][prediction_bool]
            condition = shap_values_to_refactor > 0 if prediction_bool else shap_values_to_refactor < 0
            # shap_values has positive values (to refactor) and negative ones (not to refactor)
            if condition:
                f = dict()
                feature_name = self.feature_names[i]
                f['feature_name'] = feature_name
                f['feature_value'] = int(row[feature_name])
                features_weights.append(shap_values_to_refactor)
                f['feature_ranges'] = None
                features_exp.append(f)
        # scale weights to sum to 1
        # scaled_features_exp = scale_weights(features_weights, features_exp)
        # sort features by feature weight
        sorted_features_exp = sorted(features_exp, key=compare_shap_feature_weights)
        # append feature ranking after sort
        rank = 1
        for f in sorted_features_exp:
            f['feature_rank'] = rank
            rank += 1
        shap_output['features'] = sorted_features_exp
        return shap_output