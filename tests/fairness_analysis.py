from bin.data_loaders import DataLoader
from bin.data_trainers import DataTrainer
from xai.analyse_fairness import AnalyseFairness

loader = DataLoader()
trainer = DataTrainer(loader)

file_path = 'xai/output/json/'

sensible_feature = 'SEX'

fairness = AnalyseFairness(loader, sensible_feature, trainer, file_path)

anchors_tests = {
    'files': [
        'skrub_tx_anchors_explanations.json',
        'skrub_ca_anchors_explanations.json',
        'skrub_ny_anchors_explanations.json',
        'xg_tx_anchors_explanations.json',
        'xg_ca_anchors_explanations.json',
        'xg_ny_anchors_explanations.json'
    ],
    'tests': [
        {'conditions': ['prediction'], 'values': [], 'description': 'Prediction = True/False'},
        {'conditions': ['prediction', 'precision'], 'values': [0.95], 'description': 'Prediction = True/False; Precision > 0.95'},
        {'conditions': ['prediction', 'precision', 'num_features'], 'values': [0.95, 3], 'description': 'Prediction = True/False; Precision > 0.95;\n Num_features <= 3'},
        {'conditions': ['prediction', 'precision', 'coverage'], 'values': [0.95, 0.1], 'description': 'Prediction = True/False; Precision > 0.95;\n Coverage > 0.1'}
    ]
}

shap_tests = {
    'files': [
        'skrub_tx_shap_explanations.json',
        'skrub_ca_shap_explanations.json',
        'skrub_ny_shap_explanations.json',
        'xg_tx_shap_explanations.json',
        'xg_ca_shap_explanations.json',
        'xg_ny_shap_explanations.json'
    ],
    'tests': [
        {'conditions': ['prediction'], 'values': [], 'description': 'Prediction = True/False'},
        {'conditions': ['prediction', 'precision'], 'values': [0.95], 'description': 'Prediction = True/False; Precision > 0.95'},
        {'conditions': ['prediction', 'precision', 'rank_position'], 'values': [0.95, 3], 'description': 'Prediction = True/False; Precision > 0.95;\n Rank position <= 3'}
    ]
}

fairness.generate_exp_profile_analysis(anchors_tests, save_plot=True)
fairness.generate_exp_profile_analysis(shap_tests, save_plot=True)