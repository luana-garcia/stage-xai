from bin.data_loaders import DataLoader
from bin.data_trainers import DataTrainer
from xai.analyse_fairness import AnalyseFairness

loader = DataLoader()
trainer = DataTrainer(loader)

file_path = 'xai/output/json/'

sensible_feature = 'SEX'

fairness = AnalyseFairness(loader, sensible_feature, trainer, file_path)

di_test = {
    'files': [
        'skrub_tx_anchors_explanations.json',
        'skrub_ca_anchors_explanations.json',
        'skrub_ny_anchors_explanations.json'
    ],
    'tests': [
        {'conditions': ['prediction', 'precision', 'coverage'], 'values': [0.95, 0.1], 'description': 'Prediction = True/False; Precision > 0.95;\n Coverage > 0.1'}
    ]
}

fairness.generate_exp_DI_analysis(di_test)