from bin.data_loaders import DataLoader
from bin.data_trainers import DataTrainer

from xai.exp_anchors import ExpAnchors
from xai.exp_shap import ExpSHAP

import os
import json
import numpy as np
from tqdm import tqdm

class PythonObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (bool, np.bool_)):
            return str(obj)
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def export_anchors_html(anchors, loader, i):
    exp = anchors.run_anchors(loader.X_test.iloc[i].values)

    print('Anchor: %s' % (' AND '.join(exp.names())))
    print('Precision: %.2f' % exp.precision())
    print('Coverage: %.2f' % exp.coverage())

    html = exp.as_html()
    with open(f'xai/html/explanation{i}.html', "w") as f:
        f.write(html)

def save_partial_json(file_name, exps):
    with open(file_name, "w") as outfile:
        json.dump(exps, outfile, cls=PythonObjectEncoder, indent=4)

def export_anchors_json(anchors, loader, json_name):
    exps = {}

    #export explanations to json file
    json_path = 'xai/output/json'
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    file_name = f"{json_path}/{json_name}.json"

    # Barra de progresso principal
    progress_bar = tqdm(loader.X_test.iterrows(), 
                       total=len(loader.X_test), 
                       desc="Exporting anchor explanations")
    for i, row in progress_bar:
        anchors_exp = anchors.export_anchors_exp(row)
        exps[i] = anchors_exp
        if progress_bar.n % 1000 == 0 and progress_bar.n > 0:
            save_partial_json(file_name, exps)

    save_partial_json(file_name, exps)

def export_shap_json(shap, loader, json_name):
    exps = {}

    #export explanations to json file
    json_path = 'xai/output/json'
    if not os.path.exists(json_path):
        os.makedirs(json_path)
    file_name = f"{json_path}/{json_name}.json"

    # Barra de progresso principal
    progress_bar = tqdm(loader.X_test.iterrows(), 
                       total=len(loader.X_test), 
                       desc="Exporting shap explanations")
    for i, row in progress_bar:
        shap_exp = shap.export_shap_exp(row)
        exps[i] = shap_exp
        if progress_bar.n % 1000 == 0 and progress_bar.n > 0:
            save_partial_json(file_name, exps)

    save_partial_json(file_name, exps)

def run_models_by_state_anchors(state, trainer, loader):
    # Logistic Regression
    trainer.set_logistic_regression()
    trainer.train_state(state)
    anchors = ExpAnchors(loader.X_train, trainer.model, loader.feature_names)

    export_anchors_json(anchors, loader, f'lr_{state.lower()}_anchors_explanations')

    # XGBoost
    trainer.set_xgbclassifier()
    trainer.train_state(state)
    anchors.set_model(trainer.model)

    export_anchors_json(anchors, loader, f'xg_{state.lower()}_anchors_explanations')

    # Skrub
    trainer.set_skrub()
    trainer.train_state(state)
    anchors.set_model(trainer.model.named_steps['histgradientboostingclassifier'])

    export_anchors_json(anchors, loader, f'skrub_{state.lower()}_anchors_explanations')

def run_models_by_state_shap(state, trainer, loader):
    # Logistic Regression
    trainer.set_logistic_regression()
    trainer.train_state(state)
    shap = ExpSHAP(trainer.model, trainer.model_name, 'linear', loader.feature_names,  loader.X_train)

    export_shap_json(shap, loader, f'lr_{state.lower()}_shap_explanations')

    # XGBoost
    trainer.set_xgbclassifier()
    trainer.train_state(state)
    shap.set_model(trainer.model, trainer.model_name, 'tree')

    export_shap_json(shap, loader, f'xg_{state.lower()}_shap_explanations')

    # Skrub
    trainer.set_skrub()
    trainer.train_state(state)
    shap.set_model(trainer.model.named_steps['histgradientboostingclassifier'], trainer.model_name, 'tree')

    export_shap_json(shap, loader, f'skrub_{state.lower()}_shap_explanations')

loader = DataLoader()

trainer = DataTrainer(loader)

############### TEXAS ###############
run_models_by_state_anchors('TX', trainer, loader)
run_models_by_state_shap('TX', trainer, loader)

############### CALIFORNIA ###############
run_models_by_state_shap('CA', trainer, loader)
run_models_by_state_anchors('CA', trainer, loader)

# ############### NEW YORK ###############
run_models_by_state_shap('NY', trainer, loader)
run_models_by_state_anchors('NY', trainer, loader)