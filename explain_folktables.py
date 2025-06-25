from data_loaders import DataLoader
from data_trainers import DataTrainer

from xai.exp_anchors import ExpAnchors

import os
import json
import numpy as np
from tqdm import tqdm

class PythonObjectEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (bool, np.bool_)):
            return str(obj)
        elif isinstance(obj, (np.int64, np.int32, np.float64)):
            return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
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
        anchors_exp = anchors.export_anchors_exp(row, row.values)
        exps[i] = anchors_exp
        if progress_bar.n % 1000 == 0 and progress_bar.n > 0:
            save_partial_json(file_name, exps)

    save_partial_json(file_name, exps)

def run_models_by_state(state, trainer, loader):
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

loader = DataLoader()

trainer = DataTrainer(loader)

############### TEXAS ###############
run_models_by_state('TX', trainer, loader)

############### CALIFORNIA ###############
run_models_by_state('CA', trainer, loader)

############### NEW YORK ###############
run_models_by_state('NY', trainer, loader)