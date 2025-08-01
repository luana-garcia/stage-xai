from bin.data_loaders import DataLoader
from bin.data_trainers import DataTrainer

def run_models_by_state(trainer, state):
    print('Logistic Regression:')
    trainer.set_logistic_regression()
    trainer.comp_CM_per_state(state,state)
    trainer.comp_CM_per_state("usa",state)
    trainer.show_roc_curve(True, state)
    
    print('XGBoost:')
    trainer.set_xgbclassifier()
    trainer.comp_CM_per_state(state,state)
    trainer.comp_CM_per_state("usa",state)
    trainer.show_roc_curve(True, state)
    
    print('Skrub:')
    trainer.set_skrub()
    trainer.comp_CM_per_state(state,state)
    trainer.comp_CM_per_state("usa",state)
    trainer.show_roc_curve(True, state)

    print('NN:')
    trainer.set_nn()
    trainer.comp_CM_per_state(state,state)
    trainer.comp_CM_per_state("usa",state)
    trainer.show_roc_curve(True, state)

loader = DataLoader()
trainer = DataTrainer(loader)

run_models_by_state(trainer, "CA")
run_models_by_state(trainer, "TX")
run_models_by_state(trainer, "NY")