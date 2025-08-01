from data_loaders import DataLoader
from data_trainers import DataTrainer

def test_models(trainer, state):
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

def test_sampling_sex_all_states(trainer):
    trainer.set_xgbclassifier()
    for s in us_states:
        try:
            trainer.train_sex_diff_proportions(s, "SEX")
        except:
            pass

us_states = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]

loader = DataLoader()

loader.run_ks_test_2sample('CA', 'TX')
loader.run_ks_test_2sample('CA', 'NY')
loader.run_ks_test_2sample('TX', 'NY')

# trainer = DataTrainer(loader)

# test_models(trainer, "CA")
# test_models(trainer, "TX")
# test_models(trainer, "NY")
# test_sampling_sex_all_states(trainer)