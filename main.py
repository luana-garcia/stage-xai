from data_loaders import DataLoader
from data_trainers import DataTrainer

def test_models(trainer):
    trainer.set_logistic_regression()
    trainer.comp_CM_per_state("TX","TX")
    trainer.comp_CM_per_state("usa","TX")
    trainer.show_roc_curve()

    trainer.set_xgbclassifier()
    trainer.comp_CM_per_state("TX","TX")
    trainer.comp_CM_per_state("usa","TX")
    trainer.show_roc_curve()

    trainer.set_skrub()
    trainer.comp_CM_per_state("TX","TX")
    trainer.comp_CM_per_state("usa","TX")
    trainer.show_roc_curve()

    trainer.set_nn()
    trainer.comp_CM_per_state("TX","TX")
    trainer.comp_CM_per_state("usa","TX")
    trainer.show_roc_curve()

def test_sampling_sex(trainer):
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

# loader.run_ks_test_2sample(features_ca, features_tx)

trainer = DataTrainer(loader)

test_models(trainer)
test_sampling_sex(trainer)