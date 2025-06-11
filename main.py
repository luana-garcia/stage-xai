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

loader = DataLoader()
features_ca, _, _ = loader.get_data_state('CA')
features_tx, _, _ = loader.get_data_state('TX')

# loader.run_ks_test_2sample(features_ca, features_tx)

trainer = DataTrainer(loader)

trainer.set_logistic_regression()
trainer.train_diff_proportions("TX", "SEX")

trainer.set_xgbclassifier()
trainer.train_diff_proportions("TX", "SEX")

trainer.set_skrub()
trainer.train_diff_proportions("TX", "SEX")

trainer.set_nn()
trainer.train_diff_proportions("TX", "SEX")

# test_models(trainer)