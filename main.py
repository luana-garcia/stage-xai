from bin.data_loaders import DataLoader
from bin.data_trainers import DataTrainer

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

trainer = DataTrainer(loader)
# test_sampling_sex_all_states(trainer)