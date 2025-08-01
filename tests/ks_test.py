from bin.data_loaders import DataLoader    

loader = DataLoader()

loader.run_ks_test_2sample('CA', 'TX')
loader.run_ks_test_2sample('CA', 'NY')
loader.run_ks_test_2sample('TX', 'NY')