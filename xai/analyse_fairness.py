import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from xai.bias_measure_fcts import Cpt_DI

# SENSIBLE VARIABLE: 'SEX' (1 = homme, 2 = femme)

class AnalyseFairness:
    def __init__(self, loader, sensible_feature, trainer=None, file_path='xai/output/json/'):
        self.loader = loader
        self.sensible_feature = sensible_feature
        self.trainer = trainer
        self.file_path = file_path

        self.n_pca_components = 2
        self.n_clusters = 3

    def generate_exp_profile_analysis(self, exp_tests, save_plot=False):
        for f in exp_tests.get('files'):
            state = re.search(r'_([a-zA-Z]{2})_', f)
            state = state[1].upper()
            f_name = re.sub(r'_explanations\.json$', '', f)
            self.set_context(f, state)
            self.analyse_exp_profile(exp_tests, save_plot, f_name=f_name)

    def generate_exp_DI_analysis(self, exp_tests):
        for f in exp_tests.get('files'):
            state = re.search(r'_([a-zA-Z]{2})_', f)
            state = state[1].upper()
            
            self.set_context(f, state)
            print(f"\nAnalyse des données pour l'état: {state}")
            self.analyse_exp_DI_pca_range()
            self.analyse_exp_DI_by_cluster()

    def analyse_exp_DI_pca_range(self):
        x_range = (-2, 0)
        y_range = (-2, 0)
        _, points_ids = self.filter_pca_by_range(x_range, y_range)

        di = self.calculate_DI(points_ids)

        print(f"Données PCA range:\n PC1: {x_range}, PC2: {y_range}")
        print("       - Disparate Impact =", di)

    def analyse_exp_DI_by_cluster(self):
        cluster_ids = {i: [] for i in range(self.n_clusters)}
    
        for idx, cluster_id in enumerate(self.clusters):
            cluster_ids[cluster_id].append(idx)

        for points_ids in cluster_ids.values():
            di = self.calculate_DI(points_ids)

            cluster_points = self.X_test_pca[points_ids]
            center = np.mean(cluster_points, axis=0)

            if center[0] < 0 and center[1] < 0:
                label = 'Down Left'
            elif center[0] > 0 and center[1] > 0:
                label = 'Up'
            else:
                label = 'Down Right'
                
            print("Données Cluster:", label)
            print("       - Disparate Impact =", di)
    
    def calculate_DI(self, points_ids):
        X_test_genred = self.X_test[self.X_test.index.isin(points_ids)]

        Y_test_pred = np.array([])
        for p in self.exp_data.values():
            if p.get('row_num') in points_ids:
                Y_test_pred = np.append(Y_test_pred, p.get('prediction'))

        Y_test_pred = np.where(Y_test_pred == 'True', 1.0, 0.0)
        Y_test_pred = Y_test_pred.astype(np.float64)

        return Cpt_DI(2 - X_test_genred["SEX"].values, Y_test_pred, boxplot=True)

    def filter_pca_by_range(self, x_range, y_range):
        mask = (self.X_test_pca[:, 0] >= x_range[0]) & \
            (self.X_test_pca[:, 0] <= x_range[1]) & \
            (self.X_test_pca[:, 1] >= y_range[0]) & \
            (self.X_test_pca[:, 1] <= y_range[1])
        
        filtered_points = self.X_test_pca[mask]
        filtered_indices = np.where(mask)[0]
        
        return filtered_points, filtered_indices

    def analyse_exp_profile(self, exp_tests, save=False, f_name=''):
        _, exp_genred_ids = self.analyse_sensible_var_in_exp()

        mask_genred_exp = self.mask_filtered_ids(exp_genred_ids, self.X_test.index)
        
        self.plot_pca_train_test(mask_genred_exp, save, file_name=f_name)

        plot_tests = exp_tests.get('tests')
        test_numbers = []
        test_masks = []
        for t in plot_tests:
            conditions = t.get('conditions', [])
            values = t.get('values', [])

            exp_numbers, exp_masks = self.mask_tests(conditions, values)
            test_numbers.append(exp_numbers)
            test_masks.append(exp_masks)

        self.plot_exp_distribution_tests(plot_tests, test_numbers, save, file_name=f_name)
        
        self.plot_clustered_pca_tests(mask_genred_exp, plot_tests, test_masks, save, file_name=f_name)

    def set_context(self, exp_file, state):
        self.read_exp_data(exp_file)

        self.retrieve_train_test_data(self.exp_data, state)

        self.apply_pca_on_test_data()

        self.apply_pca_clustering()

    def mask_filtered_ids(self, filtered_ids, total_ids):
        filtered_ids = [int(id) for id in filtered_ids]
        exp_mask = total_ids.isin(filtered_ids)

        return exp_mask
    
    def run_condition_test(self, conditions, cond_values):
        exp_true, exp_ids_true = self.analyse_sensible_var_in_exp(conditions, ['True'] + cond_values)
        exp_false, exp_ids_false = self.analyse_sensible_var_in_exp(conditions, ['False'] + cond_values)

        return exp_true, exp_ids_true, exp_false, exp_ids_false

    def mask_tests(self, conditions, conditions_values):
        exp_true, exp_ids_true, exp_false, exp_ids_false = self.run_condition_test(conditions, conditions_values)

        mask_true_ids = self.mask_filtered_ids(exp_ids_true, self.X_test.index)
        mask_false_ids = self.mask_filtered_ids(exp_ids_false, self.X_test.index)

        men_ids = self.X_test.index[self.X_test[self.sensible_feature].isin([1])].tolist()
        mask_men_ids = self.mask_filtered_ids(men_ids, self.X_test.index)

        women_ids = self.X_test.index[self.X_test[self.sensible_feature].isin([2])].tolist()
        mask_women_ids = self.mask_filtered_ids(women_ids, self.X_test.index)

        mask_male_true = mask_true_ids & mask_men_ids
        mask_male_false = mask_false_ids & mask_men_ids
        mask_female_true = mask_true_ids & mask_women_ids
        mask_female_false = mask_false_ids & mask_women_ids

        return [exp_true, exp_false], [mask_male_true, mask_male_false, mask_female_true, mask_female_false]

    def read_exp_data(self, file_name):
        file = self.file_path + file_name
        with open(file, 'r', encoding='utf-8') as f:
            self.exp_data = json.load(f)

    def retrieve_train_test_data(self, exp, state):
        test_ids = [int(id) for id in exp.keys()]

        features, labels, _ = self.loader.get_data_state(state)

        self.X_train = features[~features.index.isin(test_ids)]

        self.X_test = features.loc[test_ids]
        self.Y_test = labels.loc[test_ids]

    def apply_pca_on_test_data(self):
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        self.pca = PCA(n_components=self.n_pca_components)
        self.X_train_pca = self.pca.fit_transform(X_train_scaled)
        self.X_test_pca = self.pca.transform(X_test_scaled)

    def apply_pca_clustering(self):
        kmeans = KMeans(n_clusters=self.n_clusters)
        self.clusters = kmeans.fit_predict(self.X_test_pca)

    def plot_pca_train_test(self, exp_mask, save = False, file_name = ''):
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        fig.suptitle("PCA Space", fontsize=16)
        axs = axs.ravel()

        # Plot train and test data
        axs[0].scatter(self.X_train_pca[:, 0], self.X_train_pca[:, 1], alpha=0.9, label='Train Data', c='gray', marker='x')
        axs[0].scatter(self.X_test_pca[:, 0], self.X_test_pca[:, 1], label='Test Data', c='red', marker='x')
        axs[0].set_xlabel('PC1 (Variance: {:.2f}%)'.format(self.pca.explained_variance_ratio_[0]*100))
        axs[0].set_ylabel('PC2 (Variance: {:.2f}%)'.format(self.pca.explained_variance_ratio_[1]*100))
        axs[0].legend()
        axs[0].set_title('PCA Distribution: Train Set vs. Test Set')

        # Plot anchors in test data
        axs[1].scatter(self.X_test_pca[~exp_mask, 0], self.X_test_pca[~exp_mask, 1], alpha=0.9, label='Test Data', c='red', marker='x')
        axs[1].scatter(self.X_test_pca[exp_mask, 0], self.X_test_pca[exp_mask, 1], label='Genred Anchors', c='#8B1818', marker='x')
        axs[1].legend()
        axs[1].set_title('PCA Distribution: Test Set vs. Genred Anchors in it')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        if save:
            dir = './plots/pca'
            os.makedirs(dir, exist_ok=True)
            save_path = os.path.join(dir, f'pca_{file_name}.png')
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_clustered_pca_tests(self, exp_mask, personalized_tests, test_masks, save = False, file_name = ''):
        num_plots = len(personalized_tests)

        ncols = 3   # 3 columns
        nrows = (num_plots + 2 + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
        fig.suptitle("Anchors Analysis in PCA Space")
        axs = axs.ravel()

        axs[0].scatter(self.X_test_pca[:, 0], self.X_test_pca[:, 1], c=self.clusters, cmap='viridis')
        axs[0].set_title(f'KMeans Clustering (n_clusters={self.n_clusters})')

        axs[1].scatter(self.X_test_pca[:, 0], self.X_test_pca[:, 1], alpha=0.7, c=self.clusters, cmap='viridis')
        axs[1].scatter(self.X_test_pca[exp_mask, 0], self.X_test_pca[exp_mask, 1], c='#8B1818', label='Genred Anchors')
        axs[1].set_title('PCA Distribution: Test Set vs. Genred Anchors in it')
        axs[1].legend()

        ax_num = 2
        for t, m in zip(personalized_tests, test_masks):
            title = t.get('description', '')
            self.plot_exp_clustered_pca(m, title, axs[ax_num])
            ax_num += 1

        for j in range(num_plots+2, len(axs)):
            axs[j].axis('off')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        if save:
            dir = './plots/clustered_pca'
            os.makedirs(dir, exist_ok=True)
            save_path = os.path.join(dir, f'clusters_{file_name}.png')
            fig.savefig(save_path)

            plt.close(fig)
        else:
            fig.show()

    def plot_exp_clustered_pca(self, test_masks, title='', ax = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        colors=["#1952E1", "#3DD642", "#C280E6", "#D38217"]

        # Plot points
        ax.scatter(self.X_test_pca[:, 0], self.X_test_pca[:, 1], alpha=0.9, c=self.clusters, cmap='viridis')
        ax.scatter(self.X_test_pca[test_masks[0], 0], self.X_test_pca[test_masks[0], 1], c=colors[0], label='Men (True)')
        ax.scatter(self.X_test_pca[test_masks[1], 0], self.X_test_pca[test_masks[1], 1], c=colors[1], label='Men (False)')
        ax.scatter(self.X_test_pca[test_masks[2], 0], self.X_test_pca[test_masks[2], 1], c=colors[2], label='Women (True)')
        ax.scatter(self.X_test_pca[test_masks[3], 0], self.X_test_pca[test_masks[3], 1], c=colors[3], label='Women (False)')
        ax.set_title(title)
        ax.legend()

    def plot_exp_distribution_tests(self, plot_tests, exp_numbers, save = False, file_name=''):
        # Create a figure with multiple subplots
        num_plots = len(plot_tests)

        ncols = 2     # 2 columns
        nrows = (num_plots + ncols - 1) // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
        fig.suptitle(f"Distribution of Anchors by {self.sensible_feature} feature:\nTotal of test data: {len(self.exp_data.items())}")
        axs = axs.ravel()  # Flatten the array of axes

        ax_num = 0
        for t, m in zip(plot_tests, exp_numbers):
            title = t.get('description', '')
            self.plot_exp_distribution(m, title, axs[ax_num])
            ax_num += 1

        # Hide the last subplot if we have an odd number of plots
        for j in range(num_plots, len(axs)):
            axs[j].axis('off')

        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        if save:
            dir = './plots/anchors_distribution'
            os.makedirs(dir, exist_ok=True)
            save_path = os.path.join(dir, f'pca_{file_name}.png')
            fig.savefig(save_path)

            plt.close(fig)
        else:
            fig.show()

    def plot_exp_distribution(self, exp_numbers, title='', ax=None):
        # Process data
        true_pred, false_pred = exp_numbers
        categories = ['Male (SEX=1)', 'Female (SEX=2)', 'Both (SEX<=2)']
        true_counts = [true_pred.get('SEX <= 1.00', 0) or true_pred.get('SEX = 1', 0), 
                    true_pred.get('SEX > 1.00', 0) or true_pred.get('SEX = 2', 0), 
                    true_pred.get('SEX <= 2.00', 0)]
        false_counts = [false_pred.get('SEX <= 1.00', 0) or false_pred.get('SEX = 1', 0), 
                        false_pred.get('SEX > 1.00', 0) or false_pred.get('SEX = 2', 0), 
                        false_pred.get('SEX <= 2.00', 0)]

        x = np.arange(len(categories))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, true_counts, width, label='True Predictions', color='#D38217')
        rects2 = ax.bar(x + width/2, false_counts, width, label='False Predictions', color='#1952E1')
        
        # Adiciona texto e formatação
        ax.set_ylabel('Anchors count')
        ax.set_title(title, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontweight='bold')
        ax.legend()
        
        # Adiciona valores nas barras
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontweight='bold')
        
        autolabel(rects1)
        autolabel(rects2)

    def verify_conditions(self, exp_value, condition, cond_value, verify_rank_position=50):
        match condition:
            case 'precision':
                condition = exp_value.get('precision') > cond_value
            case 'prediction':
                pred_value = exp_value.get('prediction')

                if isinstance(pred_value, str):
                    pred_bool = pred_value.lower() == 'true'
                else:
                    pred_bool = bool(pred_value)
                
                if isinstance(cond_value, str):
                    v_bool = cond_value.lower() == 'true'
                else:
                    v_bool = bool(cond_value)
                
                condition = (pred_bool == v_bool)
            case 'coverage':
                try:
                    condition = exp_value.get('coverage') > cond_value
                except KeyError:
                    condition = False
            case 'num_features':
                try:
                    condition = len(exp_value.get('features')) <= cond_value
                except KeyError:
                    condition = False
            case 'rank_position':
                verify_rank_position = cond_value
            case _:
                condition = True

        return condition, verify_rank_position

    def analyse_sensible_var_in_exp(self, cond = [], cond_value = []):
        if len(cond) != len(cond_value):
            raise ValueError("As listas 'cond' e 'cond_value' devem ter o mesmo tamanho")
            
        sensible_exp = dict()
        sensible_exp_ids = []
        for key, value in self.exp_data.items():
            all_cond = True
            verify_rank_position = 50
            if len(cond) != 0:
                for c, v in zip(cond, cond_value):
                    condition, verify_rank_position = self.verify_conditions(value, c, v, verify_rank_position)
                    all_cond = all_cond and condition
            
            if all_cond:
                for i, f in enumerate(value.get('features')):
                    if f.get('feature_name') == self.sensible_feature and i <= verify_rank_position:
                        feature_range = f.get('feature_ranges')
                        feature_value = f.get('feature_value')

                        if feature_range is not None:
                            label = feature_range
                        else:
                            label = f'{self.sensible_feature} = {feature_value}'

                        # label = feature_range if feature_range is not None else feature_value
                            
                        if label not in sensible_exp:
                            sensible_exp[label] = 0
                        
                        sensible_exp[label] += 1
                        sensible_exp_ids.append(key)

        return sensible_exp, sensible_exp_ids