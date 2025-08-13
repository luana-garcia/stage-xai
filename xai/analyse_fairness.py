from bin.data_loaders import DataLoader
from bin.data_trainers import DataTrainer

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import re

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def plot_sex_anchors_comparison(ax, true_pred, false_pred, title_suffix=""):
    # Process data
    categories = ['Male (SEX=1)', 'Female (SEX=2)', 'Both (SEX<=2)']
    true_counts = [true_pred.get('SEX <= 1.00', 0) or true_pred.get('SEX = 1', 0), 
                   true_pred.get('SEX > 1.00', 0) or true_pred.get('SEX = 2', 0), 
                   true_pred.get('SEX <= 2.00', 0)]
    false_counts = [false_pred.get('SEX <= 1.00', 0) or false_pred.get('SEX = 1', 0), 
                    false_pred.get('SEX > 1.00', 0) or false_pred.get('SEX = 2', 0), 
                    false_pred.get('SEX <= 2.00', 0)]
    
    # Configure graphic
    x = np.arange(len(categories))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, true_counts, width, label='True Predictions', color='#D38217')
    rects2 = ax.bar(x + width/2, false_counts, width, label='False Predictions', color='#1952E1')
    
    # Adiciona texto e formatação
    ax.set_ylabel('Anchors count')
    ax.set_title(title_suffix, pad=15)
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

def analyse_sex_anchors(data, cond = [], cond_value = []):
    if len(cond) != len(cond_value):
        raise ValueError("As listas 'cond' e 'cond_value' devem ter o mesmo tamanho")
        
    sex_anchors = dict()
    sex_anchors_ids = []
    for key, value in data.items():
        all_cond = True
        verify_rank_position = 50
        if len(cond) != 0:
            for c, v in zip(cond, cond_value):
                if c == 'precision':
                    condition = value.get('precision') > v
                elif c == 'prediction':
                    pred_value = value.get('prediction')

                    if isinstance(pred_value, str):
                        pred_bool = pred_value.lower() == 'true'
                    else:
                        pred_bool = bool(pred_value)
                    
                    if isinstance(v, str):
                        v_bool = v.lower() == 'true'
                    else:
                        v_bool = bool(v)
                    
                    condition = (pred_bool == v_bool)
                elif c == 'coverage':
                    try:
                        condition = value.get('coverage') > v
                    except KeyError:
                        condition = False
                elif c == 'num_features':
                    try:
                        condition = len(value.get('features')) <= v
                    except KeyError:
                        condition = False
                elif c == 'rank_position':
                    verify_rank_position = v
                else:
                    condition = True
                all_cond = all_cond and condition
        
        if all_cond:
            for i, f in enumerate(value.get('features')):
                if f.get('feature_name') == 'SEX' and i <= verify_rank_position:
                    feature_range = f.get('feature_ranges')
                    feature_value = f.get('feature_value')

                    if feature_range is not None:
                        label = feature_range
                    else:
                        if feature_value == 1:
                            label = 'SEX = 1'
                        elif feature_value == 2:
                            label = 'SEX = 2'
                        else:
                            label = f'SEX = {feature_value}'

                    # label = feature_range if feature_range is not None else feature_value
                        
                    if label not in sex_anchors:
                        sex_anchors[label] = 0
                    
                    sex_anchors[label] += 1
                    sex_anchors_ids.append(key)

    return sex_anchors, sex_anchors_ids

# HOMME - 1
# FEMME - 2

def plot_pca(X_train, X_test, mask_anchors, save = False, file_name = ''):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2)  # Para 3D, use n_components=3
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle("PCA Space", fontsize=16)
    axs = axs.ravel()

    axs[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=0.9, label='Train Data', c='gray', marker='x')
    axs[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], label='Test Data', c='red', marker='x')
    axs[0].set_xlabel('PC1 (Variance: {:.2f}%)'.format(pca.explained_variance_ratio_[0]*100))
    axs[0].set_ylabel('PC2 (Variance: {:.2f}%)'.format(pca.explained_variance_ratio_[1]*100))
    axs[0].legend()
    axs[0].set_title('PCA Distribution: Train Set vs. Test Set')

    axs[1].scatter(X_test_pca[~mask_anchors, 0], X_test_pca[~mask_anchors, 1], alpha=0.9, label='Test Data', c='red', marker='x')
    axs[1].scatter(X_test_pca[mask_anchors, 0], X_test_pca[mask_anchors, 1], label='Sexed Anchors', c='#8B1818', marker='x')
    axs[1].legend()
    axs[1].set_title('PCA Distribution: Test Set vs. Sexed Anchors in it')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if save:
        dir = './plots/pca'
        os.makedirs(dir, exist_ok=True)
        save_path = os.path.join(dir, f'pca_{file_name}.png')
        plt.savefig(save_path)
    else:
        plt.show()

    return X_test_pca

def cluster_pca(X_test_pca, n_clusters=3, ax = None):
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(X_test_pca)

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 12))

    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=clusters, cmap='viridis')
    ax.set_title(f'KMeans Clustering (n_clusters={n_clusters})')

    return clusters

def mask_anchors_ids(anchors, X_test_ids):
    _, anchors_ids = analyse_sex_anchors(anchors)

    anchors_ids = [int(id) for id in anchors_ids]
    mask_anchors = X_test_ids.isin(anchors_ids)

    return mask_anchors

def plot_anchors_pca(X_test_pca, clusters, mask_male_true, mask_male_false, mask_female_true, mask_female_false, title='', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    colors=["#1952E1", "#3DD642", "#C280E6", "#D38217"]
    # Plot points
    ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], alpha=0.9, c=clusters, cmap='viridis')
    ax.scatter(X_test_pca[mask_male_true, 0], X_test_pca[mask_male_true, 1], c=colors[0], label='Men (True)')
    ax.scatter(X_test_pca[mask_male_false, 0], X_test_pca[mask_male_false, 1], c=colors[1], label='Men (False)')
    ax.scatter(X_test_pca[mask_female_true, 0], X_test_pca[mask_female_true, 1], c=colors[2], label='Women (True)')
    ax.scatter(X_test_pca[mask_female_false, 0], X_test_pca[mask_female_false, 1], c=colors[3], label='Women (False)')
    ax.set_title(title)
    ax.legend()

def mask_anchors_4divisions(test_data, anchors, conditions, conditions_values, X_test_pca, clusters, ax_pca=None, ax_hist = None, title=''):
    anchors_true, anchors_ids_true = analyse_sex_anchors(anchors, conditions, ['True'] + conditions_values)
    anchors_false, anchors_ids_false = analyse_sex_anchors(anchors, conditions, ['False'] + conditions_values)

    anchors_ids_true = [int(id) for id in anchors_ids_true]
    anchors_ids_false = [int(id) for id in anchors_ids_false]

    man_indices = test_data.index[test_data['SEX'].isin([1])].tolist()
    woman_indices = test_data.index[test_data['SEX'].isin([2])].tolist()

    mask_male_true = test_data.index.isin(anchors_ids_true) & test_data.index.isin(man_indices)
    mask_male_false = test_data.index.isin(anchors_ids_false) & test_data.index.isin(man_indices)
    mask_female_true = test_data.index.isin(anchors_ids_true) & test_data.index.isin(woman_indices)
    mask_female_false = test_data.index.isin(anchors_ids_false) & test_data.index.isin(woman_indices)

    plot_anchors_pca(X_test_pca, clusters, mask_male_true, mask_male_false, mask_female_true, mask_female_false, title, ax_pca)
    plot_sex_anchors_comparison(ax_hist, anchors_true, anchors_false, title)


def analyse_clustered_pca(file_path, loader, plot_tests, state, save=False, file_name=''):
    with open(file_path, 'r', encoding='utf-8') as f:
        anchors = json.load(f)

    X_test_ids = [int(id) for id in anchors.keys()]

    data, _, _ = loader.get_data_state(state)

    X_train = data[~data.index.isin(X_test_ids)]

    X_test = data.loc[X_test_ids]

    mask_anchors = mask_anchors_ids(anchors, X_test.index)

    X_test_pca = plot_pca(X_train, X_test, mask_anchors, save=save, file_name=file_name)

    # Create a figure with multiple subplots
    num_plots = len(plot_tests)

    ncols_pca = 3     # 3 columns
    nrows_pca = (num_plots + 2 + ncols_pca - 1) // ncols_pca

    fig_pca, axs_pca = plt.subplots(nrows_pca, ncols_pca, figsize=(ncols_pca*5, nrows_pca*5))
    fig_pca.suptitle("Anchors Analysis in PCA Space")
    axs_pca = axs_pca.ravel()

    # Create a figure with multiple subplots
    ncols_hist = 2     # 2 columns
    nrows_hist = (num_plots + ncols_hist - 1) // ncols_hist
    fig_hist, axs_hist = plt.subplots(nrows_hist, ncols_hist, figsize=(ncols_pca*4, nrows_pca*3))
    fig_hist.suptitle(f"Distribution of Anchors by 'SEX' feature:\nTotal of test data: {len(anchors.items())}")
    axs_hist = axs_hist.ravel()  # Flatten the array of axes

    clusters = cluster_pca(X_test_pca, n_clusters=3, ax = axs_pca[0])

    axs_pca[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], alpha=0.7, c=clusters, cmap='viridis')
    axs_pca[1].scatter(X_test_pca[mask_anchors, 0], X_test_pca[mask_anchors, 1], c='#8B1818', label='Sexed Anchors')
    axs_pca[1].set_title('PCA Distribution: Test Set vs. Sexed Anchors in it')
    axs_pca[1].legend()

    ax_num = 2 
    ax_hist_num = 0
    for t in plot_tests:
        conditions = t.get('conditions', [])
        values = t.get('values', [])
        description = t.get('description', '')

        mask_anchors_4divisions(X_test, anchors, conditions, values, X_test_pca, clusters, axs_pca[ax_num], axs_hist[ax_hist_num], title=description)

        ax_num += 1
        ax_hist_num += 1

    # Hide the last subplot if we have an odd number of plots
    for j in range(num_plots+2, len(axs_pca)):
        axs_pca[j].axis('off')
    
    for j in range(num_plots, len(axs_hist)):
        axs_hist[j].axis('off')

    fig_pca.tight_layout()
    fig_pca.subplots_adjust(top=0.9)

    fig_hist.tight_layout()
    fig_hist.subplots_adjust(top=0.9)
    
    if save:
        dir = './plots/clustered_pca'
        os.makedirs(dir, exist_ok=True)
        save_path = os.path.join(dir, f'clusters_{file_name}.png')
        fig_pca.savefig(save_path)

        dir = './plots/anchors_distribution'
        os.makedirs(dir, exist_ok=True)
        save_path = os.path.join(dir, f'pca_{file_name}.png')
        fig_hist.savefig(save_path)

        plt.close(fig_pca)
        plt.close(fig_hist)
    else:
        fig_pca.show()

        fig_hist.show()
        

loader = DataLoader()

trainer = DataTrainer(loader)

file_path = 'xai/output/json/'

anchors_tests = {
    'files': [
        'skrub_tx_anchors_explanations.json',
        'skrub_ca_anchors_explanations.json',
        'skrub_ny_anchors_explanations.json',
        'xg_tx_anchors_explanations.json',
        'xg_ca_anchors_explanations.json',
        'xg_ny_anchors_explanations.json'
    ],
    'tests': [
        {'conditions': ['prediction'], 'values': [], 'description': 'Prediction = True/False'},
        {'conditions': ['prediction', 'precision'], 'values': [0.95], 'description': 'Prediction = True/False; Precision > 0.95'},
        {'conditions': ['prediction', 'precision', 'num_features'], 'values': [0.95, 3], 'description': 'Prediction = True/False; Precision > 0.95;\n Num_features <= 3'},
        {'conditions': ['prediction', 'precision', 'coverage'], 'values': [0.95, 0.1], 'description': 'Prediction = True/False; Precision > 0.95;\n Coverage > 0.1'}
    ]
}

shap_tests = {
    'files': [
        'skrub_tx_shap_explanations.json',
        'skrub_ca_shap_explanations.json',
        'skrub_ny_shap_explanations.json',
        'xg_tx_shap_explanations.json',
        'xg_ca_shap_explanations.json',
        'xg_ny_shap_explanations.json'
    ],
    'tests': [
        {'conditions': ['prediction'], 'values': [], 'description': 'Prediction = True/False'},
        {'conditions': ['prediction', 'precision'], 'values': [0.95], 'description': 'Prediction = True/False; Precision > 0.95'},
        {'conditions': ['prediction', 'precision', 'rank_position'], 'values': [0.95, 3], 'description': 'Prediction = True/False; Precision > 0.95;\n Rank position <= 3'}
    ]
}

for f in anchors_tests.get('files'):
    state = re.search(r'_([a-zA-Z]{2})_', f)
    f_name = re.sub(r'_explanations\.json$', '', f)
    analyse_clustered_pca(file_path+f, loader, anchors_tests.get('tests'), state[1].upper(), save=True, file_name=f_name)

for f in shap_tests.get('files'):
    state = re.search(r'_([a-zA-Z]{2})_', f)
    f_name = re.sub(r'_explanations\.json$', '', f)
    analyse_clustered_pca(file_path+f, loader, shap_tests.get('tests'), state[1].upper(), save=True, file_name=f_name)
