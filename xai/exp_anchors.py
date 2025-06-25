from anchor import anchor_tabular

# from consensus_module.utils import scale_weights

class ExpAnchors:
    def __init__(self, data_x, ml_model, feature_names):
        self.feature_names = feature_names
        self.ml_model = ml_model
        # Setting up Anchors
        self.anchors = anchor_tabular.AnchorTabularExplainer(
            [0, 1],
            self.feature_names,
            data_x.values,
            {})
    
    def set_model(self, new_model):
        self.ml_model = new_model
        
    def run_anchors(self, row_values, anchors_threshold = 0.95):
        exp_anchors = self.anchors.explain_instance(
            row_values,
            self.ml_model.predict,
            threshold=anchors_threshold)
        return exp_anchors

    # export anchors features' explanations
    def export_anchors_exp(self, row, row_values):
        anchors_exp = self.run_anchors(row_values)
        
        anchors_output = dict()
        # general instance indices
        anchors_output['row_num'] = row.name
        anchors_output['prediction'] = anchors_exp.exp_map['prediction']

        anchors_output['precision'] = anchors_exp.precision()
        anchors_output['coverage'] = anchors_exp.coverage()
        # features' values
        features_exp = []
        features_weights = []
        rank = 1

        anchors_output['array_anchors'] = anchors_exp.exp_map['feature']

        for i in range(0, len(anchors_exp.names())):
            f = dict()
            # extract feature name from anchors' names string
            any((feature_name := substring) in str(anchors_exp.names()[i]) for substring in self.feature_names)
            f['feature_name'] = feature_name
            f['feature_value'] = int(row[feature_name])
            weight = (anchors_exp.precision(i)*anchors_exp.coverage(i))/anchors_exp.precision()
            features_weights.append(weight)
            f['feature_weight'] = None
            f['feature_ranges'] = anchors_exp.names()[i]
            f['feature_rank'] = rank # feature's order of priority in explainer's result
            features_exp.append(f)
            rank += 1
        # scale weights to sum to 1
        # scaled_features_exp = scale_weights(features_weights, features_exp)
        anchors_output['features'] = features_exp

        # anchors_output['anchor_map'] = anchors_exp.exp_map
        return anchors_output