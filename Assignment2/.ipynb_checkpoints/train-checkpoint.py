import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(filepath):
    # 1. loadingdata
    df = pd.read_csv(filepath)
    
    # 2. Null data processing
    df['bank_asset_value'] = df['bank_asset_value'].replace(-100000, np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # 3. Feature engineering
    df['education'] = df['education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['self_employed'] = df['self_employed'].map({'Yes': 1, 'No': 0})
    df['debt_to_income'] = df['loan_amount'] / df['income_annum']
    df['total_assets'] = df[['residential_assets_value', 'commercial_assets_value', 
                            'luxury_assets_value', 'bank_asset_value']].sum(axis=1)
    
    # 4. Features
    features = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
               'loan_amount', 'loan_term', 'cibil_score', 'total_assets',
               'debt_to_income']
    target = 'label'
    
    X = df[features]
    y = df[target]
    
    # 5. Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    # 6. Train and verification set
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        
        
        if (depth >= self.max_depth if self.max_depth else False) or \
           (n_labels == 1) or \
           (n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
       
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        
      
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            
            for thresh in thresholds:
                gain = self._information_gain(y, X_column, thresh)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thresh
        
        return split_idx, split_thresh
    
    def _information_gain(self, y, X_column, threshold):
      
        parent_entropy = self._entropy(y)      
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
         
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        ig = parent_entropy - child_entropy
        return ig
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            # bootstrap
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    
    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([self._most_common_label(pred) for pred in tree_preds.T])
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]


import joblib

from sklearn.metrics import f1_score
def train_and_save_model(filepath='train.csv', model_path='random_forest_model.pkl'):
    X_train, X_test, y_train, y_test, scaler = preprocess_data(filepath)
    
    rf = RandomForest(n_trees=100, max_depth=10, min_samples_split=2)
    rf.fit(X_train.values, y_train.values)
    
    #save model
    joblib.dump({'model': rf, 'scaler': scaler}, model_path)
    print(f"Model saved to {model_path}")
    
    y_train_pred = rf.predict(X_train.values)
    y_test_pred = rf.predict(X_test.values)
    
    train_acc = np.mean(y_train_pred == y_train.values)
    test_acc = np.mean(y_test_pred == y_test.values)
    
    train_f1 = f1_score(y_train.values, y_train_pred, average='macro')
    test_f1 = f1_score(y_test.values, y_test_pred, average='macro')
    
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Testing accuracy: {test_acc:.4f}")
    print(f"Training Macro-F1: {train_f1:.4f}")
    print(f"Testing Macro-F1: {test_f1:.4f}")
    
    return rf, scaler

if __name__ == "__main__":
    train_and_save_model()