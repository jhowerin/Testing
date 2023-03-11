import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection

def train_test_split(dataset: pd.DataFrame, target_col: str, test_size: float, stratify: bool, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Split dataset into features and targets
    features = dataset.drop(columns=target_col)
    targets = dataset[target_col]

    # Split features and targets into training and testing sets
    if stratify:
        train_features, test_features, train_targets, test_targets = \
            sklearn.model_selection.train_test_split(features, targets, test_size=test_size, random_state=random_state, stratify=targets)
    else:
        train_features, test_features, train_targets, test_targets = \
            sklearn.model_selection.train_test_split(features, targets, test_size=test_size, random_state=random_state)

    return train_features, test_features, train_targets, test_targets

class PreprocessDataset:
    def __init__(self, 
                 train_features:pd.DataFrame, 
                 test_features:pd.DataFrame,
                 one_hot_encode_cols:list[str],
                 min_max_scale_cols:list[str],
                 n_components:int,
                 feature_engineering_functions:dict
                 ):
        # Save the input variables as attributes of the class
        self.train_features = train_features
        self.test_features = test_features
        self.one_hot_encode_cols = one_hot_encode_cols
        self.min_max_scale_cols = min_max_scale_cols
        self.n_components = n_components
        self.feature_engineering_functions = feature_engineering_functions

    def one_hot_encode_columns_train(self) -> pd.DataFrame:
        # Use the one-hot encoder to encode categorical columns in train data
        enc = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', drop='first')
        enc.fit(self.train_features[self.one_hot_encode_cols])
        one_hot_encoded = pd.DataFrame(enc.transform(self.train_features[self.one_hot_encode_cols]).toarray(),
                                    columns=enc.get_feature_names(self.one_hot_encode_cols))
        
        # Encode unknown categories with zeroes
        unknown_categories = set(self.train_features[self.one_hot_encode_cols]) - set(enc.categories_[0])
        unknown_columns = pd.DataFrame(np.zeros((len(self.train_features), len(unknown_categories))), columns=list(unknown_categories))
        train_num_cols = self.train_features.drop(self.one_hot_encode_cols, axis=1)
        train_processed = pd.concat([train_num_cols.reset_index(drop=True), one_hot_encoded.reset_index(drop=True), unknown_columns], axis=1)

        return train_processed

    def one_hot_encode_columns_test(self) -> pd.DataFrame:
        # Use the one-hot encoder to encode categorical columns in test data
        enc = sklearn.preprocessing.OneHotEncoder(drop='first', handle_unknown='ignore')
        enc.fit(self.train_features[self.one_hot_encode_cols])
        test_encoded = enc.transform(self.test_features[self.one_hot_encode_cols])
        
        # Create an empty dataframe with the column names from the fitted encoder
        test_encoded_df = pd.DataFrame(columns=enc.get_feature_names(self.one_hot_encode_cols))
        
        # Check if any unknown categories were found during transform and add new columns with all zeroes if necessary
        if enc.categories_:
            for i, cat in enumerate(enc.categories_):
                unknown_mask = ~(self.test_features[self.one_hot_encode_cols].isin(cat).any(axis=1))
                if unknown_mask.any():
                    unknown_df = pd.DataFrame(0, index=self.test_features[unknown_mask].index, columns=[f'{col}_{j}' for j in cat[1:]])
                    test_encoded = sp.sparse.hstack([test_encoded, unknown_df])
        
        # Concatenate the encoded dataframe with the numerical columns of test data
        test_num_cols = self.test_features.drop(self.one_hot_encode_cols, axis=1)
        test_processed = pd.concat([test_num_cols.reset_index(drop=True), pd.DataFrame(test_encoded.toarray())], axis=1)

        return test_processed


    
    def min_max_scaled_columns_train(self) -> pd.DataFrame:
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaled_features = scaler.fit_transform(self.train_features[self.min_max_scale_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=self.min_max_scale_cols)
        self.train_features = pd.concat([self.train_features.drop(self.min_max_scale_cols, axis=1), scaled_df], axis=1)
        return self.train_features

    def min_max_scaled_columns_test(self) -> pd.DataFrame:
        scaler = sklearn.preprocessing.MinMaxScaler()
        scaled_features = scaler.fit_transform(self.test_features[self.min_max_scale_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=self.min_max_scale_cols)
        self.test_features = pd.concat([self.test_features.drop(self.min_max_scale_cols, axis=1), scaled_df], axis=1)
        return self.test_features

    def pca_train(self) -> pd.DataFrame:
        # Perform PCA on train features to reduce the dimensionality
        pca = sklearn.decomposition.PCA(n_components='mle')
        pca.fit(self.train_features[self.min_max_scale_cols])
        train_pca = pca.transform(self.train_features[self.min_max_scale_cols])
        train_pca_df = pd.DataFrame(data=train_pca, columns=[f'component_{i+1}' for i in range(pca.n_components_)])
        return train_pca_df

    def pca_test(self) -> pd.DataFrame:
        # Perform PCA on test features to reduce the dimensionality
        pca = sklearn.decomposition.PCA(n_components='mle')
        pca.fit(self.test_features[self.min_max_scale_cols])
        test_pca = pca.transform(self.test_features[self.min_max_scale_cols])
        test_pca_df = pd.DataFrame(data=test_pca, columns=[f'component_{i+1}' for i in range(pca.n_components_)])
        return test_pca_df

    def feature_engineering_train(self) -> pd.DataFrame:
        train_features_engineered = self.train_features.copy()
        for feature_name, feature_func in self.feature_engineering_functions.items():
            train_features_engineered[feature_name] = feature_func(train_features_engineered)
        return train_features_engineered

    def feature_engineering_test(self) -> pd.DataFrame:
        test_features_engineered = self.test_features.copy()
        for feature_name, feature_func in self.feature_engineering_functions.items():
            test_features_engineered[feature_name] = feature_func(test_features_engineered)
        return test_features_engineered

def preprocess(self) -> tuple[pd.DataFrame,pd.DataFrame]:
        # One-hot encode categorical columns for train and test sets
        train_enc = self.one_hot_encode_columns_train()
        test_enc = self.one_hot_encode_columns_test()
        
        # Min-max scale numerical columns for train and test sets
        train_scaled = self.min_max_scaled_columns_train()
        test_scaled = self.min_max_scaled_columns_test()
        
        # Apply PCA on scaled train and test sets
        train_pca = self.pca_train()
        test_pca = self.pca_test()
        
        # Apply feature engineering functions on train and test sets
        train_eng = self.feature_engineering_train()
        test_eng = self.feature_engineering_test()
        
        # Merge all processed train and test sets
        train = pd.concat([train_enc, train_scaled, train_pca, train_eng], axis=1)
        test = pd.concat([test_enc, test_scaled, test_pca, test_eng], axis=1)
        
        # Split the features and target for train and test sets
        X_train, X_test, y_train, y_test = train_test_split(train, self.target_col, test_size=0.2, stratify=self.stratify, random_state=42)
        X_train.drop(self.target_col, axis=1, inplace=True)
        X_test.drop(self.target_col, axis=1, inplace=True)
        
        return X_train, X_test, y_train, y_test

