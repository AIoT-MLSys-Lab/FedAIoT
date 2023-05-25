#!/usr/bin/env python3
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Metric, R2Score


def digitize_values(values, a, b, num_bins):
    # Sort the values
    values = np.sort(values)

    # Determine the indices that will divide values into num_bins equal parts
    indices = np.linspace(0, len(values), num_bins + 1, endpoint=False, dtype=int)

    # Create bins using these indices
    bins = [values[indices[i]:indices[i + 1]] for i in range(num_bins)]

    # Now 'bins' is a list of arrays, where each array is a bin
    # containing approximately the same number of samples.

    # If you want to assign each original value to a bin index:
    digitized_values = np.zeros_like(values, dtype=np.int32)
    for i, b in enumerate(bins):
        digitized_values[np.isin(values, b)] = i

    return digitized_values

class HandleOutliers(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        '''
        Description : It notes the 90 and 10 percentile of each features in the dataframe.
                      So that we can impute the outliers with the value of noted percentile.
        Parameters:
            X : Dataframe which you want to note percentile.
            y : It is not required.
        '''
        outlier_estimator_dict = {}
        for col in X.columns:
            upper_bound = np.percentile(X[col], 90)
            lower_bound = np.percentile(X[col], 10)
            outlier_estimator_dict[col] = {
                "upper_bound": upper_bound,
                "lower_bound": lower_bound}
        self.outlier_estimator_dict = outlier_estimator_dict
        return self

    def transform(self, X, y=None):
        '''
        Description : It replaces the outliers with the noted percentile value of respective column
        Parameters:
            X : Dataframe you want to replace outliers.
        Returns :  A Dataframe with removed outliers.
        '''
        for col in X.columns:
            col_dict = self.outlier_estimator_dict[col]
            X[col] = np.where(X[col] > col_dict['upper_bound'], col_dict['upper_bound'], X[col])
            X[col] = np.where(X[col] < col_dict['lower_bound'], col_dict['lower_bound'], X[col])

        self.final_column_names = X.columns
        return X


class AddPcaFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, number_of_pca_columns=None):
        '''
        Parameters :
            number_of_pca_columns :(Int) Number of final dimension you want.
        '''
        self.number_of_pca_columns = number_of_pca_columns
        return None

    def fit(self, X, y=None):
        '''
        Description : It fits the data in the PCA algorithm
        Parameters:
            X : Dataframe which fits the PCA algorithm
        '''
        if self.number_of_pca_columns != None:
            self.pca = PCA(n_components=self.number_of_pca_columns)
            self.pca.fit(X)
        return self

    def transform(self, X, y=None):
        '''
        Parameters :
            X : Dataframe you want to reduce the dimension
        Returns : A Dataframe with the pca features along concatinated with the input Dataframe.
        '''
        if self.number_of_pca_columns != None:
            pca_column_names = [f'pca_{val}' for val in range(1, self.number_of_pca_columns + 1)]
            pca_features = self.pca.transform(X)
            pca_features = pd.DataFrame(pca_features, columns=pca_column_names, index=X.index)
            X = pd.concat([X, pca_features], axis=1)

        return X


class AddCentralTendencyFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, measure):
        '''
        Parameters :
            measure : 'mean' or 'median' depend on which features you want to add.
        '''
        self.measure = measure
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        Description : Adds either mean or median columns of a temperature and humidity column for each observation.
        Parameter : Dataframe which you want to calculate
        Returns : Input Dataframe concatinated with the calculated features.
        '''
        if self.measure.lower() == 'mean':
            X['avg_house_temp'] = X[[col for col in X.columns if (('t' in col) and (len(col) < 3))]].mean(axis=1)
            X['avg_humidity_percentage'] = X[[col for col in X.columns if (('rh_' in col) and (len(col) < 5))]].mean(
                axis=1)

        else:
            X['med_house_temp'] = X[[col for col in X.columns if (('t' in col) and (len(col) < 3))]].median(axis=1)
            X['med_humidity_percentage'] = X[[col for col in X.columns if (('rh_' in col) and (len(col) < 5))]].median(
                axis=1)

        return X


class AddDateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['day'] = X.date.dt.day
        X['month'] = X.date.dt.month
        return X.drop('date', axis=1)


class RemoveCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y):
        '''
        Description : Remove correlated features with less correlation with target
        X : Dataframe with only features
        y : Target Series
        '''
        col_corr = set()
        corr_matrix = X.corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.85:
                    corr_i, _ = pearsonr(y, X.iloc[:, i])
                    corr_j, _ = pearsonr(y, X.iloc[:, j])
                    if abs(corr_i) < abs(corr_j):
                        colname = corr_matrix.columns[i]
                        col_corr.add(colname)
                    else:
                        colname = corr_matrix.columns[i]
                        col_corr.add(colname)

        self.correlated_columns = col_corr
        self.final_column_names = set(X.columns) - self.correlated_columns
        return self

    def transform(self, X, y=None):
        '''
        Parameter : The Dataframe you want to remove correlated features
        Returns : Dataframe by removing the correlated features.
        '''
        return X.drop(self.correlated_columns, axis=1)


class ApplyTransformation(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[['t9', 'rv1', 'rv2', 'windspeed']] = np.log1p(X[['t9', 'rv1', 'rv2', 'windspeed']])
        X['visibility'] = np.where(X['visibility'] > 40, 1, 0)
        return X


class EnergyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.targets = digitize_values(labels, np.min(labels), np.max(labels), 10)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float), torch.tensor(self.labels[idx], dtype=torch.float)


def load_dataset(split=0.2, seed=42):
    df = pd.read_csv('datasets/energy/energydata_complete.csv')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index(df.date.copy(deep=True), inplace=True)

    # Preprocess the data
    # Split the data into features and target
    df.columns = [col.lower() for col in df.columns]
    X = df.drop('appliances', axis=1)
    y = df['appliances']

    # Split the dataset into train and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=seed)

    # Scale the features to have zero mean and unit variance
    preprocessing_pipeline = Pipeline([
        ('transformation', ApplyTransformation()),
        ('remove_outliers', HandleOutliers()),
        ('add_central_tendency_features', AddCentralTendencyFeatures(measure='mean')),
        ('add_Date_Features', AddDateFeatures()),
        ('add_pca_features', AddPcaFeatures(number_of_pca_columns=3)),
        ('remove_correlated_features', RemoveCorrelatedFeatures()),
        ('standard_scalar', StandardScaler())
    ])
    # min_y = min(y_train)
    # max_y = max(y_train)
    # y_train = (y_train - min_y) / max_y
    # y_test = (y_test - min_y) / max_y
    y_train = np.log(y_train)
    y_test = np.log(y_test)
    X_train = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_test = preprocessing_pipeline.transform(X_test)
    train_data = EnergyDataset(X_train, y_train)
    test_data = EnergyDataset(X_test, y_test)
    return {
        'train': train_data,
        'test': test_data,
    }


if __name__ == '__main__':
    dt = load_dataset()
    print(len(dt['train']))
    print(dt['train'][0][0].shape)