# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:16:54 2019

@author: yuanq
"""
import enum

@enum.unique
class ENUM_MACHINE_LEARNING_TYPE(enum.Enum):
    CLASSIFICATION = 1
    CLUSTERING = 2
    REGRESSION = 3
    DIMENSIONALITY_REDUCTION = 4

@enum.unique
class ENUM_CROSS_SECTIONAL_CLASSIFICATION_MODELS(enum.Enum):
    GAUSSIAN_NIAVE_BAYES = 'GaussianNB'
    LOGISTIC = 'LogisticRegression'    
    RANDOM_FOREST = 'RandomForestClassifier'
    KNN = 'KNeighborsClassifier'  
    SGD = 'SGDClassifier'
    SVM = 'SVC'
    MLP = 'MLPClassifier'    
    
@enum.unique
class ENUM_CROSS_SECTIONAL_CLUSTERING_MODELS(enum.Enum):
    K_MEANS = 'KMeans'
    SPECTRAL = 'SpectralClustering'
    HIERARCHICAL = 'AgglomerativeClustering'
    DBSCAN = 'DBSCAN'
    BIRCH = 'Birch'
            
@enum.unique
class ENUM_CROSS_SECTIONAL_REGRESSION_MODELS(enum.Enum):
    LINEAR = 'LinearRegression'
    SGD = 'SGDRegressor'
    LASSO = 'Lasso'
    ELASTIC_NET = 'ElasticNet'
    SUPPORT_VECTOR = 'SVR'
    RIDGE = 'Ridge'

@enum.unique
class ENUM_DATA_SET_TYPE(enum.Enum):
    CROSS_SECTIONAL = 1
    TIME_SERIES = 2
    TEXT = 3
    IMAGE = 4
    VIDEO = 5
    AUDIO = 6
    
@enum.unique
class ENUM_DATA_TYPE(enum.Enum):
    NUMERIC = 1
    CATEGORICAL = 2
    TEXTUAL = 3
    DATE = 4
    

@enum.unique
class ENUM_BAYESIAN_METHOD(enum.Enum):
    GAUSSIAN = 1
    MULTINOMIAL = 2
    COMPLEMENT = 3
    BERNOULLI = 4
    OUT_OF_CORE = 5
    
@enum.unique
class ENUM_MODEL(enum.Enum):
    LOGISTIC_REGRESSION = 1

@enum.unique
class ENUM_CROSS_VALIDATION_METHOD(enum.Enum):
    K_FOLD = 1
    REPEATED_K_FOLD = 2
    LEAVE_ONE_OUT = 3
    LEAVE_P_OUT = 4
    SHUFFLE_SPLIT = 5
    STRATIFIED_K_FOLD = 6
    STRATIFIED_SHUFFLE_SPLIT = 7       