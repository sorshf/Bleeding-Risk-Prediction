#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"""The module contains the 4 train-test experiments (and the dummy classifiers experiments)."""
# =============================================================================
# Imports
from hypermodels import LastFUPHyperModel, BaselineHyperModel, FUP_RNN_HyperModel, Baseline_FUP_Multiinput_HyperModel, dummy_classifiers
from constants import instruction_dir
import keras_tuner
import pandas as pd
from cross_validation import divide_into_stratified_fractions, get_X_y_from_indeces, normalize_training_validation
from sklearn.feature_selection import GenericUnivariateSelect, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
import numpy as np
from constants import timeseries_padding_value
from generate_stats import get_counts_all_zero_or_one_stats
import copy
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
import matplotlib
import os
matplotlib.use("Agg")

def get_predefined_training_testing_indeces_30_percent():
    """Gets the predefined training-val (70% of data) and testing (30% of data) indeces.

    Returns:
        list, list: training_val_indeces, testing_indeces
    """
    predefined_indeces = pd.read_csv("./keras_tuner_results/training_testing_indeces.csv")
    
    for num_FUP in set(predefined_indeces["num_FUP"]):
        total_FUP_positive = len(predefined_indeces[(predefined_indeces["num_FUP"]==num_FUP) & (predefined_indeces["target"] ==1)])
        length_testing = len(predefined_indeces[(predefined_indeces["category"]=="testing")])
        length_training_val = len(predefined_indeces[ (predefined_indeces["category"]=="training_val")])
        
        percent_positive_testing = len(predefined_indeces[(predefined_indeces["category"]=="testing") & 
                                                        (predefined_indeces["target"]==1) &
                                                        (predefined_indeces["num_FUP"]==num_FUP)])/ length_testing * 100
        
        percent_negative_testing = len(predefined_indeces[(predefined_indeces["category"]=="testing") & 
                                                        (predefined_indeces["target"]==0) &
                                                        (predefined_indeces["num_FUP"]==num_FUP)])/ length_testing * 100
        
        percent_positive_training_val = len(predefined_indeces[(predefined_indeces["category"]=="training_val") & 
                                                        (predefined_indeces["target"]==1) &
                                                        (predefined_indeces["num_FUP"]==num_FUP)])/ length_training_val * 100
        
        percent_negative_training_val = len(predefined_indeces[(predefined_indeces["category"]=="training_val") & 
                                                        (predefined_indeces["target"]==0) &
                                                        (predefined_indeces["num_FUP"]==num_FUP)])/ length_training_val * 100
        
        print(f"{total_FUP_positive} -> {num_FUP} FUPS: testing +: {percent_positive_testing:.5f} training_val +: {percent_positive_training_val:.5f} \
        testing -: {percent_negative_testing:.5f} training_val -: {percent_negative_training_val:.5f}")
    
    
    
    training_val_indeces = list(predefined_indeces.loc[predefined_indeces["category"]=="training_val", "uniqids"])
    testing_indeces = list(predefined_indeces.loc[predefined_indeces["category"]=="testing", "uniqids"])
    
    return training_val_indeces, testing_indeces


def get_important_feature_dic_baseline(baseline_dataset, target_list):
    #This function is for feature selection
    def get_important_features(X, y, method, number_features):
        transformer = GenericUnivariateSelect(method, mode='k_best', param=number_features)
        X_new = transformer.fit(X, y)
        return X_new.get_feature_names_out()
    
    
    features_dict = dict()
    methods_list = [f_classif, mutual_info_classif]
    numbers_list = [1, 3, 5, 10, 15, 20]

    for method in methods_list:
        for number in numbers_list:
            features_dict[f"{number}_{method.__name__}"] = get_important_features(baseline_dataset, 
                                                                        target_list, 
                                                                        method, 
                                                                        number)

    #features_dict["total"] = baseline_dataset.columns
    
    return features_dict

def create_feature_set_dicts_baseline_and_FUP(baseline_dataframe, FUP_columns, target_series, patient_dataset, mode):
    """Perform feature selection for the baseline data and FUP, then returns an appropriate dict for hyperparameter tuning.

    Args:
        baseline_dataframe (pd.dataframe): The baseline data (that come from training-val data) SHOULD NOT be standardized.
        FUP_columns (list): List of FUP feature names.
        target_series (pd.series): Target varaibles for the patients.
        patient_dataset (Dataset): Patient datset that will be used for FUP feature selection.
        mode (string): Whether we want "only FUP", "only Baseline", or "Both FUP and Baseline".

    Returns:
        dict: Dicts with keys "baseline_feature_sets" and "FUPS_feature_sets" containing the features sets for baseline and FUP.
    """
    if mode in ["only Baseline", "Both FUP and Baseline"]:
        #Standardize the baseline dataframe
        baseline_dataframe_standardized = pd.DataFrame(StandardScaler().fit_transform(baseline_dataframe.copy()), columns = baseline_dataframe.columns)

        #Perform feature selection with different number of features
        baseline_feature_sets_dict = get_important_feature_dic_baseline(baseline_dataframe_standardized, target_series)

        #Convert the names to index for the values in the baseline_feature_sets_dict
        indexed_feature_set_dict = dict()
        all_cols = list(baseline_dataframe.columns)

        for feature_set in baseline_feature_sets_dict:
            indexed_feature = [all_cols.index(feature) for feature in baseline_feature_sets_dict[feature_set]]
            indexed_feature_set_dict[feature_set] = indexed_feature
        
    ####################
    ###################

    if mode in ["only FUP", "Both FUP and Baseline"]:
        #sets of features in FUP data

        FUP_features_sets = dict()

        FUP_features_sets["all_features"] = list(range(len(FUP_columns)))

        features_without_new_no_continue_yes = [feature for feature in FUP_columns if (feature.find("(Yes)")==-1) & (feature.find("(No)")==-1) & \
                                                (feature.find("(Continue)")==-1) & (feature.find("(New)")==-1)]


        FUP_features_sets["FUP_without_new_no_continue_yes"] = [FUP_columns.index(feature) for feature in features_without_new_no_continue_yes]
        
        
        stat_df = get_counts_all_zero_or_one_stats(patient_dataset, instruction_dir)
        stat_df = stat_df[stat_df["CRF"].isin(["FUPPREDICTOR", "FUPOUTCOME"])]
        high_varience_features = stat_df[(stat_df["percent_patients_all_zero"]<96) & (stat_df["percent_patients_all_one"]<96)]["feature"]
        
        #Note that get_counts_all_zero_or_one_stats returns the feature names without their name changes. So we use the following trick.
        FUP_features_sets["high_varience_features"] = [list(patient_dataset.all_patients[10].get_FUP_array().columns).index(feature) for feature in high_varience_features]
        
        
    if mode == "only Baseline":
        return {"baseline_feature_sets": indexed_feature_set_dict}
    elif mode == "only FUP":
        return {"FUPS_feature_sets":FUP_features_sets}
    elif mode == "Both FUP and Baseline":
        return {"baseline_feature_sets": indexed_feature_set_dict, "FUPS_feature_sets":FUP_features_sets}
    else:
        raise Exception(f"The mode {mode} isn't recognized.")



def run_baseline_dense_experiment(model_name, 
                                 directory_name, 
                                 metric_name, 
                                 metric_mode, 
                                 metric_cv_calc_mode, 
                                 baseline_dataframe,
                                 FUPS_dict,
                                 target_series,
                                 list_FUP_cols,
                                 patient_dataset,
                                 overwrite=False,
                                 ):
    
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        directory_name (string): The name of the directory to save the models info.
        metric_name (string): The name of the metric to optimize in hypermodel.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
        list_FUP_cols (list): The list of the feature names (in order) for the FUP columns.
        patient_dataset (Dataset): All patient dataset object (this is only used for FUP feature selection).
        overwrite (bool): Whether to overwrite the keras-tuner results directory.
    """
    
    #Create a dir to save the result of experiment
    if not os.path.exists(f"./keras_tuner_results/{model_name}"):
        os.makedirs(f"./keras_tuner_results/{model_name}")
    
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()
    
    #Record the training_val and testing indeces
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)

    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)
    
    #Create the feature selection dic (with different methods) for hyperparamter tuning
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset, mode="only Baseline")
    
    #Define the tuner
    tuner = keras_tuner.RandomSearch(
                    BaselineHyperModel(name = model_name),
                    objective=keras_tuner.Objective(f"{metric_cv_calc_mode}_val_{metric_name}", metric_mode),
                    max_trials=5000,
                    seed=1375,
                    overwrite = overwrite,
                    directory=directory_name,
                    project_name=model_name)

    #Perform the search
    tuner.search(
                x = [baseline_train_val_X, fups_train_val_X], 
                y = train_val_y,
                batch_size = 16,
                grid_search = True,
                epochs = 100,
                metric_name = metric_name,
                metric_mode = metric_mode,
                verbose = 0,
                feature_set_dict=feature_selection_dict
                )
    
    #Final training on the training-val 
    best_hp = tuner.get_best_hyperparameters()[0] #Get the best hyperparameters
    print("\n",best_hp.values)
    
    #A dict to store hp values
    best_hp_values_dict = {hp:value for hp, value in best_hp.values.items()}
    
    #Calculate the best number of epochs to train the data
    best_number_of_epochs = get_best_number_of_training_epochs(tuner, 
                                                               metric_name=f"val_{metric_name}", 
                                                               metric_mode=metric_mode, 
                                                               metric_cv_calc_mode=metric_cv_calc_mode)
    
    #Save the best hps and best number of epochs in a csv.
    best_hp_values_dict["best_number_of_epochs"] = best_number_of_epochs
    pd.DataFrame.from_dict(best_hp_values_dict, orient="index", columns=["best_hp_value"]).to_csv(f"{directory_name}/{model_name}_best_hp.csv")


    #################
    #####Training on the entire train_val dataset
    #################
    
    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, norm_train_val_fups_X, train_val_y = norm_train_val_data
    norm_test_baseline_X, _, test_y = norm_test_data

    #Train the model on the entire train_val dataset with best hyperparameters for best # of epochs
    hypermodel = BaselineHyperModel(name = model_name)
    model_keras_tuner = tuner.hypermodel.build(best_hp)
    history, model_keras_tuner = hypermodel.fit(best_hp, model_keras_tuner, 
                                    [norm_train_val_baseline_X, norm_train_val_fups_X], 
                                    train_val_y, 
                                    grid_search=False, 
                                    batch_size = 16,
                                    epochs=int(best_number_of_epochs),
                                    feature_set_dict=feature_selection_dict)
    
    #Save the trained model
    model_keras_tuner.save(f"{directory_name}/{model_name}.h5")
    
    #Load the saved model
    model = keras.models.load_model(f"{directory_name}/{model_name}.h5")
    
    #Only keep the best features (as determined by best_hp)
    #Note that we didn't need to do this when calling fit, because it is done internally in the class.
    #However, here it should be done before testing the dataset.
    print(f"Using the feature set {best_hp.values['baseline_feature_set']} for testing.")
    norm_test_baseline_X = norm_test_baseline_X.iloc[:,feature_selection_dict["baseline_feature_sets"][best_hp.values["baseline_feature_set"]]]
    norm_train_val_baseline_X = norm_train_val_baseline_X.iloc[:,feature_selection_dict["baseline_feature_sets"][best_hp.values["baseline_feature_set"]]]
    
    #Test on the testing set
    test_res = model.evaluate(norm_test_baseline_X, test_y)
    pd.DataFrame.from_dict({metric:value for metric, value in zip(model_keras_tuner.metrics_names, test_res)}, orient="index", columns=["test"]).to_csv(f"{directory_name}/{model_name}_test_results.csv")

    #Test on the testing set and training_val set
    #Note, we are testing on the test set with the same model twice (previous 2 lines) which is just for debugging.
    save_metrics_and_ROC_PR(model_name=model_name, 
                            model=model, 
                            training_x=norm_train_val_baseline_X, 
                            training_y=train_val_y, 
                            testing_x = norm_test_baseline_X, 
                            testing_y = test_y,
                            FUPS_dict = FUPS_dict)

def run_lastFUP_dense_experiment(model_name, 
                                 directory_name, 
                                 metric_name, 
                                 metric_mode, 
                                 metric_cv_calc_mode, 
                                 baseline_dataframe,
                                 FUPS_dict,
                                 target_series,
                                 list_FUP_cols,
                                 patient_dataset,                                 
                                 overwrite=False
                                 ):
    
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        directory_name (string): The name of the directory to save the models info.
        metric_name (string): The name of the metric to optimize in hypermodel.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
        list_FUP_cols (list): The list of the feature names (in order) for the FUP columns.
        patient_dataset (Dataset): All patient datset object (this is only used for FUP feature selection).        
        overwrite (bool): Whether to overwrite the keras-tuner results directory.
    """
    
    #Create a dir to save the result of experiment
    if not os.path.exists(f"./keras_tuner_results/{model_name}"):
        os.makedirs(f"./keras_tuner_results/{model_name}")
        
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()

    #record the indeces of the train_val and test dataset
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)
    
    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    patient_dataset_train_val = copy.deepcopy(patient_dataset)    #Keep only the train_val data for feature selection of the FUP data
    patient_dataset_train_val.all_patients = [patient for patient in patient_dataset_train_val.all_patients if patient.uniqid in training_val_indeces]
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset_train_val, mode="only FUP")
    
    #Define the tuner
    tuner = keras_tuner.RandomSearch(
                    LastFUPHyperModel(name = model_name),
                    objective=keras_tuner.Objective(f"{metric_cv_calc_mode}_val_{metric_name}", metric_mode),
                    max_trials=5000,
                    seed=1375,
                    overwrite = overwrite,
                    directory=directory_name,
                    project_name=model_name)

    #Perform the search
    tuner.search(
                x = [baseline_train_val_X, fups_train_val_X], 
                y = train_val_y,
                batch_size = 16,
                grid_search = True,
                epochs = 50,
                metric_name = metric_name,
                metric_mode = metric_mode,
                verbose = 0,
                feature_set_dict=feature_selection_dict
                )
    
    #Final training on the training-val 
    best_hp = tuner.get_best_hyperparameters()[0] #Get the best hyperparameters
    print("\n",best_hp.values)
    
    best_hp_values_dict = {hp:value for hp, value in best_hp.values.items()}
    #Calculate the best number of epochs to train the data
    best_number_of_epochs = get_best_number_of_training_epochs(tuner, 
                                                               metric_name=f"val_{metric_name}", 
                                                               metric_mode=metric_mode, 
                                                               metric_cv_calc_mode=metric_cv_calc_mode)
    
    #Save the best hps and best number of epochs in a csv.
    best_hp_values_dict["best_number_of_epochs"] = best_number_of_epochs
    pd.DataFrame.from_dict(best_hp_values_dict, orient="index", columns=["best_hp_value"]).to_csv(f"{directory_name}/{model_name}_best_hp.csv")

    #################
    #####Training on the entire train_val dataset using best hps
    #################
    
    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, norm_train_val_fups_X, train_val_y = norm_train_val_data
    _                        , norm_test_fups_X, test_y = norm_test_data

    #Train the model on the entire train_val dataset with best hyperparameters for best # of epochs
    hypermodel = LastFUPHyperModel(name = model_name)
    model_keras_tuner = tuner.hypermodel.build(best_hp)
    history, model_keras_tuner = hypermodel.fit(best_hp, model_keras_tuner, 
                                    [norm_train_val_baseline_X, norm_train_val_fups_X], 
                                    train_val_y, 
                                    grid_search=False, 
                                    batch_size = 16,
                                    epochs=int(best_number_of_epochs),
                                    feature_set_dict=feature_selection_dict)
    
    #Save the trained model
    model_keras_tuner.save(f"{directory_name}/{model_name}.h5")
    
    #Load the saved model
    model = keras.models.load_model(f"{directory_name}/{model_name}.h5")
    
    #Extract the last FUP for each patient in the test set.
    #Note that we didn't need to do this when calling fit, because it is done internally.
    #However, here it is not done internally.
    print(f"Using the feature set {best_hp.values['FUPS_feature_set']} for testing.")
    norm_test_fups_X_last = get_last_FUPs_array(norm_test_fups_X, timeseries_padding_value)
    norm_test_fups_X_last = norm_test_fups_X_last[:,feature_selection_dict["FUPS_feature_sets"][best_hp.values["FUPS_feature_set"]]]
    
    #Do the same for training_val and testing data
    norm_training_val_fups_X_last = get_last_FUPs_array(norm_train_val_fups_X, timeseries_padding_value)
    norm_training_val_fups_X_last = norm_training_val_fups_X_last[:,feature_selection_dict["FUPS_feature_sets"][best_hp.values["FUPS_feature_set"]]]
    
    #Test on the testing set
    test_res = model.evaluate(norm_test_fups_X_last, test_y)
    pd.DataFrame.from_dict({metric:value for metric, value in zip(model_keras_tuner.metrics_names, test_res)}, orient="index", columns=["test"]).to_csv(f"{directory_name}/{model_name}_test_result.csv")

    save_metrics_and_ROC_PR(model_name=model_name, 
                            model=model, 
                            training_x=norm_training_val_fups_X_last, 
                            training_y=train_val_y, 
                            testing_x = norm_test_fups_X_last, 
                            testing_y = test_y,
                            FUPS_dict = FUPS_dict)

def run_FUP_RNN_experiment(model_name, 
                                 directory_name, 
                                 metric_name, 
                                 metric_mode, 
                                 metric_cv_calc_mode, 
                                 baseline_dataframe,
                                 FUPS_dict,
                                 target_series,
                                 list_FUP_cols,
                                 patient_dataset,                                                                  
                                 overwrite=False
                                 ):
    
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        directory_name (string): The name of the directory to save the models info.
        metric_name (string): The name of the metric to optimize in hypermodel.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
        list_FUP_cols (list): The list of the feature names (in order) for the FUP columns.
        patient_dataset (Dataset): All patient datset object (this is only used for FUP feature selection).        
        overwrite (bool): Whether to overwrite the keras-tuner results directory.
    """
    #Create a dir to save the result of experiment
    if not os.path.exists(f"./keras_tuner_results/{model_name}"):
        os.makedirs(f"./keras_tuner_results/{model_name}")
        
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()

    #Record the training_val and testing indeces
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)
    
    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    patient_dataset_train_val = copy.deepcopy(patient_dataset)    #Keep only the train_val data for feature selection of the FUP data
    patient_dataset_train_val.all_patients = [patient for patient in patient_dataset_train_val.all_patients if patient.uniqid in training_val_indeces]
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset_train_val, mode="only FUP")
    
    #Define the tuner
    tuner = keras_tuner.RandomSearch(
                    FUP_RNN_HyperModel(name = model_name),
                    objective=keras_tuner.Objective(f"{metric_cv_calc_mode}_val_{metric_name}", metric_mode),
                    max_trials=1000,
                    seed=1375,
                    overwrite = overwrite,
                    directory=directory_name,
                    project_name=model_name)

    #Perform the search
    tuner.search(
                x = [baseline_train_val_X, fups_train_val_X], 
                y = train_val_y,
                batch_size = 16,
                grid_search = True,
                epochs = 50,
                metric_name = metric_name,
                metric_mode = metric_mode,
                verbose = 0,
                feature_set_dict=feature_selection_dict
                )
    
    #Final training on the training-val 
    best_hp = tuner.get_best_hyperparameters()[0] #Get the best hyperparameters
    print("\n",best_hp.values, "\n")
    
    best_hp_values_dict = {hp:value for hp, value in best_hp.values.items()}
    #Calculate the best number of epochs to train the data
    best_number_of_epochs = get_best_number_of_training_epochs(tuner, 
                                                               metric_name=f"val_{metric_name}", 
                                                               metric_mode=metric_mode, 
                                                               metric_cv_calc_mode=metric_cv_calc_mode)
    
    #Save the best hps and best number of epochs in a csv.
    best_hp_values_dict["best_number_of_epochs"] = best_number_of_epochs
    pd.DataFrame.from_dict(best_hp_values_dict, orient="index", columns=["best_hp_value"]).to_csv(f"{directory_name}/{model_name}_best_hp.csv")

    #################
    #####Training on the entire train_val dataset using best hps
    #################
    
    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, norm_train_val_fups_X, train_val_y = norm_train_val_data
    _                        , norm_test_fups_X, test_y = norm_test_data

    #Train the model on the entire train_val dataset with best hyperparameters for best # of epochs
    hypermodel = FUP_RNN_HyperModel(name = model_name)
    model_keras_tuner = tuner.hypermodel.build(best_hp)
    history, model_keras_tuner = hypermodel.fit(best_hp, model_keras_tuner, 
                                    [norm_train_val_baseline_X, norm_train_val_fups_X], 
                                    train_val_y, 
                                    grid_search=False, 
                                    batch_size = 16,
                                    epochs=int(best_number_of_epochs),
                                    feature_set_dict=feature_selection_dict)
    
    #Save the trained model
    model_keras_tuner.save(f"{directory_name}/{model_name}.h5")
    
    #Load the saved model
    model = keras.models.load_model(f"{directory_name}/{model_name}.h5")
    
    #Extract the last FUP for each patient in the test set.
    #Note that we didn't need to do this when calling fit, because it is done internally.
    #However, here it is not done internally.
    print(f"Using the feature set {best_hp.values['FUPS_feature_set']} for testing.")
    norm_test_fups_X = norm_test_fups_X[:,:,feature_selection_dict["FUPS_feature_sets"][best_hp.values["FUPS_feature_set"]]]
    
    #Prepare the training-val dataset for the model (like the test data)
    norm_train_val_fups_X = norm_train_val_fups_X[:,:,feature_selection_dict["FUPS_feature_sets"][best_hp.values["FUPS_feature_set"]]]    
    
    #Test on the testing set
    test_res = model.evaluate(norm_test_fups_X, test_y)
    pd.DataFrame.from_dict({metric:value for metric, value in zip(model_keras_tuner.metrics_names, test_res)}, orient="index", columns=["test"]).to_csv(f"{directory_name}/{model_name}_test_result.csv")

    save_metrics_and_ROC_PR(model_name=model_name, 
                            model=model, 
                            training_x=norm_train_val_fups_X, 
                            training_y=train_val_y, 
                            testing_x = norm_test_fups_X, 
                            testing_y = test_y,
                            FUPS_dict = FUPS_dict)

def run_Baseline_FUP_multiinput_experiment(model_name, 
                                 directory_name, 
                                 metric_name, 
                                 metric_mode, 
                                 metric_cv_calc_mode, 
                                 baseline_dataframe,
                                 FUPS_dict,
                                 target_series,
                                 list_FUP_cols,
                                 patient_dataset,                                                                  
                                 overwrite=False
                                 ):
    
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        directory_name (string): The name of the directory to save the models info.
        metric_name (string): The name of the metric to optimize in hypermodel.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
        list_FUP_cols (list): The list of the feature names (in order) for the FUP columns.
        patient_dataset (Dataset): All patient datset object (this is only used for FUP feature selection).        
        overwrite (bool): Whether to overwrite the keras-tuner results directory.
    """
    #Create a dir to save the result of experiment
    if not os.path.exists(f"./keras_tuner_results/{model_name}"):
        os.makedirs(f"./keras_tuner_results/{model_name}")
        
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()

    #Record the training and testing indeces
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)
    
    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    patient_dataset_train_val = copy.deepcopy(patient_dataset)    #Keep only the train_val data for feature selection of the FUP data
    patient_dataset_train_val.all_patients = [patient for patient in patient_dataset_train_val.all_patients if patient.uniqid in training_val_indeces]
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset_train_val, mode="Both FUP and Baseline")
    
    feature_selection_dict = {"baseline_feature_sets":{'15_f_classif':feature_selection_dict["baseline_feature_sets"]['15_f_classif'],
                                                            '20_f_classif':feature_selection_dict["baseline_feature_sets"]['20_f_classif']}, 
                                    "FUPS_feature_sets":{'all_features':feature_selection_dict["FUPS_feature_sets"]['all_features']}}

    
    #Define the tuner
    tuner = keras_tuner.RandomSearch(
                    Baseline_FUP_Multiinput_HyperModel(name = model_name),
                    objective=keras_tuner.Objective(f"{metric_cv_calc_mode}_val_{metric_name}", metric_mode),
                    max_trials=700,
                    seed=1375,
                    overwrite = overwrite,
                    directory=directory_name,
                    project_name=model_name)

    #Perform the search
    tuner.search(
                x = [baseline_train_val_X, fups_train_val_X], 
                y = train_val_y,
                batch_size = 16,
                grid_search = True,
                epochs = 100,
                metric_name = metric_name,
                metric_mode = metric_mode,
                verbose = 0,
                feature_set_dict=feature_selection_dict
                )
    
    #Final training on the training-val 
    best_hp = tuner.get_best_hyperparameters()[0] #Get the best hyperparameters
    print("\n",best_hp.values)
    
    best_hp_values_dict = {hp:value for hp, value in best_hp.values.items()}
    #Calculate the best number of epochs to train the data
    best_number_of_epochs = get_best_number_of_training_epochs(tuner, 
                                                               metric_name=f"val_{metric_name}", 
                                                               metric_mode=metric_mode, 
                                                               metric_cv_calc_mode=metric_cv_calc_mode)
    
    #Save the best hps and best number of epochs in a csv.
    best_hp_values_dict["best_number_of_epochs"] = best_number_of_epochs
    pd.DataFrame.from_dict(best_hp_values_dict, orient="index", columns=["best_hp_value"]).to_csv(f"{directory_name}/{model_name}_best_hp.csv")

    #################
    #####Training on the entire train_val dataset using best hps
    #################
    
    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, norm_train_val_fups_X, train_val_y = norm_train_val_data
    norm_test_baseline_X, norm_test_fups_X, test_y = norm_test_data

    #Train the model on the entire train_val dataset with best hyperparameters for best # of epochs
    hypermodel = Baseline_FUP_Multiinput_HyperModel(name = model_name)
    model_keras_tuner = tuner.hypermodel.build(best_hp)
    history, model_keras_tuner = hypermodel.fit(best_hp, model_keras_tuner, 
                                    [norm_train_val_baseline_X, norm_train_val_fups_X], 
                                    train_val_y, 
                                    grid_search=False, 
                                    batch_size = 16,
                                    epochs=int(best_number_of_epochs),
                                    feature_set_dict=feature_selection_dict)
    
    #Save the trained model
    model_keras_tuner.save(f"{directory_name}/{model_name}.h5")
    
    #Load the saved model
    model = keras.models.load_model(f"{directory_name}/{model_name}.h5")
    
    #"baseline_feature_sets" and "FUPS_feature_sets"
    #The choice of the feature set to use for the baseline dataset
    baseline_feature_choice = best_hp.values["baseline_feature_set"]
    features_index_to_keep_baseline = feature_selection_dict["baseline_feature_sets"][baseline_feature_choice]
    norm_train_val_baseline_X = norm_train_val_baseline_X.iloc[:,features_index_to_keep_baseline]
    norm_test_baseline_X = norm_test_baseline_X.iloc[:,features_index_to_keep_baseline]
    
    #The choice of the feature set to use for FUP data
    fups_feature_choice = best_hp.values["FUPS_feature_set"]
    features_index_to_keep_fups = feature_selection_dict["FUPS_feature_sets"][fups_feature_choice]
    norm_test_fups_X = norm_test_fups_X[:,:,features_index_to_keep_fups]
    norm_train_val_fups_X = norm_train_val_fups_X[:,:,features_index_to_keep_fups]
    
    #Note that we didn't need to do this when calling fit, because it is done internally.
    #However, here it is not done internally.
    print(f"Using the baseline feature set {baseline_feature_choice} and FUP feature set {fups_feature_choice} for testing.")

    
    #Test on the testing set
    test_res = model.evaluate([norm_test_baseline_X, norm_test_fups_X], test_y)
    pd.DataFrame.from_dict({metric:value for metric, value in zip(model_keras_tuner.metrics_names, test_res)}, orient="index", columns=["test"]).to_csv(f"{directory_name}/{model_name}_test_result.csv")

    save_metrics_and_ROC_PR(model_name=model_name, 
                            model=model, 
                            training_x=[norm_train_val_baseline_X, norm_train_val_fups_X], 
                            training_y=train_val_y, 
                            testing_x = [norm_test_baseline_X, norm_test_fups_X], 
                            testing_y = test_y,
                            FUPS_dict = FUPS_dict)

def run_dummy_experiment(model_name, 
                        baseline_dataframe,
                        FUPS_dict,
                        target_series,
                        ):
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
    """
    #Create a dir to save the result of experiment
    if not os.path.exists(f"./keras_tuner_results/{model_name}"):
        os.makedirs(f"./keras_tuner_results/{model_name}")
        
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()
    
    #Record the training_val and testing indeces
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)
    
    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, _, train_val_y = norm_train_val_data
    norm_test_baseline_X, _, test_y = norm_test_data
    
    dummy_classifiers(X_train=norm_train_val_baseline_X, 
                      y_train=train_val_y, 
                      X_test=norm_test_baseline_X, 
                      y_test=test_y,
                      FUPS_dict = FUPS_dict)
    
def run_ensemble_experiment(model_name, 
                            directory_name, 
                            baseline_dataframe,
                            FUPS_dict,
                            target_series,
                            list_FUP_cols,
                            patient_dataset):
    
    """Performs hyperparamter search, save the best model, and then test on the testing set for the lastFUP_dense experiment.

    Args:
        model_name (string): The name of the hypermodel (could be anything).
        directory_name (string): The name of the directory to save the models info.
        baseline_dataframe (pandas.DataFrame): The Baseline dataset.
        FUPS_dict (dict): The dictionary of FUP data. Keys are the ids, and values are 2D array of (timeline, features).
        target_series (pandas.Series): Pandas Series of all the patients labels.
        list_FUP_cols (list): The list of the feature names (in order) for the FUP columns.
        patient_dataset (Dataset): All patient datset object (this is only used for FUP feature selection).        
    """
    
    #Create a dir to save the result of experiment
    if not os.path.exists(f"./keras_tuner_results/{model_name}"):
        os.makedirs(f"./keras_tuner_results/{model_name}")
        
    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()

    #Record the training and testing indeces
    record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)

    #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
    #train_val data are used for hyperparameter optimization and training.
    baseline_train_val_X, fups_train_val_X, train_val_y = get_X_y_from_indeces(indeces = training_val_indeces, 
                                                                            baseline_data = baseline_dataframe, 
                                                                            FUPS_data_dic = FUPS_dict, 
                                                                            all_targets_data = target_series)

    #Create the feature selection dic (with different methods) for hyperparamter tuning
    patient_dataset_train_val = copy.deepcopy(patient_dataset)    #Keep only the train_val data for feature selection of the FUP data
    patient_dataset_train_val.all_patients = [patient for patient in patient_dataset_train_val.all_patients if patient.uniqid in training_val_indeces]
    feature_selection_dict = create_feature_set_dicts_baseline_and_FUP(baseline_train_val_X, list_FUP_cols, train_val_y, patient_dataset_train_val, mode="Both FUP and Baseline")



    #Standardize the training data, and the test data.
    norm_train_val_data , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    norm_train_val_baseline_X, norm_train_val_fups_X, train_val_y = norm_train_val_data
    norm_test_baseline_X, norm_test_fups_X, test_y = norm_test_data  
    
    #Read the best hyperparameters for the baseline and fup_rnn models
    Baseline_Dense_hp = pd.read_csv("./keras_tuner_results/Baseline_Dense/Baseline_Dense_best_hp.csv", index_col=0)
    Baseline_Dense_hp = {i[0]:i[1].values[0] for i in Baseline_Dense_hp.iterrows()}
    FUP_RNN_hp = pd.read_csv("./keras_tuner_results/FUP_RNN/FUP_RNN_best_hp.csv", index_col=0)
    FUP_RNN_hp = {i[0]:i[1].values[0] for i in FUP_RNN_hp.iterrows()}

    #"baseline_feature_sets" and "FUPS_feature_sets"
    #The choice of the feature set to use for the baseline dataset
    baseline_feature_choice = Baseline_Dense_hp["baseline_feature_set"]
    features_index_to_keep_baseline = feature_selection_dict["baseline_feature_sets"][baseline_feature_choice]
    norm_test_baseline_X = norm_test_baseline_X.iloc[:,features_index_to_keep_baseline]
    norm_train_val_baseline_X = norm_train_val_baseline_X.iloc[:,features_index_to_keep_baseline]


    #The choice of the feature set to use for FUP data
    fups_feature_choice = FUP_RNN_hp["FUPS_feature_set"]
    features_index_to_keep_fups = feature_selection_dict["FUPS_feature_sets"][fups_feature_choice]
    norm_test_fups_X = norm_test_fups_X[:,:,features_index_to_keep_fups]
    norm_train_val_fups_X = norm_train_val_fups_X[:,:,features_index_to_keep_fups]

    print(f"Using the baseline feature set {baseline_feature_choice} and FUP feature set {fups_feature_choice} for testing.")
    
    #Load the saved model of the baseline_Dense and FUP RNN
    model_Baseline_Dense = tf.keras.models.load_model("./keras_tuner_results/Baseline_Dense/Baseline_Dense.h5")
    model_Baseline_Dense._name = "Baseline_Dense"
    model_FUP_RNN = tf.keras.models.load_model("./keras_tuner_results/FUP_RNN/FUP_RNN.h5")
    model_FUP_RNN._name = "FUP_RNN"
    
    #Create the ensemble model
    input_baseline = tf.keras.layers.Input(shape=(norm_test_baseline_X.shape[-1],), dtype="float32", name="baseline_input")
    fup_input = tf.keras.layers.Input(shape=(None,norm_test_fups_X.shape[-1]), dtype="float32", name="fup_input")

    baseline_prediction = model_Baseline_Dense(input_baseline)
    FUP_prediction = model_FUP_RNN(fup_input)

    multi_output = tf.keras.layers.average([baseline_prediction, FUP_prediction])

    ensemble_model = tf.keras.Model(inputs=[input_baseline, fup_input], outputs=multi_output)
    
    ensemble_model.compile(
            optimizer = "ADAM",
            loss = "binary_crossentropy",
            metrics = [
                        tf.keras.metrics.TruePositives(name='tp'),
                        tf.keras.metrics.FalsePositives(name='fp'),
                        tf.keras.metrics.TrueNegatives(name='tn'),
                        tf.keras.metrics.FalseNegatives(name='fn'), 
                        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
                        ],
                    )
    
    #Save the ensemble model
    ensemble_model.save(f"{directory_name}/{model_name}.h5")
    
    #Test on the testing set
    test_res = ensemble_model.evaluate([norm_test_baseline_X, norm_test_fups_X], test_y)
    pd.DataFrame.from_dict({metric:value for metric, value in zip(ensemble_model.metrics_names, test_res)}, orient="index", columns=["test"]).to_csv(f"{directory_name}/{model_name}_test_result.csv")

    save_metrics_and_ROC_PR(model_name=model_name, 
                            model=ensemble_model, 
                            training_x=[norm_train_val_baseline_X, norm_train_val_fups_X], 
                            training_y=train_val_y, 
                            testing_x = [norm_test_baseline_X, norm_test_fups_X], 
                            testing_y = test_y,
                            FUPS_dict = FUPS_dict)
    
    
    

def get_best_number_of_training_epochs(tuner, metric_name, metric_mode, metric_cv_calc_mode):
    """Calculates the best number of epochs to train the model for.

    Args:
        tuner (keras_tuner.tuner): A tuned keras tuner object.
        metric_name (string): The name of the metric we are trying to optimize.
        metric_mode (string): Either optimize based on the "min" or "max" of the metric.
        metric_cv_calc_mode (string): Either "mean" or "median" to use for selecting the best hyperparameter.

    Returns:
        int: Best number of epochs to train the model. 
    """
    
    #Access the results pandas dataframe obtained when tunning
    cv_dic = pd.read_csv(f"./keras_tuner_results/{tuner.hypermodel.name}/{tuner.hypermodel.name}_grid_search_results.csv", index_col="trial_number")
    
    #set of all the trial numbers
    trial_nums = list(set([int(entry.split("_")[1]) for entry in cv_dic.index]))
    
    #Two lists that will contain the mean and median of the metric of interest.
    #
    median_lists = []
    mean_lists = []

    #For each trial, calculate the mean and median of the metric of interest across all the folds.
    for i in trial_nums:
        median_ = cv_dic[cv_dic.index.str.startswith(f"trial_{i}_")][metric_name].median()
        mean_ = cv_dic[cv_dic.index.str.startswith(f"trial_{i}_")][metric_name].mean()
        median_lists.append(median_)
        mean_lists.append(mean_)
        #print(f"Trial {i}, Mean {mean_}, Median {median_}")

    #Using the mean lists, calculate the best trial
    
    if metric_mode == "max":
        best_trial_num_by_mean = trial_nums[np.argmax(mean_lists)]
        best_trial_num_by_median = trial_nums[np.argmax(median_lists)]
        best_metric_by_mean = np.max(mean_lists)
        best_metric_by_median = np.max(median_lists)
    elif metric_mode == "min":
        best_trial_num_by_mean = trial_nums[np.argmin(mean_lists)]
        best_trial_num_by_median = trial_nums[np.argmin(median_lists)]
        best_metric_by_mean = np.min(mean_lists)
        best_metric_by_median = np.min(median_lists)

    #Use the best trial to find the best number of epochs to train the model
    if metric_cv_calc_mode == "mean":
        best_number_of_epochs = round(cv_dic[cv_dic.index.str.startswith(f"trial_{best_trial_num_by_mean}_")]["best_epoch"].mean())
        print("Best Trial by mean is:", best_trial_num_by_mean, f"with {metric_name}: {best_metric_by_mean}") 
        print("Best number of epochs by mean is:", best_number_of_epochs)
        return best_number_of_epochs
    
    elif metric_cv_calc_mode == "median":
        best_number_of_epochs = round(cv_dic[cv_dic.index.str.startswith(f"trial_{best_trial_num_by_median}_")]["best_epoch"].mean())
        print("Best Trial by median is:", best_trial_num_by_median, f"with {metric_name}: {best_metric_by_median}")
        print("Best number of epochs by median is:", best_number_of_epochs)
        return best_number_of_epochs
        
def plot_history(history, name):
    fig, axs = plt.subplots(len(history.history.keys())//4, 2, figsize=(15, 20))

    for key, ax in zip([val for val in history.history.keys() if "val" not in val], axs.flat):
        for metric in [f"{key}", f"val_{key}"]:
            ax.plot(range(1, len(history.history[metric])+1), history.history[metric],"-o", label=metric)
            ax.legend()
            ax.set_xlabel("Epochs")
            
    fig.savefig(f"{name}_train_val_history.png")

def get_last_FUPs_array(fups_X, timeseries_padding_value):
    """Creating an array of the the last FUPs.

    Args:
        fups_X (np.array): The fup array of shape sample:timestamps:features
        timeseries_padding_value (float): The float value used to pad the timeseries data.

    Returns:
        np.array: Last FUP arry of shape sample:features
    """
    new_array_list = []

    for sample in range(len(fups_X)):
        new_array_list.append([])
        for timeline in range(fups_X.shape[1]):
            if (fups_X[sample][timeline]!=np.array([timeseries_padding_value]*fups_X.shape[2], dtype='float32')).all():
                new_array_list[sample].append(fups_X[sample][timeline])
                
    final_FUP_array = []

    for patient_timelines in new_array_list:
        final_FUP_array.append(patient_timelines[-1]) #Extracting the final FUP
        
    return np.array(final_FUP_array)

def save_metrics_and_ROC_PR(model_name, model, training_x, training_y, testing_x, testing_y, FUPS_dict): 
    """Evaluate the model on the training_x and testing_x, then saves the results as pickled pandas name.pkl

    Args:
        model_name (string): Name of the pickled file to be saved.
        model (keras.model): A trained model.
        training_x (numpy.array): Training data (used to train the model).
        training_y (numpy.array): Target y's for the training data.
        testing_x (numpy.array): Training data that is already normalized and has appropriate dimentions.
        testing_y (numpy.array): Target y's for the testing data.
        FUPS_dict (dict): The dictionary of the FUP data. Keys are the ids, and values are 2D array of (timeline, features).
    """
    
    all_data_dic = dict()
    
    for x, y, name in zip([training_x, testing_x], [training_y, testing_y], ["training", "testing"]):
        
        #Get the model's metrics
        res = model.evaluate(x, y)
        res_dict = {metric:value for metric, value in zip(model.metrics_names, res)}
        
        #Get the Precision, Recall, FPR
        y_pred = model.predict(x)
        m = tf.keras.metrics.AUC()
        m.update_state(y, y_pred)
        
        #Record exactly what are the predictions for each sample on the test dataset
        if name == "testing":
            y_pred_classes = (y_pred > 0.5).astype("int32").flatten()
            number_of_FUP = [len(FUPS_dict[uniqid]) for uniqid in list(y.index)]
            record_dict = {"uniqid":list(y.index),"FUP_numbers":number_of_FUP, "y_actual":y.values,
                        "y_pred":y_pred.flatten(), "y_pred_classes":y_pred_classes}

            pd.DataFrame(record_dict).to_csv(f"keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")

        # Set `x` and `y` values for the curves based on `curve` config.
        recall = tf.math.divide_no_nan(
            m.true_positives,
            tf.math.add(m.true_positives, m.false_negatives))

        fp_rate = tf.math.divide_no_nan(
                m.false_positives,
                tf.math.add(m.false_positives, m.true_negatives))

        precision = tf.math.divide_no_nan(
                m.true_positives,
                tf.math.add(m.true_positives, m.false_positives))
        
        res_dict["precision_curve"] = precision.numpy()
        res_dict["fp_rate"] = fp_rate.numpy()
        res_dict["recall_curve"] = recall.numpy()
        res_dict["thresholds"] = m.thresholds
        
        all_data_dic[name] = res_dict
    
    pd.DataFrame.from_dict(all_data_dic, orient="index").to_pickle(f"keras_tuner_results/{model_name}/{model_name}_train_test_results.pkl")
    
def record_training_testing_indeces(model_name, training_val_indeces, testing_indeces):
    ids = training_val_indeces+testing_indeces
    labels = ["training_val"]*len(training_val_indeces) + ["testing"]*len(testing_indeces)
    pd.DataFrame(data = {"uniqids":ids, "category":labels}).sort_values(by=["category", "uniqids"]).reset_index(drop=True).to_csv(f"./keras_tuner_results/{model_name}/{model_name}_testing_training_indeces.csv")
    
