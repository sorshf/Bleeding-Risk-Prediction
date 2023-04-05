import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clinical_score import main as clinical_score_main
import json
import os as os
from venn import venn

from cross_validation import divide_into_stratified_fractions, get_X_y_from_indeces, normalize_training_validation
from hypermodel_experiments import create_feature_set_dicts_baseline_and_FUP
import copy

from data_preparation import prepare_patient_dataset
from constants import data_dir, instruction_dir, discrepency_dir


model_dict = {
    "All_models": ["Baseline_Dense", "FUP_RNN", "LastFUP_Dense", "Ensemble","FUP_Baseline_Multiinput", 'CHAP','ACCP','RIETE','VTE-BLEED','HAS-BLED','OBRI'],
    "Clinical_models": ['CHAP','ACCP','RIETE','VTE-BLEED','HAS-BLED','OBRI'],
    "ML_models": ["Baseline_Dense", "LastFUP_Dense","FUP_RNN", "Ensemble","FUP_Baseline_Multiinput"]
}

#create tf_get_auc function that calculates the roc auc and pr auc, test on the 
def get_auc_using_tf(tn_list, tp_list, fn_list, fp_list):
    """Uses tensorflow and keras to calculate the auc of ROC and PR curves from the tn, tp, fn, fp lists.

    Args:
        tp_list (list): List of number of true positives at 200 thresholds.
        tn_list (list): List of number of true negatives at 200 thresholds.
        fn_list (list): List of number of false negatives at 200 thresholds.
        fp_list (list): List of number of false positives at 200 thresholds.

    Returns:
        float, float: AUC_ROC, AUC_PR
    """
    
    m_ROC = tf.keras.metrics.AUC(curve="ROC")
    m_ROC.true_negatives = np.array(tn_list)
    m_ROC.true_positives = np.array(tp_list)
    m_ROC.false_negatives = np.array(fn_list)
    m_ROC.false_positives = np.array(fp_list)
    
    m_PR = tf.keras.metrics.AUC(curve="PR")
    m_PR.true_negatives = np.array(tn_list)
    m_PR.true_positives = np.array(tp_list)
    m_PR.false_negatives = np.array(fn_list)
    m_PR.false_positives = np.array(fp_list)
    
    return m_ROC.result().numpy(), m_PR.result().numpy()


def calc_PR_ROC_from_y_pred(y_pred, y_actual):
    """Calculates lists of tp, fp, fn, tn, precision, recall, and FPR at 200 thresholds from the prediction score of a binary prblem.
    Note that prediction score could be either probability (0 <= prob <= 1) or clinical score (score > 1).

    Args:
        y_pred (list): List of probability or clinical score calculated by the model.
        y_actual (list(int)): List of actual classes (0 or 1) that the instaces belong to.

    Returns:
        list, list, list, list, list, list, list, : Lists of tp, fp, fn, tn, precision, recall, and FPR.
    """
        
    epsilon = 1e-7
    thr_list = np.linspace(start=min(y_pred), stop=max(y_pred), num=198)
    thr_list = [-epsilon, *thr_list, max(y_pred)+epsilon]
    tp_list = [np.where((y_pred>=thr)&(y_actual==1) , 1, 0).sum().astype("float32") for thr in thr_list]
    fp_list = [np.where((y_pred>=thr)&(y_actual==0) , 1, 0).sum().astype("float32") for thr in thr_list]
    fn_list = [np.where((y_pred<thr)&(y_actual==1) , 1, 0).sum().astype("float32") for thr in thr_list]
    tn_list = [np.where((y_pred<thr)&(y_actual==0) , 1, 0).sum().astype("float32") for thr in thr_list]
    
    recall_curve = tf.math.divide_no_nan(
            tp_list,
            tf.math.add(tp_list, fn_list))

    FPR = tf.math.divide_no_nan(
            fp_list,
            tf.math.add(fp_list, tn_list))

    precision_curve = tf.math.divide_no_nan(
            tp_list,
            tf.math.add(tp_list, fp_list))
    
    return tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR



def plot_iterated_k_fold_scores(metric_name = "val_prc"):
    """Plots the 3 iterated k-fold scores data for the ML models with the best median metric_name.
    
    Args:
        metric_name (string): Which metric to be used to determine the best classifiers, and to be plotted.

    """
    all_plotting_data = []


    for name in model_dict["ML_models"]:
        if name != "Ensemble":

            cv_dic = pd.read_csv(f"./keras_tuner_results/{name}/{name}_grid_search_results.csv", index_col="trial_number")
            cv_dic["trial"] = [trial_num.split("_")[1] for trial_num in cv_dic.index]

            trial_nums = list(set([int(entry.split("_")[1]) for entry in cv_dic.index]))

            median_dict = dict()

            #For each trial, calculate themedian of the metric of interest across all the folds.
            for i in trial_nums:
                median_ = cv_dic[cv_dic.index.str.startswith(f"trial_{i}_")][metric_name].median()
                median_dict[i] = median_
                
            median_dict = {k: v for k, v in sorted(median_dict.items(), key=lambda item: item[1], reverse=True)}
            three_best_trial_nums = [str(val) for val in list(median_dict.keys())[0:3]]

            plotting_data = cv_dic[cv_dic["trial"].isin(three_best_trial_nums)][[f"{metric_name}", "trial"]]
            plotting_data["classifier"] = name
            
            #Order the trials
            plotting_data["trial"] = plotting_data["trial"].replace({three_best_trial_nums[0]:"Best architecture", 
                                                                    three_best_trial_nums[1]:"Second best architecture",
                                                                    three_best_trial_nums[2]:"Third best architectur"})
            
            plotting_data = plotting_data.sort_values(by="trial", ascending=True)
            
            all_plotting_data.append(plotting_data)

    all_plotting_data = pd.concat(all_plotting_data)
    
    fig, ax = plt.subplots(figsize=(12,6))
    g = sns.boxplot(x="classifier", y="val_prc", hue="trial",  data=all_plotting_data, palette="colorblind", ax=ax)
    sns.swarmplot(x="classifier", y="val_prc", hue="trial", dodge=True,color="black", data=all_plotting_data, alpha=0.5, ax=ax)

    g.legend_.set_title(None)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[0:3],labels[0:3])

    ax.set_ylabel("Area under precision-recall curve \n (Iterated 2-fold cv on validation set)")
    ax.set_xlabel("Classifiers")
    
    fig.savefig("./results_pics/iterated_k_fold_results_PR.png")
 
 
    
def plot_validations_train_test():
    """Plot the ROC and PR curve for the ML_models on the training-val data and test data.
    """
    fig, ax = plt.subplots()
    
    model_names = [name for name in model_dict["ML_models"] if name != "Ensemble"]
    
    for name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
        roc_pcr_data = pd.read_pickle(f"./keras_tuner_results/{name}/{name}_train_test_results.pkl")

        for dataset in ["training","testing"]:
            fp_rate = list(roc_pcr_data.loc[dataset, "fp_rate"])
            recall_curve = list(roc_pcr_data.loc[dataset, "recall_curve"])
            auc = roc_pcr_data.loc[dataset, "auc"]
            if dataset == "testing":
                marker = "-."
                alpha = 1
            else:
                marker = "-"
                alpha = 0.5
            plt.plot(fp_rate, recall_curve, marker, label=f"{name}_{dataset} {auc:.3}", color=color, alpha=alpha)
            
    plt.legend()
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("Recall")
    
    fig.savefig(f"./results_pics/roc_ML_models_training_testing.pdf")
    
    fig, ax = plt.subplots()
    
    for name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
        roc_pcr_data = pd.read_pickle(f"./keras_tuner_results/{name}/{name}_train_test_results.pkl")

        for dataset in ["training","testing"]:
            precision_curve = list(roc_pcr_data.loc[dataset, "precision_curve"])
            recall_curve = list(roc_pcr_data.loc[dataset, "recall_curve"])
            prc = roc_pcr_data.loc[dataset, "prc"]
            if dataset == "testing":
                marker = "-."
                alpha = 1
            else:
                marker = "-"
                alpha = 0.5
            plt.plot(recall_curve, precision_curve, marker, label=f"{name}_{dataset} {prc:.5}", color=color, alpha=alpha)
        
    plt.legend()
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    
    plt.savefig(f"./results_pics/pr_ML_models_training_testing.pdf")
  

def save_deatiled_metrics_test():
    
    all_data = []
      
    for model_name in model_dict["All_models"]:
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
        y_pred_classes = np.array(detailed_test_res["y_pred_classes"])
        y_pred = np.array(detailed_test_res["y_pred"])
        y_actual = np.array(detailed_test_res["y_actual"])
        tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR = calc_PR_ROC_from_y_pred(y_pred, y_actual)
        
        ROC_AUC, PR_AUC = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
        
        tp = np.where((y_pred_classes==1)&(y_actual==1) , 1, 0).sum().astype("float32")
        fp = np.where((y_pred_classes==1)&(y_actual==0) , 1, 0).sum().astype("float32")
        fn = np.where((y_pred_classes==0)&(y_actual==1) , 1, 0).sum().astype("float32")
        tn = np.where((y_pred_classes==0)&(y_actual==0) , 1, 0).sum().astype("float32")
        
        recall = tf.math.divide_no_nan(
                tp,
                tf.math.add(tp, fn)).numpy()

        FPR = tf.math.divide_no_nan(
                fp,
                tf.math.add(fp, tn)).numpy()

        precision = tf.math.divide_no_nan(
                tp,
                tf.math.add(tp, fp)).numpy()
        
        all_data.append({
            "Name": model_name,
            "PR_AUC": round(PR_AUC, 3),
            "ROC_AUC": round(ROC_AUC, 3),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "Specificity": round(1 - FPR, 3),
            "FPR": round(FPR, 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
             
        })
        
    pd.DataFrame.from_records(all_data).sort_values(by="PR_AUC", ascending=False).to_csv("./results_pics/detailed_metrics_table_all_models.csv")
        
    
def plot_ROC_PR():
    """Plot the ROC and PR curve for all of the ML models and the clinical scores.
    """
    
    for model_set in model_dict:
        model_names = model_dict[model_set]
        
        fig, ax = plt.subplots(figsize=(6,5))


        for model_name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
            detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
            y_pred = np.array(detailed_test_res["y_pred"])
            y_actual = np.array(detailed_test_res["y_actual"])
            tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR = calc_PR_ROC_from_y_pred(y_pred, y_actual)
            
            ROC_AUC, _ = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
            
            if model_name in model_dict["Clinical_models"]:
                marker = "--"
            else:
                marker = "*-"
            

            plt.plot(FPR, recall_curve, marker, label=f"{model_name} {ROC_AUC:.3}", color=color)
            
        #The random classifier
        plt.plot([0,1], [0, 1], ":", color="black", label="Random_clf 0.50")


        handles, labels = plt.gca().get_legend_handles_labels()
        handle_label_obj = [(h,l, float(l.split(" ")[1])) for h, l in zip(handles, labels)]

        handle_label_obj = sorted(handle_label_obj, key=lambda hl:hl[2], reverse=True)


        plt.legend([h[0] for h in handle_label_obj],[h[1] for h in handle_label_obj])

                
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("Recall")
        
        fig.savefig(f"./results_pics/roc_curve_{model_set}.pdf")
    
    #############################
    
    for model_set in model_dict:
        model_names = model_dict[model_set]

        fig, ax = plt.subplots(figsize=(6,5))

        for model_name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
            detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
            y_pred = np.array(detailed_test_res["y_pred"])
            y_actual = np.array(detailed_test_res["y_actual"])
            tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR = calc_PR_ROC_from_y_pred(y_pred, y_actual)
            
            _, PR_AUC = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
            
            
            if model_name in model_dict["Clinical_models"]:
                marker = "--"
            else:
                marker = "*-"
            

            plt.plot(recall_curve, precision_curve, marker, label=f"{model_name} {PR_AUC:.3}", color=color)

        
        num_positive = detailed_test_res["y_actual"].sum()
        total = len(detailed_test_res)
        pr_baseline = num_positive/total
        
        #The random classifier
        plt.plot([0, 1],[pr_baseline, pr_baseline], ":", color="black", label=f"Random_clf {pr_baseline:.3}")
        
        
        handles, labels = plt.gca().get_legend_handles_labels()
        handle_label_obj = [(h,l, float(l.split(" ")[1])) for h, l in zip(handles, labels)]

        handle_label_obj = sorted(handle_label_obj, key=lambda hl:hl[2], reverse=True)


        plt.legend([h[0] for h in handle_label_obj],[h[1] for h in handle_label_obj])
                
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        
        fig.savefig(f"./results_pics/pr_curve_{model_set}.pdf")


def plot_confusion_matrix():
    """Plot confusion matrix for all of the ML models and Clinical models.
    """

    for name in model_dict["All_models"]:
        
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{name}/{name}_detailed_test_results.csv")

        mosaic = [["All","All","All", "1", "2", "3", "4", "5", "6"],
                ["All","All","All", "7", "8", "9", "10", "11", "12"]]

        fig, axd = plt.subplot_mosaic(mosaic, figsize=(15, 4), layout="constrained")

        for fup_num in ["All","1", "2", "3", "4", "5", "6","7", "8", "9", "10", "11", "12"]:
            

            tp, fp, tn, fn = get_conf_matrix(test_res_df=detailed_test_res, fup_number=fup_num, mode="count")
            heatmap = [[tn,fp],
                    [fn,tp]]
            
            if fup_num != "All":
                sns.heatmap(heatmap, annot=True,linewidths=0.1,cmap=sns.color_palette("viridis", as_cmap=True), square=True, 
                            annot_kws={"size": 10}, fmt="g", ax=axd[fup_num], cbar=False)
            else:
                sns.heatmap(heatmap, annot=True,linewidths=0.1,cmap=sns.color_palette("viridis", as_cmap=True), square=True, 
                            annot_kws={"size": 10}, fmt="g", ax=axd[fup_num], cbar=True)
            
            

            axd[fup_num].set_xlabel("Predicted Label")
            axd[fup_num].set_ylabel("True Label")
            axd[fup_num].set_title(f"FUP={fup_num}")

        fig.suptitle(name, size=25)
        fig.savefig(f"./results_pics/{name}_confusion_matrix.png")   


def extract_the_best_hps(number_of_best_hps):
    """Extract the best hyperparameters from the trained ML models.
    
    Args:
        number_of_best_hps(int): Number of best hps to save as csv.
    """

    for model_name in model_dict["ML_models"]:
        if model_name != "Ensemble":

            all_trials = []
            for trial in os.listdir(f"./keras_tuner_results/{model_name}/{model_name}/"):
                if os.path.isdir(f"./keras_tuner_results/{model_name}/{model_name}/{trial}"):
                    with open(f"./keras_tuner_results/{model_name}/{model_name}/{trial}/trial.json", 'rb') as file:
                        data = json.load(file)
                        score = data["score"]
                        data = data["hyperparameters"]["values"]
                        data["trial"] = int(trial.split("_")[1])
                        data["score"] = score
                        all_trials.append(data)
                        
            all_trials = pd.DataFrame.from_records(all_trials)
            
            all_trials.sort_values(by="score", ascending=False)[0:number_of_best_hps].to_csv(f"./results_pics/{model_name}_top_{number_of_best_hps}_hps.csv")


def get_conf_matrix(test_res_df, fup_number, mode):
    """Use the test result table (with y_actual and y_pred_classes for each instance), calculated number of 
    tp, tn, fp, and tn.

    Args:
        test_res_df (pandas.df): Pandas dataframe with y_actual and y_pred_classes for each instance of the test set.
        fup_number (string): Either "All" which returns all of the [tp, tn, fp, tn] regardless of the number of FUP for that instance,
                            or a "int" which returns [tp, tn, fp, tn] for instances with "int" number of FUPs.
        mode (string): Either "count" which returns number of [tp, tn, fp, tn], or "values" which returns the dataframe subset of [tp, tn, fp, tn].

    Returns:
        _type_: _description_
    """
    if fup_number == "All":
        tp = test_res_df[(test_res_df["y_actual"]==1)&(test_res_df["y_pred_classes"]==1)]
        tn = test_res_df[(test_res_df["y_actual"]==0)&(test_res_df["y_pred_classes"]==0)]
        fp = test_res_df[(test_res_df["y_actual"]==0)&(test_res_df["y_pred_classes"]==1)]
        fn = test_res_df[(test_res_df["y_actual"]==1)&(test_res_df["y_pred_classes"]==0)]
    else:
        tp = test_res_df[(test_res_df["FUP_numbers"]==int(fup_number))&(test_res_df["y_actual"]==1)&(test_res_df["y_pred_classes"]==1)]
        tn = test_res_df[(test_res_df["FUP_numbers"]==int(fup_number))&(test_res_df["y_actual"]==0)&(test_res_df["y_pred_classes"]==0)]
        fp = test_res_df[(test_res_df["FUP_numbers"]==int(fup_number))&(test_res_df["y_actual"]==0)&(test_res_df["y_pred_classes"]==1)]
        fn = test_res_df[(test_res_df["FUP_numbers"]==int(fup_number))&(test_res_df["y_actual"]==1)&(test_res_df["y_pred_classes"]==0)]
    
    if mode == "count":
        return len(tp), len(fp), len(tn), len(fn)
    elif mode == "values":
        return tp, fp, tn, fn
        

def get_tn_fp_fn_tn():
    """Plot the Venn diagram and saves detailed csv between tp or fn of Baseline Dense and FUP_RNN, and 
    tn fp of the Baseline Dense and FUP_RNN.
    """
    Baseline_model_detailed_test_res = pd.read_csv(f"./keras_tuner_results/Baseline_Dense/Baseline_Dense_detailed_test_results.csv")
    FUP_RNN_model_detailed_test_res = pd.read_csv(f"./keras_tuner_results/FUP_RNN/FUP_RNN_detailed_test_results.csv")

    tp_baseline, fp_baseline, tn_baseline, fn_baseline = get_conf_matrix(Baseline_model_detailed_test_res, fup_number="All", mode="values")
    tp_FUP, fp_FUP, tn_FUP, fn_FUP = get_conf_matrix(FUP_RNN_model_detailed_test_res, fup_number="All", mode="values")
    
    
    #######
    #Draw tp fn data
    fig, ax = plt.subplots(figsize=(7,7))

    venn({
        "tp_Baseline-Dense": set(tp_baseline.uniqid),
        "tp_FUP-RNN": set(tp_FUP.uniqid),
        "fn_Baseline-Dense": set(fn_baseline.uniqid),
        "fn_FUP-RNN": set(fn_FUP.uniqid),

    }, ax=ax)

    fig.savefig("./results_pics/tp_fn_on_test_set.png")
    
    ######
    
    tp_fn_dict = {"Positive_both_got_correct": set(tp_baseline.uniqid).intersection(set(tp_FUP.uniqid)),
              "Positive_only_baseline_got_correct": set(tp_baseline.uniqid).intersection(set(fn_FUP.uniqid)),
              "Positive_only_FUP_got_correct": set(tp_FUP.uniqid).intersection(set(fn_baseline.uniqid)),
              "Positive_both_got_wrong": set(fn_FUP.uniqid).intersection(set(fn_baseline.uniqid))}
    

    df_positive = []
    for val in tp_fn_dict:
        df_positive.append(pd.DataFrame.from_dict({"uniqid":list(tp_fn_dict[val]), 
                            "condition": val}))
        
    df_positive = pd.concat(df_positive)

    df_positive["prob_Baseline_Dense"] = df_positive.apply(lambda x:Baseline_model_detailed_test_res.loc[Baseline_model_detailed_test_res["uniqid"]==x["uniqid"],
                                                                                                    "y_pred"].values[0], axis=1)

    df_positive["prob_FUP_RNN"] = df_positive.apply(lambda x:FUP_RNN_model_detailed_test_res.loc[FUP_RNN_model_detailed_test_res["uniqid"]==x["uniqid"],
                                                                                                    "y_pred"].values[0], axis=1)

    
    df_positive.to_csv("./results_pics/tp_fn_on_test_set.csv", index=False)

    ##############
    
    fig, ax = plt.subplots(figsize=(7,7))

    venn({
        "tn_Baseline-Dense": set(tn_baseline.uniqid),
        "tn_FUP-RNN": set(tn_FUP.uniqid),
        "fp_Baseline-Dense": set(fp_baseline.uniqid),
        "fp_FUP-RNN": set(fp_FUP.uniqid),

    }, ax=ax)

    fig.savefig("./results_pics/tn_fp_on_test_set.png")
    
    ################
    
    tn_fp_dict = {"Negative_both_got_correct": set(tn_baseline.uniqid).intersection(set(tn_FUP.uniqid)),
              "Negative_only_baseline_got_correct": set(tn_baseline.uniqid).intersection(set(fp_FUP.uniqid)),
              "Negative_only_FUP_got_correct": set(tn_FUP.uniqid).intersection(set(fp_baseline.uniqid)),
              "Negative_both_got_wrong": set(fp_FUP.uniqid).intersection(set(fp_baseline.uniqid))}

    df_negative = []
    for val in tn_fp_dict:
        df_negative.append(pd.DataFrame.from_dict({"uniqid":list(tn_fp_dict[val]), 
                            "condition": val}))
        
    df_negative = pd.concat(df_negative)

    df_negative["prob_Baseline_Dense"] = df_negative.apply(lambda x:Baseline_model_detailed_test_res.loc[Baseline_model_detailed_test_res["uniqid"]==x["uniqid"],
                                                                                                    "y_pred"].values[0], axis=1)

    df_negative["prob_FUP_RNN"] = df_negative.apply(lambda x:FUP_RNN_model_detailed_test_res.loc[FUP_RNN_model_detailed_test_res["uniqid"]==x["uniqid"],
                                                                                                    "y_pred"].values[0], axis=1)

    df_negative.to_csv("./results_pics/tn_fp_on_test_set.csv", index=False)
    
    
def create_feature_sets_json():
    """Generates a json with the feature sets used to train the baseline and FUP models.
    """


        
    #Read the patient dataset
    patient_dataset = prepare_patient_dataset(data_dir, instruction_dir, discrepency_dir)

    #Remove patients without baseline, remove the FUPS after bleeding/termination, fill FUP data for those without FUP data
    patient_dataset.filter_patients_sequentially(mode="fill_patients_without_FUP")
    print(patient_dataset.get_all_targets_counts(), "\n")

    #Add one feature to each patient indicating year since baseline to each FUP
    patient_dataset.add_FUP_since_baseline()

    #Get the BASELINE, and Follow-up data from patient dataset
    FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = patient_dataset.get_data_x_y(baseline_filter=["uniqid", "dtbas", "vteindxdt", "stdyoatdt", "inrbas"], 
                                                                                                FUP_filter=[])
    print(f"Follow-up data has {len(FUPS_dict)} examples and {len(list_FUP_cols)} features.")
    print(f"Baseline data has {len(baseline_dataframe)} examples and {len(baseline_dataframe.columns)} features.")
    print(f"The target data has {len(target_series)} data.", "\n")
    

    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)


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

    for feature_set in feature_selection_dict["baseline_feature_sets"]:
        index = feature_selection_dict["baseline_feature_sets"][feature_set]
        
        feature_selection_dict["baseline_feature_sets"][feature_set] = [baseline_dataframe.columns[i] for i in index]
        
    for feature_set in feature_selection_dict["FUPS_feature_sets"]:
        index = feature_selection_dict["FUPS_feature_sets"][feature_set]
        
        feature_selection_dict["FUPS_feature_sets"][feature_set] = [list_FUP_cols[i] for i in index]



    with open("./keras_tuner_results/feature_sets.json", 'w') as file:
        file.write(json.dumps(feature_selection_dict))
       
def main():
    
    create_feature_sets_json()
    
    clinical_score_main()
    
    plot_iterated_k_fold_scores()
    
    plot_validations_train_test()
    
    plot_ROC_PR()
    
    plot_confusion_matrix()
    
    extract_the_best_hps(number_of_best_hps=200)
    
    get_tn_fp_fn_tn()

    save_deatiled_metrics_test()
    
if __name__=="__main__":
    main()