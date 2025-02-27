import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import json
import os as os
from venn import venn

from cross_validation import divide_into_stratified_fractions, get_X_y_from_indeces, normalize_training_validation
from hypermodel_experiments import create_feature_set_dicts_baseline_and_FUP
import copy

from constants import instruction_dir, timeseries_padding_value
from tensorflow import keras
from data_preparation import get_abb_to_long_dic
import re

from mlxtend.evaluate import cochrans_q
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
from matplotlib import colors
import random 

from train import get_formatted_Baseline_FUP
from hypermodel_experiments import get_predefined_training_testing_indeces_30_percent
from hypermodel_experiments import record_training_testing_indeces
from clinical_score import ClinicalScore

from statistical_tests import correct_p_values, plot_p_value_heatmap, compare_estimates_t_statistics

model_dict = {
    "All_models": ["Baseline_Dense", "FUP_RNN", "LastFUP_Dense", "Ensemble", 'CHAP','ACCP','RIETE','VTE-BLEED','HAS-BLED','OBRI'],
    "Clinical_models": ['CHAP','ACCP','RIETE','VTE-BLEED','HAS-BLED','OBRI'],
    "ML_models": ["Baseline_Dense", "LastFUP_Dense","FUP_RNN", "Ensemble",]
}

model_paper_dic = {
    "Baseline_Dense":"Baseline-ANN",
    "LastFUP_Dense":"LastFUP-ANN",
    "FUP_RNN":"FUP-RNN",
    "Ensemble":"Ensemble",
    #"FUP_Baseline_Multiinput":"Multimodal", Multimodal has been eliminated
    'CHAP':"CHAP",
    'ACCP':'ACCP',
    'RIETE':'RIETE',
    'VTE-BLEED':'VTE-BLEED',
    'HAS-BLED':'HAS-BLED',
    'OBRI':'OBRI',
    "Random_clf": "Random-Classifier"
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
    """Plot and record csv the ROC and PR curve for the ML_models on the training-val data and test data.
    """
    
    #AUROC and PRAUC for each model is saved in ROC_PR_dic for csv
    ROC_PR_dic = dict()
    
    fig, ax = plt.subplots()
    
    model_names = model_dict["ML_models"]
    
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
            ROC_PR_dic[f"{model_paper_dic[name]}_{dataset}_AUROC"] = f"{auc:.3}"
            plt.plot(fp_rate, recall_curve, marker, label=f"{model_paper_dic[name]}_{dataset} {auc:.3}", color=color, alpha=alpha)
            
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
            ROC_PR_dic[f"{model_paper_dic[name]}_{dataset}_AUPRC"] = f"{prc:.5}"
            plt.plot(recall_curve, precision_curve, marker, label=f"{model_paper_dic[name]}_{dataset} {prc:.5}", color=color, alpha=alpha)
        
    plt.legend()
    
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    
    plt.savefig(f"./results_pics/pr_ML_models_training_testing.pdf")
    
    
    data  = pd.DataFrame.from_dict(ROC_PR_dic, orient="index", columns=["value"])

    data["dataset"] = data.apply(lambda x: x.name.split("_")[1], axis=1)
    data["metric"] = data.apply(lambda x: x.name.split("_")[2], axis=1)
    data["model"] = data.apply(lambda x: x.name.split("_")[0], axis=1)

    data = data.pivot(index="model", columns=["metric", "dataset"], values="value")
    
    data.to_csv("./results_pics/detailed_training_testing_AUROC_AUPRC.csv")

  

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
        
        accuracy = (tn+tp)/(tn+tp+fp+fn)
        
        all_data.append({
            "Name": model_paper_dic[model_name],
            "Accuracy": round(accuracy, 3),
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
        
    pd.DataFrame.from_records(all_data).to_csv("./results_pics/detailed_metrics_table_all_models.csv")
        
    
def plot_ROC_PR():
    """Plot the ROC and PR curve for all of the ML models and the clinical scores.
    """
    
    for model_set in ['All_models']:
        model_names = model_dict[model_set]
        
        fig, ax = plt.subplots(figsize=(6/1.1,5/1.1))


        for model_name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
            #model_name = "CHAP"
            detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
            y_pred = np.array(detailed_test_res["y_pred"])
            y_actual = np.array(detailed_test_res["y_actual"])
            tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR = calc_PR_ROC_from_y_pred(y_pred, y_actual)
            
            ROC_AUC, _ = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
            
            if model_name in model_dict["Clinical_models"]:
                marker = "--"
            else:
                marker = "-"
            

            plt.plot(FPR, recall_curve, marker, label=f"{model_name} {ROC_AUC:.3}",linewidth=1.2,markersize=4, color=color)
            #break
            
        #The random classifier
        plt.plot([0,1], [0, 1], ":", color="black", label="Random_clf 0.50")


        handles, labels = plt.gca().get_legend_handles_labels()
        handle_label_obj = [(h,model_paper_dic[l.split(" ")[0]], float(l.split(" ")[1])) for h, l in zip(handles, labels)]

        handle_label_obj = sorted(handle_label_obj, key=lambda hl:hl[2], reverse=True)


        plt.legend([h[0] for h in handle_label_obj],[str(h[1])+f" ({h[2]})" for h in handle_label_obj], loc='best', title="Model (AUROC)", fancybox=True, 
                   fontsize=9)

                
        ax.set_xlabel("False Positive Rate (1-Specificity)", fontdict={"fontsize":13})
        ax.set_ylabel("Recall", fontdict={"fontsize":13})
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        fig.savefig(f"./results_pics/roc_curve_{model_set}.pdf")
    
    #############################
    
    for model_set in ['All_models']:
        model_names = model_dict[model_set]

        fig, ax = plt.subplots(figsize=(6/1.1,5/1.1))

        for model_name, color in zip(model_names, list(sns.color_palette("colorblind", len(model_names)))):
            detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
            y_pred = np.array(detailed_test_res["y_pred"])
            y_actual = np.array(detailed_test_res["y_actual"])
            tp_list, fp_list, fn_list, tn_list, precision_curve, recall_curve, FPR = calc_PR_ROC_from_y_pred(y_pred, y_actual)
            
            _, PR_AUC = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
            
            
            if model_name in model_dict["Clinical_models"]:
                marker = "--"
            else:
                marker = "-"
            

            plt.plot(recall_curve, precision_curve, marker, label=f"{model_name} {PR_AUC:.3}",linewidth=1.2,markersize=4, color=color)

        
        num_positive = detailed_test_res["y_actual"].sum()
        total = len(detailed_test_res)
        pr_baseline = num_positive/total
        
        #The random classifier
        plt.plot([0, 1],[pr_baseline, pr_baseline], ":", color="black", label=f"Random_clf {pr_baseline:.3}")
        
        
        handles, labels = plt.gca().get_legend_handles_labels()
        handle_label_obj = [(h,model_paper_dic[l.split(" ")[0]], float(l.split(" ")[1])) for h, l in zip(handles, labels)]

        handle_label_obj = sorted(handle_label_obj, key=lambda hl:hl[2], reverse=True)


        plt.legend([h[0] for h in handle_label_obj],[str(h[1])+f" ({h[2]})" for h in handle_label_obj], loc='best', title="Model (AUPRC)", fancybox=True, 
                   fontsize=9)
                
        ax.set_xlabel("Recall", fontdict={"fontsize":13})
        ax.set_ylabel("Precision", fontdict={"fontsize":13})
        ax.tick_params(axis='both', which='major', labelsize=12)

        
        fig.savefig(f"./results_pics/pr_curve_{model_set}.pdf")


def plot_confusion_matrix():
    """Plot confusion matrix for all of the ML models and Clinical models.
    """
    
    #Adding the custom color bar to the confusion matrices
    cmap = (colors.ListedColormap(['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725'])
        .with_extremes(over='9e-3', under='9e-4'))
    bounds = [0, 0.5, 10, 20, 100, 1000]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    
    for name in model_dict["All_models"]:
        
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{name}/{name}_detailed_test_results.csv")
        
        detailed_test_res["FUP_numbers"] = detailed_test_res.apply(lambda x: x["FUP_numbers"]-1, axis=1)

        mosaic = [["All","All","All", "0", "1", "2", "3", "4", "5", "6"],
                ["All","All","All", "7", "8", "9", "10", "11", "12", "13"]]

        fig, axd = plt.subplot_mosaic(mosaic, figsize=(13.5, 4), layout="constrained")

        for fup_num in ["All", "0", "1", "2", "3", "4", "5", "6","7", "8", "9", "10", "11", "12"]:
            

            tp, fp, tn, fn = get_conf_matrix(test_res_df=detailed_test_res, fup_number=fup_num, mode="count")
            heatmap = [[tn,fp],
                    [fn,tp]]
            
            if fup_num != "All":
                sns.heatmap(heatmap, annot=True,linewidths=2,cmap=cmap, norm=norm, square=True, 
                            annot_kws={"size": 11}, fmt="g", ax=axd[fup_num], cbar=False)
                label_size = 8
                
            else:
                sns.heatmap(heatmap, annot=True,linewidths=4,cmap=cmap, norm=norm, square=True, 
                            annot_kws={"size": 15}, fmt="g", ax=axd[fup_num], cbar=True, cbar_kws = dict(use_gridspec=False,location="left"))
                
                
                label_size = 16
            
            axd[fup_num].tick_params(labelsize=label_size)


            axd[fup_num].set_xlabel("Predicted Label", fontdict={"fontsize":label_size})
            axd[fup_num].set_ylabel("True Label", fontdict={"fontsize":label_size})
            axd[fup_num].set_title(f"{fup_num} FUPS", size=label_size*1.4)

        #The 13th confusion matrix should be empty (no patient with 13 fup in the test set)
        axd['13'].axis('off')
        
        fig.suptitle(model_paper_dic[name], size=30)
        fig.savefig(f"./results_pics/{name}_confusion_matrix_new.pdf", bbox_inches='tight')   


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


    patient_dataset, FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = get_formatted_Baseline_FUP(mode="Formatted")
    

    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()


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
    
    #######################
    #Read the json file and save it as csv  
        
    with open("./keras_tuner_results/feature_sets.json") as f:
        data = json.load(f)
    
    def concat_list_contents(list_):
        content = ""
        for i, item in enumerate(list_):
            sep = "" if i == 0 else ", "
            content = content + sep + str(item)
        return content.replace("\u2014", "-")

    new_dict = dict()

    for feature_type in data.keys():
        for feature_set in data[feature_type].keys():
            new_dict[feature_set] = concat_list_contents(data[feature_type][feature_set])
            
    pd.DataFrame.from_dict(new_dict, orient="index").to_csv("./keras_tuner_results/feature_sets.csv")


def plot_FUP_count_density():
    """Plots the count and density diagrams for the patients' number of follow-ups.
    """
    
    patient_dataset, FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = get_formatted_Baseline_FUP(mode="Formatted")
    
    
    
    bleeding_frequency = []
    non_bleeder_frequency = []

    for patient in patient_dataset:
        #For the bleeders
        if (patient.get_target() == 1): 
            if patient.missing_FUP:
                bleeding_frequency.append(0)           
            else:
                bleeding_frequency.append(len(patient.get_FUP_array())-1) #-1 is because all of the patients had zeroth FUP
        #For the non-bleeders
        else:
            if patient.missing_FUP: 
                non_bleeder_frequency.append(0)            
            else:
                non_bleeder_frequency.append(len(patient.get_FUP_array())-1) #-1 is because all of the patients had zeroth FUP
    
    
    #unique frequencies of FUPs        
    follow_up_numbers = len(set(bleeding_frequency).union(set(non_bleeder_frequency)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,3))

    color1, color2 = list(sns.color_palette("colorblind", 2))

    sns.histplot(non_bleeder_frequency, discrete=True, multiple="dodge", color=color1,
                label="Non-bleeders", common_norm=False,stat="count", ax=ax1, alpha=0.5)

    sns.histplot(bleeding_frequency, discrete=True,multiple="dodge",color=color2,
                label="Bleeders",common_norm=False, stat="count", ax=ax1,alpha=0.5)

    ax1.set_xticks(range(follow_up_numbers))

    ax1.legend()

    ax1.set_xlabel("Number of follow-ups", fontdict={"fontsize":12})
    ax1.set_ylabel("Count", fontdict={"fontsize":12})

    #########

    sns.histplot(non_bleeder_frequency, discrete=True, multiple="dodge",color=color1,
                label="Non-bleeders", common_norm=False,stat="density", ax=ax2, alpha=0.5)

    sns.histplot(bleeding_frequency, discrete=True,multiple="dodge",color=color2,
                label="Bleeders", common_norm=False, stat="density", ax=ax2, alpha=0.5)

    ax2.set_xticks(range(follow_up_numbers))
    ax2.legend()

    ax2.set_xlabel("Number of follow-ups", fontdict={"fontsize":12})
    ax2.set_ylabel("Density", fontdict={"fontsize":12})
    
    
    fig.savefig("./results_pics/number_FUP_count_density.pdf", bbox_inches='tight')



def plot_permutaion_feature_importance_RNN_FUP(number_of_permutations=100):
    """Plot permutation importance figures for the FUP_RNN model.
    """
        
    _, FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = get_formatted_Baseline_FUP(mode="Formatted")


    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)

    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()

    #Standardize the training data, and the test data.
    _ , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    _ , norm_test_fups_X, test_y = norm_test_data
    
    
    SUM_PADDED_FUP = timeseries_padding_value * norm_test_fups_X.shape[-1]


    #Load the saved model
    model = keras.models.load_model("./keras_tuner_results/FUP_RNN/FUP_RNN.h5")


    #Calculate the prc and auc on the non-purturbed dataset
    test_res = model.evaluate(norm_test_fups_X, test_y, verbose=0)
    result_dic = {metric:value for metric, value in zip(model.metrics_names,test_res)}
    auroc = result_dic["auc"]
    auprc = result_dic["prc"]

    #Create a copy of the test data, vertically stack them along the time axis, then get rid of the padded sequences
    test_fups_2_dims = copy.deepcopy(norm_test_fups_X)
    test_fups_2_dims = test_fups_2_dims.reshape(test_fups_2_dims.shape[0] * test_fups_2_dims.shape[1], test_fups_2_dims.shape[2]) #reshape(754*14, 45)
    random_sampling_fups_dataset = test_fups_2_dims[np.sum(test_fups_2_dims, axis=1) != SUM_PADDED_FUP]


    #A function to purtub time series dataset
    def purturb_timeseries(test_fups_X, column_index):
        
        purturbed_patients_list = []

        for patient in copy.deepcopy(test_fups_X):
            new_timesteps = []
            for time in patient:
                if np.sum(time) != SUM_PADDED_FUP:
                    time[column_index] = random_sampling_fups_dataset[np.random.choice(range(len(random_sampling_fups_dataset))) ,column_index]
                    new_timesteps.append(time)
                else:
                    new_timesteps.append(time)
                    
            purturbed_patients_list.append(new_timesteps)
            
        return np.array(purturbed_patients_list)


    #Add space in the names of the features to make them look good on figures
    def turn_space_into_newline(a_string):
        spaces_list = [a.start() for a in re.finditer(" ", a_string)]
        s = list(a_string)
        if len(spaces_list) > 3:
            s[spaces_list[int(len(spaces_list)/2)-1]] = "\n"
        return "".join(s)

                
    new_FUP_col_list = [turn_space_into_newline(col) for col in list_FUP_cols]

    #Group the one-hot encoded features in the grouped_FUP_col_list.
    grouped_FUP_col_list = dict()
    counter = 0

    for col in new_FUP_col_list:
        is_grouped_feature = False
        for group_feat_name in ["Hypertension", "Diabetes Mellitus", "Myocardial Infarction", "Stroke", "Atrial fibrillation", "Post-thrombotic syndrome", "Recent Cancer diagnosis"]:
            if ((group_feat_name in col) & (group_feat_name not in grouped_FUP_col_list)):
                grouped_FUP_col_list[group_feat_name] = [counter]
                is_grouped_feature = True
                counter += 1
            elif ((group_feat_name in col) & (group_feat_name in grouped_FUP_col_list)):
                is_grouped_feature = True
                grouped_FUP_col_list[group_feat_name].append(counter)
                counter += 1
        
        if (not is_grouped_feature):
            grouped_FUP_col_list[col] = [counter]          
            counter += 1
    
    print(f"The total number of FUP features after grouping one hot encoded features is {len(grouped_FUP_col_list)}")
            
    # For each column in FUP test set (as defined by grouped_FUP_col_list)
    ## For number of permutation
    ### Copy the intact FUP test set
    ### permute the column for the FUP test set
    ### record the prc and auc

    prc_permutation_results = {col:[] for col in grouped_FUP_col_list.keys()}
    roc_permutation_results = {col:[] for col in grouped_FUP_col_list.keys()}

    for col_name in grouped_FUP_col_list.keys():
        for _ in range(number_of_permutations): #Number of permutations
            fup_test_copy = copy.deepcopy(norm_test_fups_X)
            
            #perturb column
            purturbed_data = purturb_timeseries(fup_test_copy, column_index=grouped_FUP_col_list[col_name])
            
            test_res = model.evaluate(purturbed_data, test_y, verbose=0)
            result_dic = {metric:value for metric, value in zip(model.metrics_names,test_res)}
            
            prc_permutation_results[col_name].append(auprc-result_dic["prc"])
            roc_permutation_results[col_name].append(auroc-result_dic["auc"])
            
        print(col_name, "permutation is done.")


    color2 = sns.color_palette("colorblind", 15)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,3))


    data_plot = pd.DataFrame.from_dict({k: v for k, v in sorted(roc_permutation_results.items(), key=lambda item: np.mean(item[1]), reverse=True)})
    data_plot.to_pickle("./results_pics/permutation_test_results.pkl")
    data_plot = data_plot.iloc[:,0:5]
    #ax1.boxplot(data_plot, vert=False)
    sns.boxplot(data_plot, ax=ax1, orient='h', color=color2[9], showfliers = False)
    # ax1.set_yticklabels(data_plot.columns, fontsize=8)
    ax1.vlines(0,-1,45, linestyles="--", color="grey", alpha=0.5)
    ax1.set_xlabel("Loss in AUROC", fontdict={"fontsize":15})
    sns.stripplot(data_plot, ax=ax1, orient='h', palette='dark:.15', marker=".", alpha=0.2)
    # ax1.set_yticklabels(ax1.get_yticklabels(),ha="center")
    ax1.set_ylabel("Predictor Variables", fontdict={"fontsize":15})
    ax1.tick_params(axis='both', which='major', labelsize=15)


    data_plot = pd.DataFrame.from_dict({k: v for k, v in sorted(prc_permutation_results.items(), key=lambda item: np.mean(item[1]), reverse=True)})
    #ax2.boxplot(data_plot, vert=False)
    data_plot = data_plot.iloc[:,0:5]

    sns.boxplot(data_plot, ax=ax2, orient='h',color=color2[9],showfliers = False)
    # ax2.set_yticklabels(data_plot.columns, fontsize=8)
    ax2.vlines(0,-1,45, linestyles="--", color="grey", alpha=0.5)
    ax2.set_xlabel("Loss in AUPRC", fontdict={"fontsize":15})
    sns.stripplot(data_plot, ax=ax2, orient='h', palette='dark:.15', marker=".", alpha=0.2)
    # ax2.set_yticklabels(ax2.get_yticklabels(),ha="center")

    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    fig.savefig("./results_pics/feature_importance_RNN_FUP_5values.pdf")



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,12))


    data_plot = pd.DataFrame.from_dict({k: v for k, v in sorted(roc_permutation_results.items(), key=lambda item: np.mean(item[1]), reverse=True)})
    #ax1.boxplot(data_plot, vert=False)
    sns.boxplot(data_plot, ax=ax1, orient='h', color=color2[9], showfliers = False)
    # ax1.set_yticklabels(data_plot.columns, fontsize=8)
    ax1.vlines(0,-1,45, linestyles="--", color="grey", alpha=0.5)
    ax1.set_xlabel("Loss in AUROC", fontdict={"fontsize":16})
    sns.stripplot(data_plot, ax=ax1, orient='h', palette='dark:.15', marker=".", alpha=0.2)
    # ax1.set_yticklabels(ax1.get_yticklabels(),ha="center")
    ax1.set_ylabel("Predictor Variables", fontdict={"fontsize":16})


    data_plot = pd.DataFrame.from_dict({k: v for k, v in sorted(prc_permutation_results.items(), key=lambda item: np.mean(item[1]), reverse=True)})
    #ax2.boxplot(data_plot, vert=False)
    sns.boxplot(data_plot, ax=ax2, orient='h',color=color2[9], showfliers = False)
    # ax2.set_yticklabels(data_plot.columns, fontsize=8)
    ax2.vlines(0,-1,45, linestyles="--", color="grey", alpha=0.5)
    ax2.set_xlabel("Loss in AUPRC", fontdict={"fontsize":15})
    sns.stripplot(data_plot, ax=ax2, orient='h', palette='dark:.15', marker=".", alpha=0.2)
    # ax2.set_yticklabels(ax2.get_yticklabels(),ha="center")


    plt.tight_layout()

    ax1.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    fig.savefig("./results_pics/feature_importance_RNN_FUP_all_values.pdf")
 
 
 
def mcnemar_analysis():
    """Perform Cochran's Q test, followed by pairwise McNemar test. Generate a pic of the p-values as a heatmap"""
    
    #Create a dictionary with the predicted class of the data on the testing set
    y_pred_dict = dict()

    #Populate the dictionary made above
    for model_name in model_dict["All_models"]:
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
        
        y_pred = np.array(detailed_test_res["y_pred_classes"])
        y_actual = np.array(detailed_test_res["y_actual"])
        
        y_pred_dict[model_paper_dic[model_name]] = y_pred
        

    #Perform Cochrane's Q test
    q_cochrane, p_value_cochrane = cochrans_q(y_actual, 
                            y_pred_dict["Baseline-ANN"],
                            y_pred_dict["FUP-RNN"],
                            y_pred_dict["LastFUP-ANN"],
                            y_pred_dict["Ensemble"],
                            # y_pred_dict["Multimodal"], Removed multimodal
                            y_pred_dict["CHAP"],
                            y_pred_dict["ACCP"],
                            y_pred_dict["RIETE"],
                            y_pred_dict["VTE-BLEED"],
                            y_pred_dict["HAS-BLED"],
                            y_pred_dict["OBRI"],
                            )


    #Perform pairwise mcnemar's test
    stat_test_results = pd.DataFrame(columns=y_pred_dict.keys(), index=y_pred_dict.keys())
    for model_1 in y_pred_dict.keys():
        for model_2 in y_pred_dict.keys():
            chi2, p_value = mcnemar(mcnemar_table(y_actual, 
                            y_pred_dict[model_1],
                            y_pred_dict[model_2]),
                            corrected=True, exact=True)
            stat_test_results.loc[model_1, model_2] = "{:.2e}".format(p_value)
            
   
    #Find the order of models from highest to lowest accuracy
    #This is so that we can build a top diagonal matrix for p-values
    accuracy_dict = dict()
      
    for model_name in model_dict["All_models"]:
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
        y_pred_classes = np.array(detailed_test_res["y_pred_classes"])
        y_pred = np.array(detailed_test_res["y_pred"])
        y_actual = np.array(detailed_test_res["y_actual"])
                
        tp = np.where((y_pred_classes==1)&(y_actual==1) , 1, 0).sum().astype("float32")
        fp = np.where((y_pred_classes==1)&(y_actual==0) , 1, 0).sum().astype("float32")
        fn = np.where((y_pred_classes==0)&(y_actual==1) , 1, 0).sum().astype("float32")
        tn = np.where((y_pred_classes==0)&(y_actual==0) , 1, 0).sum().astype("float32")
        
        accuracy = (tn+tp)/(tn+tp+fp+fn)
        
        accuracy_dict[model_name] = accuracy

    accuracy_dict = dict(sorted(accuracy_dict.items(), key=lambda item: item[1], reverse=True))
    row_order = list(accuracy_dict.keys())
    row_order = [model_paper_dic[val] for val in row_order]
    
    #Correct the order of stat test results
    stat_test_results = stat_test_results.reindex(row_order)
    stat_test_results = stat_test_results[row_order[::-1]] #Reverse the row order
                
    #Correct p-values for multiple hypothesis testing
    stat_test_results_corrected, multitest_correction = correct_p_values(stat_test_results, multitest_correction="fdr_bh")
    
    #Remove the last row and the last column of the p-value tables becuase they are redundant
    stat_test_results_corrected = stat_test_results_corrected.drop(stat_test_results_corrected.columns[-1], axis=1)
    stat_test_results_corrected = stat_test_results_corrected.drop(stat_test_results_corrected.index[-1], axis=0)  
    
    #Plot the hitmap of the corrected p_values
    plot_p_value_heatmap(stat_test_results_corrected, title="McNemar test",
                         save_path="./results_pics/", multitest_correction=multitest_correction, 
                     omnibus_p_value=f"Cochran q: {p_value_cochrane}", plot_name="McNemar")
    

def plot_FUP_RNN_probabilities_output():
    """For the FUP_RNN model, generates the probabilites of bleeding for all the bleeders in the test set and random 
       subset of non-bleeders in the test set.
    """
    
    patient_dataset, FUPS_dict, list_FUP_cols, baseline_dataframe, target_series = get_formatted_Baseline_FUP(mode="Formatted")


    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()

    #Standardize the training data, and the test data.
    _ , norm_test_data = normalize_training_validation(training_indeces = training_val_indeces, 
                                                                        validation_indeces = testing_indeces, 
                                                                        baseline_data = baseline_dataframe, 
                                                                        FUPS_data_dict = FUPS_dict, 
                                                                        all_targets_data = target_series, 
                                                                        timeseries_padding_value=timeseries_padding_value)

    #unwrap the training-val and testing variables
    _ , norm_test_fups_X, test_y = norm_test_data
    
    SUM_PADDED_FUP = timeseries_padding_value * norm_test_fups_X.shape[-1]

    #Load the saved model
    model = keras.models.load_model("./keras_tuner_results/FUP_RNN/FUP_RNN.h5")
    
    #This should be run because of a bug in tf!!
    test_res = model.predict(norm_test_fups_X[0:1])
    
    #Calculate the progression of probabilites 
    probability_progression = dict()
    for i, uniqid in enumerate(testing_indeces):
        
        all_prob = []
        
        for t in range(13):
            if np.sum(norm_test_fups_X[i:i+1,t:t+1,:]) == SUM_PADDED_FUP: #These are the timestamps that were padded
                break
            
            prob = model.predict(norm_test_fups_X[i:i+1,0:t+1,:])
            all_prob.append(float(prob))
            
        probability_progression[uniqid] = all_prob
        
        
    #Create two dictionaries to store positive (bleeders) and negative (non-bleeders) patients as uniqid:num_FUP
    id_to_target_dict = {test_id:test_y[test_id] for test_id in testing_indeces}
    id_to_num_FUP_dict_pos = {key_:len(val_) for key_, val_ in probability_progression.items() if id_to_target_dict[key_]==1}
    id_to_num_FUP_dict_neg = {key_:len(val_) for key_, val_ in probability_progression.items() if id_to_target_dict[key_]==0}
    
    #For the patients with missing FUP (that were artificially generated), we set the number of their FUP as zero
    for id_ in id_to_num_FUP_dict_neg:
        if patient_dataset[id_].missing_FUP:
            id_to_num_FUP_dict_neg[id_] = 0
        
    for id_ in id_to_num_FUP_dict_pos:
        if patient_dataset[id_].missing_FUP:
            id_to_num_FUP_dict_pos[id_] = 0

    #For the dictionary with non-bleeders, because they are many, we randomly sample some of them
    random.seed(1)

    dict_of_num_FUP_to_list_of_uniqids = dict()

    for num_FUP in range(13):
        #Initialize the list which is the value for the keys
        if num_FUP not in dict_of_num_FUP_to_list_of_uniqids:
            dict_of_num_FUP_to_list_of_uniqids[num_FUP] = []
            
        #Populate the dict
        for uniqid in id_to_num_FUP_dict_neg:
            if id_to_num_FUP_dict_neg[uniqid] == num_FUP:
                dict_of_num_FUP_to_list_of_uniqids[num_FUP].append(uniqid)
                
    #Randomly choose 10, 10, 6 , 4, 4, 3, 3, 2, 2, 2, 2 patients from the negatives (non-bleeders)
    dict_of_num_FUP_to_list_of_uniqids[0] = random.sample(dict_of_num_FUP_to_list_of_uniqids[0], 10)
    dict_of_num_FUP_to_list_of_uniqids[1] = random.sample(dict_of_num_FUP_to_list_of_uniqids[1], 10)
    dict_of_num_FUP_to_list_of_uniqids[2] = random.sample(dict_of_num_FUP_to_list_of_uniqids[2], 6)
    dict_of_num_FUP_to_list_of_uniqids[3] = random.sample(dict_of_num_FUP_to_list_of_uniqids[3], 4)
    dict_of_num_FUP_to_list_of_uniqids[4] = random.sample(dict_of_num_FUP_to_list_of_uniqids[4], 4)
    dict_of_num_FUP_to_list_of_uniqids[5] = random.sample(dict_of_num_FUP_to_list_of_uniqids[5], 3)
    dict_of_num_FUP_to_list_of_uniqids[6] = random.sample(dict_of_num_FUP_to_list_of_uniqids[6], 3)
    dict_of_num_FUP_to_list_of_uniqids[7] = random.sample(dict_of_num_FUP_to_list_of_uniqids[7], 3)
    dict_of_num_FUP_to_list_of_uniqids[8] = random.sample(dict_of_num_FUP_to_list_of_uniqids[8], 2)
    dict_of_num_FUP_to_list_of_uniqids[9] = random.sample(dict_of_num_FUP_to_list_of_uniqids[9], 2)
    dict_of_num_FUP_to_list_of_uniqids[10] = random.sample(dict_of_num_FUP_to_list_of_uniqids[10], 2)
    dict_of_num_FUP_to_list_of_uniqids[11] = random.sample(dict_of_num_FUP_to_list_of_uniqids[11], 2)
    dict_of_num_FUP_to_list_of_uniqids[12] = random.sample(dict_of_num_FUP_to_list_of_uniqids[12], 2)
    
    temp_dict = dict()
    for key, value in dict_of_num_FUP_to_list_of_uniqids.items():
        for uniqid in value:
            temp_dict[uniqid] = key
            
    id_to_num_FUP_dict_neg = temp_dict
    
    
    #Function to graph the probabilties
    def draw_probability_FUP_RNN(dict_of_id_to_num_FUP, dict_of_probabilities_progression, color, name):

        number_of_FUP_set = set(dict_of_id_to_num_FUP.values())

        plt.rc('xtick', labelsize=6)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=8)    # fontsize of the tick labels
        
        fig_height = len(number_of_FUP_set)*11.7/8
        fig_width = 8.3/2 if len(number_of_FUP_set) < 9 else 8.3/1.1
        

        fig = plt.figure(layout="constrained", figsize=(fig_width, fig_height))
        
        
        subfigs = fig.subfigures(len(number_of_FUP_set), 1) 
        
        for subfig_num, number_of_FUP in enumerate(number_of_FUP_set):
            ids_to_draw = [id_ for id_ in dict_of_id_to_num_FUP if  dict_of_id_to_num_FUP[id_] == number_of_FUP ]
            number_of_pics = len(ids_to_draw)

            axs = subfigs[subfig_num].subplots(1,number_of_pics,sharey=True)

            if len(ids_to_draw)==1:
                axs = [axs]

            for i, ax, uniqid in zip(range(1, number_of_pics+1), axs, ids_to_draw):
                

                probs = dict_of_probabilities_progression[uniqid]
                
                
                ax.plot(range(1, len(probs)+1), probs, color=color)

                ax.scatter(range(1, len(probs)+1), probs, color=color)

                ax.set_ylim(bottom = 0, top = 1)

                ax.set_xticklabels(["FUP"+" "+str(lab) for lab in range(1, len(probs)+1)])
                
                if number_of_FUP == 0:
                    ax.set_xticklabels(["FUP"+" "+str(lab) for lab in range(0, len(probs)+1)])    
                
                ax.set_xticks(range(1, len(probs)+1))

                ax.axhline(y=0.5, xmin = 0, xmax=len(probs), color='gray', linestyle="dotted", alpha=0.5)

                ax.set_xlim(0.5, len(probs) + 0.5)
                
                for x, y in zip(range(1, len(probs)+1), probs):
                    ax.text(x-0.28, y+0.07, f'{y:.3}', fontdict={'fontsize':6})
                
                if i == 1:
                    ax.set_ylabel("$\it{P}$ $(bleeding)$")
                    
        fig.savefig(f"./results_pics/FUP_RNN_probabilities_{name}.pdf")  

    
    draw_probability_FUP_RNN(dict_of_id_to_num_FUP=id_to_num_FUP_dict_neg, dict_of_probabilities_progression=probability_progression, color='blue', name="non_bleeders")               
    draw_probability_FUP_RNN(dict_of_id_to_num_FUP=id_to_num_FUP_dict_pos, dict_of_probabilities_progression=probability_progression, color='red', name="bleeders")   

def get_clinical_scores_performance():
    
    _, FUPS_dict, _, baseline_dataframe, target_series = get_formatted_Baseline_FUP(mode="Raw")
    

    #Divide all the data into training and testing portions (two stratified parts) where the testing part consists 30% of the data
    #Note, the training_val and testing portions are generated deterministically.
    #training_val_indeces, testing_indeces = divide_into_stratified_fractions(FUPS_dict, target_series.copy(), fraction=0.3)
    
    #Despite divide_into_stratified_fractions being deterministic, small modifications in data, will change the order of indeces
    #To make the code reproducible, even after modification in dataset, we will use the predefined set of indeces
    training_val_indeces, testing_indeces = get_predefined_training_testing_indeces_30_percent()
    
    for model_name in ['CHAP','ACCP','RIETE','VTE-BLEED','HAS-BLED','OBRI']:
        #Create a dir to save the result of experiment
        if not os.path.exists(f"./keras_tuner_results/{model_name}"):
            os.makedirs(f"./keras_tuner_results/{model_name}")

        #Record the training_val and testing indeces
        record_training_testing_indeces(model_name, training_val_indeces, testing_indeces)

        #Using the indeces for training_val, extract the training-val data from all the data in both BASELINE and follow-ups
        #train_val data are used for hyperparameter optimization and training.
        baseline_test_X, _, test_y = get_X_y_from_indeces(indeces = testing_indeces, 
                                                        baseline_data = baseline_dataframe, 
                                                        FUPS_data_dic = FUPS_dict, 
                                                        all_targets_data = target_series)
        
        #Change the names of the columns in the baseline data to make them compatible with the clinical score
        abb_to_long_dic = get_abb_to_long_dic(instructions_dir=instruction_dir, CRF_name="BASELINE")
        baseline_test_X.columns = [abb_to_long_dic[col] if col in abb_to_long_dic else col for col in baseline_test_X.columns]

        #Create the score object
        mychap_clf = ClinicalScore(model_name)

        #Fitting doesn't do anything special, it is just rerequired for the class
        mychap_clf = mychap_clf.fit(baseline_test_X, test_y)

        #Record exactly what are the predictions for each sample on the test dataset
        y_pred_classes = mychap_clf.predict(baseline_test_X)
        y_pred = mychap_clf.predict(baseline_test_X, mode="score")
        
        number_of_FUP = [len(FUPS_dict[uniqid]) for uniqid in list(test_y.index)]
        record_dict = {"uniqid":list(test_y.index),"FUP_numbers":number_of_FUP, "y_actual":test_y.values, "y_pred":y_pred,
                    "y_pred_classes":y_pred_classes}

        #Save the detailed results
        pd.DataFrame(record_dict).to_csv(f"keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")    

def statistical_comparison_AUROC_AUPRC():
    """Corrects and visualizes the results of the Delong test. Perform comparison for AUPRC.
    """
    
    #Note: Delong test must be done in R. 
    if not os.path.isfile("./keras_tuner_results/delong_test_not_corrected.csv"):
        raise(ValueError("Delong test must be done in R prior to running this function."))
    
    #Read the data produced from R
    delong_data = pd.read_csv("./keras_tuner_results/delong_test_not_corrected.csv", index_col=0)
    
    #Modify the names slightly as they are in the paper.
    delong_data.columns = [model_paper_dic[val] for val in delong_data.columns]
    delong_data.index = [model_paper_dic[val] for val in delong_data.index]
    
    #Find the order of models from highest to lowest AUROC
    #This is to add top left diagonal heatmap
    roc_auc_dict = dict()
      
    for model_name in model_dict["All_models"]:
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
        
        y_pred = np.array(detailed_test_res["y_pred"])
        y_actual = np.array(detailed_test_res["y_actual"])
        tp_list, fp_list, fn_list, tn_list, _, _, _ = calc_PR_ROC_from_y_pred(y_pred, y_actual)
        
        ROC_AUC, _ = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
        
        roc_auc_dict[model_name] = ROC_AUC

    roc_auc_dict = dict(sorted(roc_auc_dict.items(), key=lambda item: item[1], reverse=True))
    row_order = list(roc_auc_dict.keys())
    row_order = [model_paper_dic[val] for val in row_order]
    
    #Correct the order of stat test results
    delong_data = delong_data.reindex(row_order)
    delong_data = delong_data[row_order[::-1]] #Reverse the row order
        
    #Correct the p-values
    corected_delong_p_values, multitest_method = correct_p_values(delong_data, multitest_correction='fdr_bh')
    
    #Remove last row and last column (redundant info)
    corected_delong_p_values = corected_delong_p_values.drop(corected_delong_p_values.columns[-1], axis=1)
    corected_delong_p_values = corected_delong_p_values.drop(corected_delong_p_values.index[-1], axis=0)  
    
    #Save the data
    plot_p_value_heatmap(corected_delong_p_values, title="Delong's Test",
                         save_path="./results_pics/", multitest_correction=multitest_method, 
                     omnibus_p_value=f"None", plot_name="Delong")
    
    ################################################
    
    AUPRC_dics = dict()
    
    AUPRC_df_pvalues = pd.DataFrame(index = model_dict["All_models"], columns = model_dict["All_models"])
    
    for model_name in model_dict["All_models"]:
        detailed_test_res = pd.read_csv(f"./keras_tuner_results/{model_name}/{model_name}_detailed_test_results.csv")
            
        y_pred = np.array(detailed_test_res["y_pred"])
        y_actual = np.array(detailed_test_res["y_actual"])
        tp_list, fp_list, fn_list, tn_list, _, _, _ = calc_PR_ROC_from_y_pred(y_pred, y_actual)
        
        _, PR_AUC = get_auc_using_tf(tn_list, tp_list, fn_list, fp_list)
        
        AUPRC_dics[model_name] = PR_AUC
        
    for model1 in AUPRC_dics.keys():
        for model2 in AUPRC_dics.keys():
            pval = compare_estimates_t_statistics(value1=AUPRC_dics[model1],
                                                  value2=AUPRC_dics[model2],
                                                  num_positives=sum(y_actual),
                                                  total_sample_size=len(y_actual),
                                                  mode="binomial")
            
            AUPRC_df_pvalues.loc[model1, model2] = pval
    
    
    #Modify the names slightly as they are in the paper.
    AUPRC_df_pvalues.columns = [model_paper_dic[val] for val in AUPRC_df_pvalues.columns]
    AUPRC_df_pvalues.index = [model_paper_dic[val] for val in AUPRC_df_pvalues.index]  
    
    #Order the AUPRCs in descending order
    AUPRC_dics = dict(sorted(AUPRC_dics.items(), key=lambda item: item[1], reverse=True))
    row_order = list(AUPRC_dics.keys())
    row_order = [model_paper_dic[val] for val in row_order]
    
    #Correct the order of stat test results
    AUPRC_df_pvalues = AUPRC_df_pvalues.reindex(row_order)
    AUPRC_df_pvalues = AUPRC_df_pvalues[row_order[::-1]] #Reverse the column order
    
    
    #Correct the p-values
    corected_AUPRC_p_values, multitest_method = correct_p_values(AUPRC_df_pvalues, multitest_correction='fdr_bh')
    
    #Remove last row cols
    corected_AUPRC_p_values = corected_AUPRC_p_values.drop(corected_AUPRC_p_values.columns[-1], axis=1)
    corected_AUPRC_p_values = corected_AUPRC_p_values.drop(corected_AUPRC_p_values.index[-1], axis=0)  
    
    #Save the data
    plot_p_value_heatmap(corected_AUPRC_p_values, title="AUPRC t-statistics",
                         save_path="./results_pics/", multitest_correction=multitest_method, 
                     omnibus_p_value=f"None", plot_name="AUPRC_t_stats")        
        
                   
def main():
    
    create_feature_sets_json()
    
    get_clinical_scores_performance()
    
    plot_iterated_k_fold_scores()
    
    plot_validations_train_test()
    
    plot_ROC_PR()  
    
    plot_confusion_matrix()
    
    extract_the_best_hps(number_of_best_hps=200)
    
    get_tn_fp_fn_tn()

    save_deatiled_metrics_test()
    
    plot_FUP_count_density()    
        
    plot_permutaion_feature_importance_RNN_FUP()    
    
    mcnemar_analysis()
    
    plot_FUP_RNN_probabilities_output()
    
    statistical_comparison_AUROC_AUPRC()
    
if __name__=="__main__":
    main()