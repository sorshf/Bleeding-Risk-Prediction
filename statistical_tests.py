#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Soroush Shahryari Fard
# =============================================================================
"This module performs all the statistics."
# =============================================================================
# Imports

from statsmodels.stats import multitest
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker
import scipy


def correct_p_values(p_value_df, multitest_correction, alpha=0.05):
    """Performs p-value correction on the top left diagonal of a square dataframe (with equal number of cols and rows).
    Note: The p-values in the bottom right diagonal of the dataframe will not be affected.

    Args:
        p_value_df (pandas.df): Pandas dataframe of the p-values
        multitest_correction (str): The method used to correct multiple hypothesis testing.
                                    https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
        alpha (float, optional): p-value significance level. Defaults to 0.05.

    Returns:
        corrected_stat_test_results (pd.DataFrame): Pandas dataframe of the corrected p-values.
        multitest_correction (str): The string of the type of multitest used to correct p-values.
    """        
    #Create a list of ordered p-values
    ordered_p_value_lists = []
    for j in range(len(p_value_df.columns)):
        for i in range(len(p_value_df.index)):
            if i < (len(p_value_df.columns) - j - 1):
                ordered_p_value_lists.append(float(p_value_df.iloc[i,j]))
    print(f"There are {len(ordered_p_value_lists)} p-values to be corrected")
    #Correct the p-values 
    _, corrected_p_values, _, _ = multitest.multipletests(pvals=ordered_p_value_lists, alpha=alpha, 
                                                          method=multitest_correction)

    #Create a new pandas df with corrected p-values
    corrected_stat_test_results = p_value_df.copy()
    index=0
    for j in range(len(corrected_stat_test_results.columns)):
        for i in range(len(corrected_stat_test_results.index)):
            if i < (len(corrected_stat_test_results.columns) - j - 1):
                corrected_stat_test_results.iloc[i,j] = corrected_p_values[index]
                index += 1
                
    return corrected_stat_test_results, multitest_correction

def plot_p_value_heatmap(p_value_df, title, save_path, multitest_correction,
                         omnibus_p_value, plot_name, p_value_threshold=0.05):
    """Plot heatmap of p-values for the top left diagonal of a squared dataframe.

    Args:
        p_value_df (pandas.df): pandas dataframe of the corrected p-values.
        title (str): The metric we are comparing that becomes the title of the heatmap.
        save_path (str): The path to where the figure needs to be saved.
        multitest_correction (str): The name of the multitest correction used.
        omnibus_p_value (str): The result of the ombnibus test which also becomes the x-axis label.
        plot_name (str): The name of the plot to save.
        p_value_threshold (float, optional): the significance level threshold. Defaults to 0.05.
    """
    
    #Plot the hitmap for the p-values
    fig, ax = plt.subplots(figsize=(13,7))

    
    #Create the mask for the bottom right of the dataframe
    mask = np.zeros_like(p_value_df, dtype=bool)
    mask[np.tril_indices_from(mask, -1)] = True
    mask = np.fliplr(mask)

    #Color map for the squares
    cmap = (colors.ListedColormap(['#20e84f', '#f2d666']))
    bounds = [min(p_value_df.astype('float').values.flatten())*(0.1), p_value_threshold, 1.0]
    
    #In case the smallest p-value was bigger than the p_value
    if min(p_value_df.astype('float').values.flatten()) > p_value_threshold:
        bounds = [p_value_threshold*(0.1), p_value_threshold, 1.0]
        
    #Format exponent to 10 *
    def format(value, pos=0):
        value_ = f"{value:.2e}"
        value_ =  value_.replace("e", "×10^{")
        value_ = value_ + "}"
        return f'${value_}$'
        
    #Plot the heatmap
    norm = colors.BoundaryNorm(bounds, cmap.N)
    labels = np.asarray([format(value) for value in p_value_df.astype(float).values.flatten()]).reshape(p_value_df.shape)   
    fontsize=8.5
        
    sns.heatmap(p_value_df.astype(float),annot=labels, cmap=cmap, norm=norm,fmt="", annot_kws={"fontsize":fontsize}, 
                square=False,linewidths=.7, ax=ax, cbar=True, cbar_kws={'label':"$\it{p}$-value", "shrink": 0.75},
                mask=mask)

    #Format the color bar
    colorbar = ax.collections[0].colorbar
    formatter = ticker.FuncFormatter(format)
    colorbar.ax.yaxis.set_major_formatter(formatter)
    colorbar.ax.set_position([0.65, 0.20, 0.03, 0.4])  # [left, bottom, width, height]

    ax.xaxis.tick_top() #Add the x-axis label to the top
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(f"{omnibus_p_value}\n{multitest_correction}", labelpad=10)
    
    #Save the figure
    fig.savefig(f"{save_path}{plot_name}_{multitest_correction}.pdf", bbox_inches='tight')

def confidence_interval_estimation(value, num_positives, total_sample_size, mode):
    """Calculates 95% confidence interval for a point estimate such as AUPRC using binomial or logit interval methods.
        https://pages.cs.wisc.edu/~boyd/aucpr_final.pdf

    Args:
        value (float): Value of the point estimate.
        num_positives (int): Number of positive samples in the dataset where the value was calculated.
        total_sample_size (int): Total number of samples in the dataset where the value was calculated.
        mode (str): Either 'logit' or 'binomial'.

    Returns:
        float, float, float, flaot: Value, Lower-CI, Upper-CI, Standard Error
    """

    if mode == "logit":
        logit_AUPRC = np.log(value/(1-value))

        se_logistic_scale = 1/np.sqrt(num_positives * value * (1 - value)) #SE is on the logistic scale.

        lower_ci = np.exp(logit_AUPRC-(1.96*se_logistic_scale))/(1+np.exp(logit_AUPRC-(1.96*se_logistic_scale)))
        upper_ci = np.exp(logit_AUPRC+(1.96*se_logistic_scale))/(1+np.exp(logit_AUPRC+(1.96*se_logistic_scale)))

        se_from_cis = (upper_ci-lower_ci) / 3.92
        
        return value, lower_ci, upper_ci, se_from_cis
        
    elif mode == "binomial":
        se = np.sqrt(value*(1-value)/total_sample_size)
        lower_ci = value - (1.96 * se)
        upper_ci = value + (1.96 * se)
        
        return value, lower_ci, upper_ci, se
        
    else:
        raise(ValueError(f"Mode {mode} is undefined. Mode should be 'logit' or 'binomial'. "))    


def compare_estimates_t_statistics(value1, value2, num_positives, total_sample_size, mode):
    """Compare two point estimates using t-statistics.
    #https://stats.libretexts.org/Courses/Luther_College/Psyc_350%3ABehavioral_Statistics_(Toussaint)/08%3A_Tests_of_Means/8.03%3A_Difference_between_Two_Means

    Args:
        value1 (float): First value of the metric of interest.
        value2 (float): Second value of the metric of interest.
        num_positives (int): Number of positive samples in the dataset where the value was calculated.
        total_sample_size (int): Total number of samples in the dataset where the value was calculated.
        mode (str): Either 'logit' or 'binomial'.

    Returns:
        float: p-value of the difference between the two values.
    """
        
    _, _, _, se_value1 = confidence_interval_estimation(value=value1, num_positives=num_positives, total_sample_size=total_sample_size, mode=mode)
    _, _, _, se_value2 = confidence_interval_estimation(value=value2, num_positives=num_positives, total_sample_size=total_sample_size, mode=mode)
    
    #https://www.bmj.com/about-bmj/resources-readers/publications/statistics-square-one/5-differences-between-means-type-i-an
    #https://online.stat.psu.edu/stat100/lesson/9/9.3
    #https://online.stat.psu.edu/stat100/lesson/10/10.3
    #Standard error of difference between means.
    t_stat = (value1 - value2)/np.sqrt( (se_value1**2) + (se_value2**2) )
        
    dof = total_sample_size - 1
    
    pval = scipy.stats.t.sf(np.abs(t_stat), dof)*2
    
    return pval    