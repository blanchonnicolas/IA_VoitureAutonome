# %%
import numpy as np
import pandas as pd
import matplotlib
#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#import seaborn as sns
#sns.set_theme()
#sns.set_palette("Greens_d")
#sns.light_palette("seagreen", as_cmap=True)
#sns.color_palette("Greens_d")

#import plotly.express as px
import tarfile
import json
import csv
import random
import time
import re

import cv2

from datetime import datetime

# Import custom helper libraries
import os
#from os import listdir, path
#from os.path import isfile, join, splitext

import sys
#import data.helpers as data_helpers
#import visualization.helpers as viz_helpers

from joblib import dump, load
import pickle

#from PIL import Image

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# %%
# function
def Test_Imported_Functions():
    print("Functions have been properly imported !")

# %% [markdown]
# ## Load data

# %%
def load_nrows_data(DATA_URL, nrows, DATASET_COLUMNS):
    data_loaded = pd.read_csv(DATA_URL, nrows=nrows, names=DATASET_COLUMNS)
    return (data_loaded)

def load_all_data(DATA_URL, DATASET_COLUMNS):
    data_loaded = pd.read_csv(DATA_URL, names=DATASET_COLUMNS)
    return (data_loaded)

def load_formatted_data(DATA_URL, DATASET_ENCODING, DATASET_COLUMNS):
    data_loaded = pd.read_csv(DATA_URL, encoding=DATASET_ENCODING, names=DATASET_COLUMNS)
    return (data_loaded)


# %% [markdown]
# ## Clean data

# %%
def clean_data(data): 
    #Remove rows where important information are missing:
    data_cleaned = data.dropna(axis = 0, how='all')
    #Clean duplicates
    data_cleaned = data_cleaned.drop_duplicates()
    #Change content in lowercase
    data_cleaned = data_cleaned.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)
    #Filter order_dataset
    return (data_cleaned)

# %% [markdown]
# ## Plot Dataframe

# %%


# %%
def colors_from_values_integer(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

# def colors_from_values(values, palette_name):
#     pal = sns.color_palette(palette_name, len(values))
#     rank = values.argsort().argsort()   # http://stackoverflow.com/a/6266510/1628638
#     palette=np.array(pal[::-1])[rank]
#     return (palette)

def colors_from_values_float(values: pd.Series, palette_name:str, ascending=True):
    # convert to indices
    values = values.sort_values(ascending=True).reset_index()
    indices = values.sort_values(by=values.columns[0]).index
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

# %%
#Function to plot Fill ratio in specified columns
def plot_fill_ratio(data: pd.DataFrame, colunms_selected: list):
    data_fill_ratio = pd.DataFrame(columns=['column_name', 'null_count', 'notnull_count'])
    data_fill_ratio.drop(data_fill_ratio.index, inplace=True)        
    for col in colunms_selected: 
        null_count = data[col].isna().sum()
        notnull_count = data[col].notna().sum()
        new_row = pd.DataFrame({'column_name':[col], 'null_count':[null_count], 'notnull_count':[notnull_count]})
        data_fill_ratio = pd.concat([data_fill_ratio, new_row], ignore_index = None, axis = 0)
    data_fill_ratio_study = pd.melt(data_fill_ratio.reset_index(), id_vars=['column_name'], value_vars=['null_count', 'notnull_count'])
    fig, ax = plt.subplots(figsize=(16,10))
    #ax = sns.barplot(data=data_fill_ratio_study, x='value', y='column_name', hue='variable')
    
    ax = sns.barplot(data=data_fill_ratio_study, x='value', y='column_name', hue='variable', palette="Greens_d")
    ax.set_title('Null and NotNull Count per columns in dataframe')
    plt.show()
    data_fill_ratio_study.drop(data_fill_ratio_study.index, inplace=True)
    return(data_fill_ratio)



# %%
#Function to plot occurence by value present in specified column
def plot_occurence_line(data: pd.DataFrame, colunm_name):
    fig = px.line(data[colunm_name].value_counts())
    fig.update_layout(
        title_text=f"Number of occurence by {colunm_name} .\nTOTAL = {len(data[colunm_name])}",
        width=900,
        height=600,
        #markers=True,
    )
    fig.show()

# %%
#Function to plot distribution of dates
def plot_peryearmonth(data: pd.DataFrame, date_column, plot_hue: bool, hue_column):
    data['date_yearmonth'] = pd.to_datetime(data[date_column]).dt.to_period('M')
    plt.figure(figsize=(15,10))
    if (plot_hue == True):
        ax1 = sns.countplot(x="date_yearmonth", data=data.sort_values('date_yearmonth'), hue=hue_column, palette="Greens_d")
    else:
        ax1 = sns.countplot(x="date_yearmonth", data=data.sort_values('date_yearmonth'), palette="Greens_d")
    ax1.set_title(f'Distribution of {date_column} per month')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.show()

# %%
def pairplot_columns(data: pd.DataFrame, colunms_selected: list, plot_hue: bool, hue_column):
    #fig, ax = plt.subplots(figsize=(15,10))
    if (plot_hue == True):
        ax = sns.pairplot(data[colunms_selected], 
                             hue=hue_column, 
                             hue_order=sorted(data[hue_column].unique(),
                             reverse=True)
                            )
    else:
        ax = ax=sns.pairplot(data[colunms_selected]
                            )
    ax.fig.suptitle(f'Pairplot on selected columns')
    plt.title(f'Pairplot on selected columns {colunms_selected}')
    plt.show()

# %%
#Function to plot PIE Chart of n tops values in dataframe
def plot_ntops_pie(data: pd.DataFrame, colunm_name, ntops: int, plot_others: bool, plot_na: bool):
    podium_tops = pd.DataFrame(data[colunm_name].value_counts(dropna=True, sort=True).head(ntops))
    if (plot_others == True):
        remainings_counts = sum(data[colunm_name].value_counts(dropna=True)[ntops:])
        remainings_below = pd.DataFrame({colunm_name : [remainings_counts]}, index=['others'])
        podium_tops = pd.concat([podium_tops, remainings_below], ignore_index = None, axis = 0)
    if (plot_na == True):
        na_counts = data[colunm_name].isna().sum()
        remainings_na = pd.DataFrame({colunm_name : [na_counts]}, index=['NAN'])
        podium_tops = pd.concat([podium_tops, remainings_na], ignore_index = None, axis = 0)
    
    
    #Définir la taille du graphique
    plt.figure(figsize=(8,8))
    #Définir lae type du graphique, ici PIE CHart avec en Labels l'index du nom des libelle
    #l'autopct sert ici à afficher le % calculé avec 1 décimal derriere la virgule
    plt.pie(podium_tops[colunm_name], labels=podium_tops.index, autopct='%1.1f%%')
    #Afficher la légende en dessous du graphique au centre
    plt.legend(loc='upper left', bbox_to_anchor=(0.1, -0.01), fancybox=True, shadow=None, ncol=2)
    plt.title(f"{ntops} most presents values identified in column {colunm_name} .\nTOTAL unique = {len(data[colunm_name].unique())}")
    #Afficher le graphique
    plt.show()
    return(podium_tops)



# %%
def plot_ntops_bar(data: pd.DataFrame, colunm_name, ntops: int, plot_others: bool, plot_na: bool):
    podium_tops = pd.DataFrame(data[colunm_name].value_counts(dropna=True, sort=True).head(ntops))
    if (plot_others == True):
        remainings_counts = sum(data[colunm_name].value_counts(dropna=True)[ntops:])
        remainings_below = pd.DataFrame({colunm_name : [remainings_counts]}, index=['others'])
        podium_tops = pd.concat([podium_tops, remainings_below], ignore_index = None, axis = 0)
    if (plot_na == True):
        na_counts = data[colunm_name].isna().sum()
        remainings_na = pd.DataFrame({colunm_name : [na_counts]}, index=['NAN'])
        podium_tops = pd.concat([podium_tops, remainings_na], ignore_index = None, axis = 0)
    #podium_tops = podium_tops.reset_index(drop=True)
    #Définir la taille du graphique
    fig, ax = plt.subplots(figsize=(15,10))
    #Définir lae type du graphique, ici BARPLOT avec en Labels l'index du nom des libelle
    ax = sns.barplot(data=podium_tops, x=podium_tops.index, y=colunm_name, palette=colors_from_values_integer(podium_tops[colunm_name], "Greens_d"))
    plt.title(f"{ntops} most presents values identified in column {colunm_name} .\nTOTAL unique = {len(data[colunm_name].unique())}")
    #Afficher le graphique
    plt.show()
    return(podium_tops)


# %%
#Create function that study boxplot
def plot_boxplot(data: pd.DataFrame, x_axis, colunms_selected: list, plot_outliers: bool): 
    for col in colunms_selected:
        sns.set()
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.boxplot(x=x_axis, 
                    y=col, # column is chosen here
                    data=data,
                    #order=["a", "b"],
                    showfliers = plot_outliers,
                    showmeans=True,
                    )  
        sns.despine(offset=10, trim=True) 
        plt.show()
        

# %%
#Create function that study histogramme
def plot_histogramme(histo_data: pd.DataFrame, column_value, colunms_group, plot_outliers: bool): 
    fig, ax = plt.subplots(figsize=(15, 5))
    #Plot the distribution
    ax = sns.displot(data=histo_data, x=column_value, hue=colunms_group)
    ax.move_legend(ax1, "upper right", bbox_to_anchor=(.55, .45), title=f'histogramme of {column_value}')
    plt.title(f"Distribution of {column_value} values")
    #plt.legend(loc='upper right')
    plt.ylabel("Count")
    plt.xlabel(f"{column_value} ranges")
    plt.show()

# %% [markdown]
# ## Réduction de dimension
# ### ACP (A Vérifier)

# %%
#PCA functions:
#Functions below are used for ACP
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(8,8))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='r')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

# %%
#Functions below are used for ACP
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(8,8))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
                #plt.scatter(centres_reduced[:, d1], centres_reduced[:, d2], alpha=alpha, marker='x', s=100, linewidths=2,color='k', zorder=10)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                    #plt.scatter(centres_reduced[:, d1], centres_reduced[:, d2], alpha=alpha, marker='x', s=100, linewidths=2,color='k', zorder=10)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) / 1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)


# %%
#Functions below are used for ACP            
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)


# %%
def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster==i])
    
    # Create the plot
    fig = plt.figure(figsize=(12, 15))
    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):    
        plt.subplot(num_clusters, 1, i+1)
        for j,c in enumerate(cluster_points): 
            if i!= j:
                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])
        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])

        # Stagger the axes
        ax=plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)        


# %%
def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax=plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20) 

# %%
#Functions below are used for Data Clustering    
def plot_dendrogram(linked, names):
    plt.figure(figsize=(10,15))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        linked,
        labels = names,
        orientation = "left",
        show_leaf_counts=True
    )
    plt.show()


# %% [markdown]
# ### Matrice de Confusion

# %%
#Fonction pour le graphe de Matrice de Confusion
def matrix_pred_model(model, model_name, y_test, y_pred, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.title(f"Matrice de confusion de {model_name}")
    sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,cmap = 'Greens',fmt="d",cbar=False)
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe initiale")
    plt.show()

# %%
def plot_roc_auc_curve(model_name, fpr, tpr, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.title(f"ROC Curve for {model_name}")
    plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=model_name)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    # show the plot
    plt.show()

# %% [markdown]
# ### Determiner le seuil de décision (Classification binaire)

# %%
def plot_threshold_scores(data: pd.DataFrame, optimized_seuil):
    plt.figure(figsize=(10,5))
    plt.plot(data['seuil'], data['FBeta-Score'], color='coral', lw=2, label='FBeta-Score')
    plt.plot(data['seuil'], data['Precision_score'], color='cyan', lw=2, label='Precision_score')
    plt.plot(data['seuil'], data['Accuracy_score'], color='blue', lw=2, label='Accuracy_score')
    plt.plot(data['seuil'], data['Recall_score'], color='green', lw=2, label='Recall_score')
    plt.plot(data['seuil'], data['F1_score'], color='red', lw=2, label='F1_score')
    #plt.plot(store_score_thresholds['seuil'], store_score_thresholds['Roc_AUC_score'], color='red', lw=2, label='Roc_AUC_score')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Seuil', fontsize=14)
    plt.ylabel('Scores', fontsize=14)
    plt.legend(loc="upper left")
    plt.title(f'Score optimized for binary classification obtained with threshold = {optimized_seuil}')
    plt.show()


# %%
def decode_sentiment(score, seuil=0.5):
    if score <= seuil:
        label = 0
    elif score > seuil:
        label = 1
    return label

# %%
#Fonction pour retourner les score des modèles

###############################        ATTENTION : INITIALISATION A PREVOIR AVANT UTILISATION     ################
#Initialisation de la table des résultats
#score_column_names = ["Model Type","Model Name","seuil","F1-Score", "Recall_score", "Precision_score", "Accuracy_score"]
#store_score= pd.DataFrame(columns = score_column_names)

# def evaluation(model,model_name,score_column_names,X_test,y_test, seuil = 0.5, binary_transform=False):
#     # On récupère la prédiction de la valeur positive
#     if binary_transform == True:
#         y_prob = model.predict(X_test)
#         y_pred = y_prob
#     else:
#         y_prob = model.predict_proba(X_test)[:,1]
#         y_pred = np.where(y_prob > seuil, 1, 0)
    
#     # On créé un vecteur de prédiction à partir du vecteur de probabilités
#     false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob) # y_prob instead of y_prob #, pos_label=4
#     Roc_AUC_score = auc(false_positive_rate, true_positive_rate)
    
#     F1_score = f1_score(y_test, y_pred)
#     FBeta_score = fbeta_score(y_test, y_pred, average='binary', beta=0.5, pos_label=1) #make_scorer(fbeta_score, beta = 2, pos_label=0 ,average = 'binary')
#     Recall_score = recall_score(y_test, y_pred)
#     Precision_score = precision_score(y_test, y_pred)
#     Accuracy_score = accuracy_score(y_test, y_pred)
    
#     #Plot functions
#     matrix_pred_model(model, model_name, y_test, y_pred) 
#     plot_roc_auc_curve(model_name, false_positive_rate, true_positive_rate)
    
#     score_results = pd.Series([model, model_name, seuil, F1_score, FBeta_score, Recall_score, Precision_score, Accuracy_score, Roc_AUC_score])
#     score_results_stored = pd.DataFrame([score_results.values],  columns = score_column_names)
#     return(score_results_stored)

# def evaluation_to_correct(model,model_name,score_column_names,store_score,X_test,y_test, seuil = 0.5):
#     #Si le seuil n'est pas important
#     y_pred = model.predict(X_test)
#     F1_score = f1_score(y_test, y_pred)#, pos_label=4
#     Recall_score = recall_score(y_test, y_pred) #, pos_label=4
#     Precision_score = precision_score(y_test, y_pred) #, pos_label=4
#     Accuracy_score = accuracy_score(y_test, y_pred)
    
#     matrix_pred_model(model, model_name, y_test,y_pred)  
    
#     #global store_score
#     score_results = pd.Series([model, model_name, seuil, F1_score, Recall_score, Precision_score, Accuracy_score])
#     score_results_stored = pd.DataFrame([score_results.values],  columns = score_column_names)
#     store_score = pd.concat([store_score, score_results_stored], axis=0)
#     return(store_score)

def evaluation(model,model_name,score_column_names,X_test,y_test, seuil = 0.5, binary_predict=False, predict_proba_OK=False):
    # On récupère la prédiction de la valeur positive
    if binary_predict == True:
        y_prob = model.predict(X_test)
        y_pred = y_prob
    elif ((binary_predict == False) and (predict_proba_OK == True)):
        y_prob = model.predict_proba(X_test)[:,1]
        y_pred = np.where(y_prob > seuil, 1, 0)
    elif predict_proba_OK == False:
        y_prob = model.predict(X_test)
        y_pred = np.where(y_prob > seuil, 1, 0)
        y_pred = y_pred.astype(int)
    
    # On créé un vecteur de prédiction à partir du vecteur de probabilités
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob) # y_prob instead of y_prob #, pos_label=4
    Roc_AUC_score = auc(false_positive_rate, true_positive_rate)
    
    F1_score = f1_score(y_test, y_pred)
    FBeta_score = fbeta_score(y_test, y_pred, average='binary', beta=0.5, pos_label=1) #make_scorer(fbeta_score, beta = 2, pos_label=0 ,average = 'binary')
    Recall_score = recall_score(y_test, y_pred)
    Precision_score = precision_score(y_test, y_pred)
    Accuracy_score = accuracy_score(y_test, y_pred)
    
    #Plot functions
    matrix_pred_model(model, model_name, y_test, y_pred) 
    plot_roc_auc_curve(model_name, false_positive_rate, true_positive_rate)
    
    score_results = pd.Series([model, model_name, seuil, F1_score, FBeta_score, Recall_score, Precision_score, Accuracy_score, Roc_AUC_score])
    score_results_stored = pd.DataFrame([score_results.values],  columns = score_column_names)
    return(score_results_stored)

# %%
def plot_model_result(data: pd.DataFrame, score, model_name):
    #Définir la taille du graphique
    fig, ax = plt.subplots(figsize=(15,10))
    #Définir lae type du graphique, ici BARPLOT avec en Labels l'index du nom des libelle
    ax = sns.barplot(data=data, y=model_name, x=score, palette=colors_from_values_float(data[score], "Greens_d"))
    ax.set_xlim((data[score].min() - 0.05), (data[score].max() + 0.02))
    #ax = sns.barplot(data=data, x=model_name, y=score)
    plt.title(f"Score {score} max value is {round(data[score].max(), 2)} computed in model {data.loc[data[score] == data[score].max(), model_name]}")
    #Afficher le graphique
    plt.show()

# %%
def plot_history(history):
    acc=history.history['accuracy']
    val_acc=history.history['val_accuracy']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    x=range(1, len(acc) + 1)
    
    plt.figure(figsize=(15,5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'g', label='Training accuracy')
    plt.plot(x, val_acc, 'c', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'g', label='Training loss')
    plt.plot(x, val_loss, 'c', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()

# %%
def display_learning_curves(history):
    acc = history.history["mean_io_u"]
    val_acc = history.history["val_mean_io_u"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(30)

    fig = plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label="train mean_io_u")
    plt.plot(epochs_range, val_acc, label="validataion mean_io_u")
    plt.title("mean_io_u")
    plt.xlabel("Epoch")
    plt.ylabel("mean_io_u")
    plt.legend(loc="lower right")

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="validataion loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.show()

# %%
def display_learning_curves_iou(history):
    iou = history.history["iou_score"]
    val_iou = history.history["val_iou_score"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range=range(1, len(iou) + 1)
    #epochs_range = range(n_epochs)

    fig = plt.figure(figsize=(15,5))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, iou, label="train iou_score")
    plt.plot(epochs_range, val_iou, label="validataion iou_score")
    plt.title("iou_score")
    plt.xlabel("Epoch")
    plt.ylabel("iou_score")
    plt.legend(loc="upper left")

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="validataion loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.suptitle(f"IoU Socre and Loss evolution during {model_test.name} training, (stopped by callback at epochs : {len(iou)})  ")
    #fig.tight_layout()
    plt.show()

# %%
def display_learning_curves_iou_dice(history, model_name):
    iou = history.history["iou_score"]
    val_iou = history.history["val_iou_score"]

    fscore = history.history["f1-score"]
    val_fscore = history.history["val_f1-score"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range=range(1, len(iou) + 1)
    #epochs_range = range(n_epochs)

    fig = plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.plot(epochs_range, iou, label="train iou_score")
    plt.plot(epochs_range, val_iou, label="validataion iou_score")
    plt.title("iou_score")
    plt.xlabel("Epoch")
    plt.ylabel("iou_score")
    plt.legend(loc="upper left")

    plt.subplot(1,3,2)
    plt.plot(epochs_range, fscore, label="train Dice coeff")
    plt.plot(epochs_range, val_fscore, label="validataion Dice coeff")
    plt.title("f1-score or Dice coeff")
    plt.xlabel("Epoch")
    plt.ylabel("f1-score")
    plt.legend(loc="upper left")

    plt.subplot(1,3,3)
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="validataion loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.suptitle(f"IoU Socre, Dice Coeff and Loss evolution during {model_name} training, (stopped by callback at epochs : {len(iou)})  ")
    #fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Specific For Image project 8

# %%
def list_files(directory):
    #From Data path, list folders and files and store information in dataframe
    train_val_test = []
    cities = []
    data = []
    for phase in sorted(os.listdir(directory)):
        train_val_test.append((phase))
        for city in sorted(os.listdir(os.path.join(directory, phase))):
            cities.append((city))
            for files in sorted(os.listdir(os.path.join(directory, phase, city))):
                filename_path = os.path.join(directory, phase, city, files)
                data.append((phase, city, files, filename_path))       
    return (pd.DataFrame(data, columns=['Phase', 'City', 'File', 'Path']))

# %%
def get_info_from_filename(data: pd.DataFrame):
    data['File_id1'] = data['File'].str.split('_').str[1]
    data['File_id2'] = data['File'].str.split('_').str[2]
    data['File_type'] = data['File'].str.split('_').str[3]
    data['File_unique_index'] = data['City'] + '_' + data['File_id1'] + '_' + data['File_id2']
    return data

# %%
def link_images_masks(data_masks: pd.DataFrame, data_images: pd.DataFrame):
    """
    Permet de lier les images et les masques dans un seul dataframe
    :param data_images: DataFrame contenant les informations sur les images
    :param data_masks: DataFrame contenant les informations sur les masks
    :return: cleaned DataFrame
    """
    #Keep only lanelIDs from masks
    clean_data_masks = data_masks.loc[data_masks['File'].str.endswith('labelIds.png') == True, :].copy()
    #clean_data_masks = data_masks.loc[(data_masks['File'].str.contains('labelIds') == True), :]

    #Get Info from filename
    clean_data_images = get_info_from_filename(data_images)
    clean_data_masks = get_info_from_filename(clean_data_masks)

    #Merge images and masks dataframes on File_id1 and File_id2
    #data = pd.merge(clean_data_images, clean_data_masks, how='inner', on=['File_id1', 'File_id2'])
    #data = pd.merge(clean_data_images, clean_data_masks, how='inner', on=['File_id1', 'File_id2', 'City', 'Phase'])
    data = pd.merge(clean_data_images, clean_data_masks, how='inner', on=['File_unique_index', 'City', 'File_id1', 'File_id2', 'Phase'])
    return data

# %%
def convert(list):
    return tuple(list)
def count_color_rgb(image):
    all_color = {}
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            color =  convert(image[x,y].tolist())
            if not color in all_color.keys():
                all_color[color] = 1
            else:
                all_color[color] += 1
    return sorted(all_color.items())

def get_image_labelid(path:str):
    """Récupére le mask parfaitement formaté.
    argument:
    - path (type str): chemin d'accès complet ou partiel de l'image "_labelIds.png"
    """
    return convert_mask(cv2.imread(path,0))

# %%
cats = {
	'construction': [11, 12, 13, 14, 15, 16],
	'flat': [7, 8, 9, 10],
	'human': [24, 25],
	'nature': [21, 22],
	'object': [17, 18, 19, 20],
	'sky': [23],
	'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1],
	'void': [0, 1, 2, 3, 4, 5, 6],
}

def convert_mask(img):
    """Cette méthode permet de convertir l'image '_labelids.png' du jeu de données de CityScapes.
    La méthode permet de récupérer l'image au format one_hot_encoder ou au format label_encoder.
    arguments:
    - img (type numpy.array): image du jeu de données CityScapes '...labelids.png' au format numpy  
    return:
    mask (type numpy.array) : mask pour la segmentation sémantique au format label encoder avec les 8 catégories principal (void, flat, construction, object, nature, sky, human, vehicle)
    """
    #print(len(img.shape))
    if len(img.shape) == 3:
        img = np.squeeze(img[:, :, 0])
    else:
        img = np.squeeze(img)
        #print(img.shape)
    mask = np.zeros((img.shape[0], img.shape[1], 8), dtype=np.uint16)
    print(mask.shape)
    for i in range(-1, 34):
        if i in cats['void']: 
            #Logical OR is applied to the elements of mask and labelIds_img. If mask.shape != labelIds_img.shape,.
            mask[:, :, 0] = np.logical_or(mask[:, :, 0], (img == i))
        elif i in cats['flat']:
            mask[:, :, 1] = np.logical_or(mask[:, :, 1], (img == i))
        elif i in cats['construction']:
            mask[:, :, 2] = np.logical_or(mask[:, :, 2], (img == i))
        elif i in cats['object']:
            mask[:, :, 3] = np.logical_or(mask[:, :, 3], (img == i))
        elif i in cats['nature']:
            mask[:, :, 4] = np.logical_or(mask[:, :, 4], (img == i))
        elif i in cats['sky']:
            mask[:, :, 5] = np.logical_or(mask[:, :, 5], (img == i))
        elif i in cats['human']:
            mask[:, :, 6] = np.logical_or(mask[:, :, 6], (img == i))
        elif i in cats['vehicle']:
            mask[:, :, 7] = np.logical_or(mask[:, :, 7], (img == i))
    return np.array(np.argmax(mask, axis=2), dtype='uint8')

# %% [markdown]
# # DataLoader_simple from dataset - Project 8 (Sans Conversion des Masques)

# %%
# DATALOADER - Notre classe hérite de la classe Keras.utils.Sequence
# Elle permet de créer un générateur de données
# Cette classe parente vous assure dans le cas ou vous souhaitez utiliser du calcul parallèle avec vos threads, de garantir de parcourir une seule et unique fois vos données au cours d’une époch
class Dataloader_simple(keras.utils.Sequence):
    """Load data from dataset to build bacthes
    Args:
        dataset : dataframe listing data and paths
        n_sample : to work with a reduced dataset
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle dataset each epoch.
        resize: Boolean,  if `True` resize images and mask to same dimensions
        resize_width & resize_height: New dimensions after resizing
        display: display images when calling dataloader
        phase: allow to set if dataloader shall filter on train, val or test images/masks
        augmentation: Variable defining augmentation of image and mask
        normalization: Boolean,  if `True` normalizes images and masks RGB values ; The range for each individual colour is 0-255
    """
    # We provide one dataset, containing labelIds and 
    def __init__(self, dataset, n_sample, batch_size=1, shuffle=False, resize=None, resize_width=256, resize_height=128, display=None, phase='train', augmentation=None, normalization=None):
        self.n_sample = n_sample
        self.phase = phase
        self.dataset = dataset.loc[dataset['Phase'] == self.phase, :].sample(self.n_sample) #ou dataset si on veut prendre tour le jeu de donnée
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize = resize
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.augmentation = augmentation
        self.normalization = normalization
        self.display = display
        self.on_epoch_end()

    def __getitem__(self, i):
        # collect batch images and masks
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        #print('start barch is ', start)
        #print('stop barch is ', stop)
        images = []
        masks = []
        for j in range(start, stop):
            # Store entire paths to image_file and mask_file variable
            image_file = list(self.dataset['Path_x'])[j]
            mask_file = list(self.dataset['Path_y'])[j]
            unique_index_file = list(self.dataset['File_unique_index'])[j]

            # OpenImages and masks using OpenCV ibrary, and convert in np array
            image = cv2.imread(image_file)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = cv2.imread(mask_file)
            #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # apply resize
            if self.resize:
                dim = (self.resize_width, self.resize_height)
                original_height = image.shape[0]
                original_width = image.shape[1]
                original_channels = image.shape[2]
                image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # resize image
                mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
                new_height = image.shape[0]
                new_width = image.shape[1]
                new_channels = image.shape[2]
                

            # apply display
            if self.display:
                #Display Image and Mask
                plt.figure(figsize=(25, 5))
                plt.subplots_adjust(hspace=0.5)
                plt.subplot(120 + 1 + 0)
                plt.imshow(image)
                plt.subplot(120 + 1 + 1)
                plt.imshow(mask)
                plt.grid(False)
                plt.suptitle(f'Images and Masks {unique_index_file}')
                plt.show()
                if self.resize:
                    print('original height was ', original_height, ' and new height is ', new_height)
                    print('original width was ', original_width, ' and new width is ', new_width)
                    print('original channels was ', original_channels, ' and new channels is ', new_channels)
                    

            # apply normalization : Normalizing RGB values ; The range for each individual colour is 0-255 (as 2^8 = 256 possibilities).
            if self.normalization:
                #normalizedImg = np.zeros((800, 800))
                image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX) #ou np.array(image/255, dtype='uint8')
                #mask = np.array(mask/255, dtype='uint8') # cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX) #mask/255
            
                # apply display
                if self.display:
                    print(f'----------------Normalized Image and Mask {j} from batch collected-----------------')
                    #Display Image and Mask
                    plt.figure(figsize=(25, 5))
                    plt.subplots_adjust(hspace=0.5)
                    plt.subplot(120 + 1 + 0)
                    plt.imshow(image)
                    plt.subplot(120 + 1 + 1)
                    plt.imshow(mask)
                    plt.grid(False)
                    plt.suptitle(f'Normalized Images and Masks {unique_index_file}')
                    plt.show()
            
            # apply augmentations
            if ((self.augmentation is not None) & (self.phase == 'train')):
                # Augment an image
                sample = self.augmentation(image=image, mask=mask)
                augmented_image, augmented_mask = sample['image'], sample['mask']
                # apply display
                if self.display:
                    print(f'----------------Augmented Image and Mask {j} from batch collected-----------------')
                    #Display Image and Mask
                    plt.figure(figsize=(25, 5))
                    plt.subplots_adjust(hspace=0.5)
                    plt.subplot(120 + 1 + 0)
                    plt.imshow(augmented_image)
                    plt.subplot(120 + 1 + 1)
                    plt.imshow(augmented_mask)
                    plt.grid(False)
                    plt.suptitle(f'Augmented Images and Masks {unique_index_file}')
                    plt.show()
                    images.append(augmented_image)
                    masks.append(augmented_mask)
                    
            # append image and mask to batch
            images.append(image)
            masks.append(mask)   
            print(f'---------------- Image and Mask {j} from batch collected in Data Generator-----------------')

        # transpose list of lists
        image_batch = np.stack(images, axis=0) #A confirmer si images ou image
        mask_batch = np.stack(masks, axis=0) #A confirmer si masks ou mask

        #Add 1 dimension to return image_batch and mask_batch in 4 dimensions (batch number, height, width, channels), avec channel = 1 pour les masques et =3 pour les images
        if len(mask_batch.shape) == 3:
            mask_batch = np.expand_dims(mask_batch, axis = 3)

        #print('mask_batch shape is ', mask_batch.shape)
        return image_batch, mask_batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.dataset) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle dataset each epoch"""
        if self.shuffle:
            np.random.shuffle(self.dataset.values)

# %% [markdown]
# # DataLoader_advanced from dataset - Project 8 (Avec Conversion des Masques)

# %%
# DATALOADER - Notre classe hérite de la classe Keras.utils.Sequence
# Elle permet de créer un générateur de données
# Cette classe parente vous assure dans le cas ou vous souhaitez utiliser du calcul parallèle avec vos threads, de garantir de parcourir une seule et unique fois vos données au cours d’une époch
class Dataloader_advanced(keras.utils.Sequence):
    """Load data from dataset to build bacthes
    Args:
        dataset : dataframe listing data and paths
        n_sample : to work with a reduced dataset
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle dataset each epoch.
        resize: Boolean,  if `True` resize images and mask to same dimensions
        resize_width & resize_height: New dimensions after resizing
        display: display images when calling dataloader
        phase: allow to set if dataloader shall filter on train, val or test images/masks
        augmentation: Variable defining augmentation of image and mask
        normalization: Boolean,  if `True` normalizes images and masks RGB values ; The range for each individual colour is 0-255
    """
    # We provide one dataset, containing labelIds and 
    def __init__(self, dataset, n_sample, batch_size=1, shuffle=False, resize=None, resize_width=256, resize_height=128, convert=None, display=None, phase='train', augmentation=None, normalization=None):
        self.n_sample = n_sample
        self.phase = phase
        self.dataset = dataset.loc[dataset['Phase'] == self.phase, :].sample(self.n_sample) #ou dataset si on veut prendre tour le jeu de donnée
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.resize = resize
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.convert = convert
        self.augmentation = augmentation
        self.normalization = normalization
        self.display = display
        self.on_epoch_end()

    CATS = {
        'void': [0, 1, 2, 3, 4, 5, 6],
        'flat': [7, 8, 9, 10],
        'construction': [11, 12, 13, 14, 15, 16],
        'object': [17, 18, 19, 20],
        'nature': [21, 22],
        'sky': [23],
        'human': [24, 25],
        'vehicle': [26, 27, 28, 29, 30, 31, 32, 33,-1]
    }
    
    def _convert_mask(self,img):
        if len(img.shape) == 3:
            img = np.squeeze(img[:, :, 0])
        else:
            img = np.squeeze(img)
        mask = np.zeros((img.shape[0], img.shape[1], 8),dtype=np.uint8)
        for i in range(-1, 34):
            if i in self.CATS['void']:
                mask[:,:,0] = np.logical_or(mask[:,:,0],(img==i))
            elif i in self.CATS['flat']:
                mask[:,:,1] = np.logical_or(mask[:,:,1],(img==i))
            elif i in self.CATS['construction']:
                mask[:,:,2] = np.logical_or(mask[:,:,2],(img==i))
            elif i in self.CATS['object']:
                mask[:,:,3] = np.logical_or(mask[:,:,3],(img==i))
            elif i in self.CATS['nature']:
                mask[:,:,4] = np.logical_or(mask[:,:,4],(img==i))
            elif i in self.CATS['sky']:
                mask[:,:,5] = np.logical_or(mask[:,:,5],(img==i))
            elif i in self.CATS['human']:
                mask[:,:,6] = np.logical_or(mask[:,:,6],(img==i))
            elif i in self.CATS['vehicle']:
                mask[:,:,7] = np.logical_or(mask[:,:,7],(img==i))
        return np.array(np.argmax(mask,axis=2), dtype='uint8')

    def __getitem__(self, i):
        # collect batch images and masks
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        #print('start barch is ', start)
        #print('stop barch is ', stop)
        images = []
        masks = []
        for j in range(start, stop):
            # Store entire paths to image_file and mask_file variable
            image_file = list(self.dataset['Path_x'])[j]
            mask_file = list(self.dataset['Path_y'])[j]
            unique_index_file = list(self.dataset['File_unique_index'])[j]

            # OpenImages and masks using OpenCV ibrary, and convert in np array
            image = cv2.imread(image_file)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mask = cv2.imread(mask_file)
            #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # apply resize
            if self.resize:
                dim = (self.resize_width, self.resize_height)
                original_height = image.shape[0]
                original_width = image.shape[1]
                original_channels = image.shape[2]
                image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) # resize image
                mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
                new_height = image.shape[0]
                new_width = image.shape[1]
                new_channels = image.shape[2]
                
            if self.convert:
                mask = self._convert_mask(mask)
                
            # apply display
            if self.display:
                #Display Image and Mask
                plt.figure(figsize=(25, 5))
                plt.subplots_adjust(hspace=0.5)
                plt.subplot(120 + 1 + 0)
                plt.imshow(image)
                plt.subplot(120 + 1 + 1)
                plt.imshow(mask)
                plt.grid(False)
                plt.suptitle(f'Images and Masks {unique_index_file}')
                plt.show()
                if self.resize:
                    print('original height was ', original_height, ' and new height is ', new_height)
                    print('original width was ', original_width, ' and new width is ', new_width)
                    print('original channels was ', original_channels, ' and new channels is ', new_channels)
                    

            # apply normalization : Normalizing RGB values ; The range for each individual colour is 0-255 (as 2^8 = 256 possibilities).
            if self.normalization:
                #normalizedImg = np.zeros((800, 800))
                image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX) #np.array(image/255, dtype='uint8') 
                #mask = np.array(mask/255, dtype='uint8') # cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX) #mask/255
                # apply display
                if self.display:
                    print(f'----------------Normalized Image and Mask {j} from batch collected-----------------')
                    #Display Image and Mask
                    plt.figure(figsize=(25, 5))
                    plt.subplots_adjust(hspace=0.5)
                    plt.subplot(120 + 1 + 0)
                    plt.imshow(image)
                    plt.subplot(120 + 1 + 1)
                    plt.imshow(mask)
                    plt.grid(False)
                    plt.suptitle(f'Normalized Images and Masks {unique_index_file}')
                    plt.show()
            
            # apply augmentations
            if ((self.augmentation is not None) & (self.phase == 'train')):
                # Augment an image
                sample = self.augmentation(image=image, mask=mask)
                augmented_image, augmented_mask = sample['image'], sample['mask']
                # apply display
                if self.display:
                    print(f'----------------Augmented Image and Mask {j} from batch collected-----------------')
                    #Display Image and Mask
                    plt.figure(figsize=(25, 5))
                    plt.subplots_adjust(hspace=0.5)
                    plt.subplot(120 + 1 + 0)
                    plt.imshow(augmented_image)
                    plt.subplot(120 + 1 + 1)
                    plt.imshow(augmented_mask)
                    plt.grid(False)
                    plt.suptitle(f'Augmented Images and Masks {unique_index_file}')
                    plt.show()
                    images.append(augmented_image)
                    masks.append(augmented_mask)
                    
            # append image and mask to batch
            images.append(image)
            masks.append(mask)   
            print(f'---------------- Image and Mask {j} from batch collected in Data Generator-----------------')

        # transpose list of lists
        image_batch = np.stack(images, axis=0) #A confirmer si images ou image
        mask_batch = np.stack(masks, axis=0) #A confirmer si masks ou mask

        #Add 1 dimension to return image_batch and mask_batch in 4 dimensions (batch number, height, width, channels), avec channel = 1 pour les masques et =3 pour les images
        if len(mask_batch.shape) == 3:
            mask_batch = np.expand_dims(mask_batch, axis = 3)

        #print('mask_batch shape is ', mask_batch.shape)
        return image_batch, mask_batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.dataset) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle dataset each epoch"""
        if self.shuffle:
            np.random.shuffle(self.dataset.values)

# %%
# # DATALOADER - Notre classe hérite de la classe Keras.utils.Sequence
# # Elle permet de créer un générateur de données
# # Cette classe parente vous assure dans le cas ou vous souhaitez utiliser du calcul parallèle avec vos threads, de garantir de parcourir une seule et unique fois vos données au cours d’une époch
# class GeneratorCitySpace(keras.utils.Sequence):
        
#     CATS = {
#         'void': [0, 1, 2, 3, 4, 5, 6],
#         'flat': [7, 8, 9, 10],
#         'construction': [11, 12, 13, 14, 15, 16],
#         'object': [17, 18, 19, 20],
#         'nature': [21, 22],
#         'sky': [23],
#         'human': [24, 25],
#         'vehicle': [26, 27, 28, 29, 30, 31, 32, 33,-1]
#     }
    
#     def _convert_mask(self,img):
#         img = np.squeeze(img)
#         mask = np.zeros((img.shape[0], img.shape[1], 8),dtype=np.uint8)
#         for i in range(-1, 34):
#             if i in self.CATS['void']:
#                 mask[:,:,0] = np.logical_or(mask[:,:,0],(img==i))
#             elif i in self.CATS['flat']:
#                 mask[:,:,1] = np.logical_or(mask[:,:,1],(img==i))
#             elif i in self.CATS['construction']:
#                 mask[:,:,2] = np.logical_or(mask[:,:,2],(img==i))
#             elif i in self.CATS['object']:
#                 mask[:,:,3] = np.logical_or(mask[:,:,3],(img==i))
#             elif i in self.CATS['nature']:
#                 mask[:,:,4] = np.logical_or(mask[:,:,4],(img==i))
#             elif i in self.CATS['sky']:
#                 mask[:,:,5] = np.logical_or(mask[:,:,5],(img==i))
#             elif i in self.CATS['human']:
#                 mask[:,:,6] = np.logical_or(mask[:,:,6],(img==i))
#             elif i in self.CATS['vehicle']:
#                 mask[:,:,7] = np.logical_or(mask[:,:,7],(img==i))
#         return np.array(np.argmax(mask,axis=2), dtype='uint8')
    
#     def _transform_data(self,X,Y):
#         if len(Y.shape) == 3:
#             Y = np.expand_dims(Y, axis = 3)
#         X = X /255. 
#         return np.array(X,dtype=np.uint8), Y
    
#     def __init__(self, image_filenames, labels, batch_size,crop_x,crop_y):
#         """Générateur de données avec augmentation des images
#         """
#         self.image_filenames, self.labels = image_filenames, labels
#         self.batch_size = batch_size
#         self.crop_x,self.crop_y = crop_x, crop_y

#     def __len__(self):
#         return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

#     def __getitem__(self, idx):
#         batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
#         x=[cv2.resize(cv2.imread(path_X),(self.crop_x,self.crop_y)) for path_X in batch_x]
#         y = [cv2.resize(self._convert_mask(cv2.imread(path_Y,0)),(self.crop_x,self.crop_y)) for path_Y in batch_y]
#         y=np.array(y)
#         x=np.array(x)
#         return self._transform_data(x,y)

# %%
# DATALOADER - Notre classe hérite de la classe Keras.utils.Sequence
# Elle permet de créer un générateur de données
# Cette classe parente vous assure dans le cas ou vous souhaitez utiliser du calcul parallèle avec vos threads, de garantir de parcourir une seule et unique fois vos données au cours d’une époch
class GeneratorCitySpace(keras.utils.Sequence):
    """Load data from dataset to build bacthes
    Args:
        dataset : dataframe listing data and paths
        n_sample : to work with a reduced dataset
        batch_size: Integet number of images in batch.
        resize_width & resize_height: New dimensions after resizing
        phase: allow to set if dataloader shall filter on train, val or test images/masks
        augmentation: Variable defining augmentation of image and mask
    """
    # We provide one dataset, containing labelIds Masks and Images
    def __init__(self, dataset, n_sample=None, batch_size=1, resize_width=256, resize_height=128, phase='train', augmentation=None):
        self.n_sample = n_sample
        self.phase = phase
        if (self.n_sample is None):
            self.dataset = dataset.loc[dataset['Phase'] == self.phase, :]
        else:
            self.dataset = dataset.loc[dataset['Phase'] == self.phase, :].sample(self.n_sample) #ou dataset si on veut prendre tour le jeu de donnée
        self.batch_size = batch_size
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.augmentation = augmentation
        self.on_epoch_end()

    CATS = {
        'void': [0, 1, 2, 3, 4, 5, 6],
        'flat': [7, 8, 9, 10],
        'construction': [11, 12, 13, 14, 15, 16],
        'object': [17, 18, 19, 20],
        'nature': [21, 22],
        'sky': [23],
        'human': [24, 25],
        'vehicle': [26, 27, 28, 29, 30, 31, 32, 33,-1]
    }
    
    def _convert_mask(self,img):
        img = np.squeeze(img)
        mask = np.zeros((img.shape[0], img.shape[1], 8),dtype=np.uint8)
        for i in range(-1, 34):
            if i in self.CATS['void']:
                mask[:,:,0] = np.logical_or(mask[:,:,0],(img==i))
            elif i in self.CATS['flat']:
                mask[:,:,1] = np.logical_or(mask[:,:,1],(img==i))
            elif i in self.CATS['construction']:
                mask[:,:,2] = np.logical_or(mask[:,:,2],(img==i))
            elif i in self.CATS['object']:
                mask[:,:,3] = np.logical_or(mask[:,:,3],(img==i))
            elif i in self.CATS['nature']:
                mask[:,:,4] = np.logical_or(mask[:,:,4],(img==i))
            elif i in self.CATS['sky']:
                mask[:,:,5] = np.logical_or(mask[:,:,5],(img==i))
            elif i in self.CATS['human']:
                mask[:,:,6] = np.logical_or(mask[:,:,6],(img==i))
            elif i in self.CATS['vehicle']:
                mask[:,:,7] = np.logical_or(mask[:,:,7],(img==i))
        return np.array(mask,dtype='uint8') #retrun the mask OneHotEncoded
        #return np.array(np.argmax(mask,axis=2), dtype='uint8') #return the mask with the labelId

    def __getitem__(self, i):
        # collect batch images and masks
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        images = []
        masks = []
        for j in range(start, stop):
            # OpenImages and masks, and apply resize + mask color conversion + Normalization
            image = (cv2.resize(cv2.imread(list(self.dataset['Path_x'])[j]), (self.resize_width, self.resize_height), interpolation = cv2.INTER_AREA)) #image/255 ou normalizedImg
            mask = self._convert_mask(cv2.resize(cv2.imread(list(self.dataset['Path_y'])[j],0), (self.resize_width, self.resize_height), interpolation = cv2.INTER_AREA)) #mask/255
            #mask = cv2.resize(cv2.imread(list(self.dataset['Path_y'])[j]), (self.resize_width, self.resize_height), interpolation = cv2.INTER_AREA) #mask/255
            
            if ((self.augmentation is not None) & (self.phase == 'train')):
                # apply augmentations
                sample = self.augmentation(image=image, mask=mask)
                augmented_image, augmented_mask = sample['image'], sample['mask'] 
                #augmented_image = augmented_image/255
                #augmented_mask = augmented_mask/255
                # append augmented image and mask to batch
                images.append(augmented_image)
                masks.append(augmented_mask)
            image = image/255
            #mask = mask/255


            # append image and mask to batch
            images.append(image)
            masks.append(mask)   

        # transpose list of lists = Data Generator outputs
        image_batch = np.stack(images, axis=0) 
        #Add 1 dimension to return image_batch and mask_batch in 4 dimensions (batch number, height, width, channels), avec channel = 1 pour les masques et =3 pour les images
        #mask_batch = np.expand_dims(np.stack(masks, axis=0), axis = 3)
        mask_batch = np.stack(masks, axis=0) 

        # Data Generator outputs
        return image_batch, mask_batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.dataset) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle dataset each epoch"""
        np.random.shuffle(self.dataset.values)

# %% [markdown]
# # Model UNET

# %%
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.layers import Activation, MaxPool2D, Concatenate

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50


#Encoder block: Double convolution avec maxpooling
#Durant cette phase, la taille de l'image réduit graduellement, alors que la profondeur s'accroit.
#Cela signifie que le réseau apprend le "QUOI", mais perd le "Où"
def encoder_block(input, num_filters, maxpool=None, dropout=None):
    x = Conv2D(num_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input)
    x = Conv2D(num_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    if dropout:
        d = Dropout(0.5)(x)
        if maxpool:
            p = MaxPooling2D(pool_size=(2, 2))(d)    
            return x, p, d
        else:
            return x, d
    if maxpool:
        p = MaxPooling2D(pool_size=(2, 2))(x)    
        return x, p
    else:
        return x

#Decoder block
#Durant cette phase, le décoder accroit la taille de l'image, alors que la prodondeur réduit
#Il retrouve l'information "Où" avec les Up_Sampling. 
#Au travers de la concaténation, i les Skip-Connections des couches.
# #A la suite de chaque concaténation, nous appliquons encore les coubles convolutions
def decoder_block(input, skip_connect, num_filters):
    x = Conv2D(num_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(input))
    x = concatenate([skip_connect, x], axis = 3) #skip connections that concatenate the encoder feature map with the decoder
    x = Conv2D(num_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    x = Conv2D(num_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
    return x    

    #Build Unet using the blocks
def build_unet_block(input_shape, nb_class, n_filters = 32):
    inputs = Input(input_shape) #(resize_height, resize_width, 3) #Transposed compare to input dim of data generator
 
    conv1, pool1 = encoder_block(inputs, n_filters * 1, maxpool=True, dropout=None)
    conv2, pool2 = encoder_block(pool1, n_filters * 2, maxpool=True, dropout=None)
    conv3, pool3 = encoder_block(pool2, n_filters * 4, maxpool=True, dropout=None)
    conv4, pool4, drop4 = encoder_block(pool3, n_filters * 8, maxpool=True, dropout=True)
    conv5, drop5 = encoder_block(pool4, n_filters * 16, maxpool=None, dropout=True)

    conv6 = decoder_block(drop5, drop4, n_filters * 8)
    conv7 = decoder_block(conv6, conv3, n_filters * 4)
    conv8 = decoder_block(conv7, conv2, n_filters * 2)
    conv9 = decoder_block(conv8, conv1, n_filters * 1)
    
    outputs = Conv2D(nb_class, 1, activation = 'softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs], name='U-Net')
    return model

# %%
def build_unet():
    inputs = Input((resize_width, resize_height, 3))
    # Input
    s = Lambda(x)(inputs)
    # Layer 1 
    c1 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPool2D((2, 2))(c1)

    c2 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPool2D((2, 2))(c2)

    c3 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPool2D((2, 2))(c3)


    c4 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPool2D((2, 2))(c4)


    c5 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)



    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)



    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)



    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(8, (1, 1), activation='softmax')(c9)

    return Model(inputs=[inputs], outputs=[outputs])


# %%
def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x
  
def get_unet(input_img, n_filters = 64, dropout = 0.1, batchnorm = True):
    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(8, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


# %% [markdown]
# ## UNET Metrics and Loss

# %%
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.cast(K.flatten(y_true), K.floatx()) #K.flatten(y_true) #
    y_pred_f = K.cast(K.flatten(y_pred), K.floatx()) #K.flatten(y_pred) #
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def total_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + (3*dice_loss(y_true, y_pred))
    return loss

# %% [markdown]
# 


