import numpy as np
import sympy as sp
from sympy import lambdify,symbols,integrate,solve
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize,curve_fit
import torch
import math
import random
from matplotlib.ticker import MaxNLocator
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def plot_pred_vs_obs(x_obs,x1_obs,y_pred,y_obs,save_name,style='log'):

    plt.figure(figsize=(2.5, 2.5), dpi=300)
    colors = ['#845EC2','#0081CF','#4B4453','#B0A8B9','#C34A36','#FF8066']
    for i,(x,x1,p,o) in enumerate(zip(x_obs,x1_obs,y_pred,y_obs)):
        plt.plot(x, o,c=colors[i], label=f"T = {x1[0]} K",linewidth=1)
        plt.scatter(x, p,marker='^',s=5,c=colors[i])
        # 坐标轴优化，使其与文章风格匹配
    plt.xlim([0.02, 2])
    plt.ylim([0.2, 1])
    if style=='log':
        plt.xscale("log")  # 保持y轴线性
        plt.yscale("log")  # 保持y轴线性
    plt.xlim([0.02, 2])
    plt.ylim([0.2, 1])
    if style == 'log':
        plt.xticks([0.02, 0.05, 0.1, 0.5, 1, 2], [0.02, 0.05, 0.1, 0.5, 1, 2], fontproperties='Arial', fontsize=7)
        plt.yticks([ 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   fontproperties='Arial', fontsize=7)
    if style=='linear':
        plt.xticks([0.1, 0.5, 1, 2], [0.1, 0.5, 1, 2], fontproperties='Arial', fontsize=7)
        plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   fontproperties='Arial', fontsize=7)
    # 设置对数刻度

    # 设置标签字体和字号
    plt.xlabel("Stress $\sigma$ (MPa)", fontname="Arial", fontsize=7)
    plt.ylabel("Normalized Plastic Strain Rate", fontname="Arial", fontsize=7)

    plt.legend(prop={"family": "Arial", "size": 6}, ncol=2)
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格

    plt.savefig(f"plot_save/{save_name}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"plot_save/{save_name}.png", bbox_inches='tight', dpi=300)

    plt.show()
def calculate_relative_error(pred_concate,y_concate):
    pred_concate=np.array(pred_concate)
    y_concate=np.array(y_concate)
    valid_idx = ~np.isnan(pred_concate)

    pred_concate = pred_concate[valid_idx]
    y_concate = y_concate[valid_idx]
    MSE=np.mean((pred_concate-y_concate)**2)
    R2=1 - (((y_concate - pred_concate) ** 2).sum() / ((y_concate - y_concate.mean()) ** 2).sum())
    return MSE,R2
def plot_pred_vs_obs_kesi(x_obs,x1_obs,y_pred,y_obs,save_name,style='log'):
    Kesi = ['2e-2', '3e-3', '3e-4', '4e-5']
    plt.figure(figsize=(2.5, 2.5), dpi=300)
    colors = ['#845EC2','#0081CF','#4B4453','#B0A8B9','#C34A36','#FF8066']
    for i,(x,x1,p,o) in enumerate(zip(x_obs,x1_obs,y_pred,y_obs)):
        plt.plot(x, o,c=colors[i], label=f"kesi= {Kesi[i]}",linewidth=1)
        plt.scatter(x, p,marker='^',s=5,c=colors[i])
        # 坐标轴优化，使其与文章风格匹配
    plt.xlim([0.02, 2])
    plt.ylim([0.6, 1])
    if style=='log':
        plt.xscale("log")  # 保持y轴线性
        plt.yscale("log")  # 保持y轴线性
    if style == 'log':
        plt.xticks([0.02, 0.05, 0.1, 0.5, 1, 2], [0.02, 0.05, 0.1, 0.5, 1, 2], fontproperties='Arial', fontsize=7)
        plt.yticks([ 0.6, 0.7, 0.8, 0.9, 1.0], [0.6, 0.7, 0.8, 0.9, 1.0],
                   fontproperties='Arial', fontsize=7)
    if style=='linear':
        plt.xticks([0.1, 0.5, 1, 2], [0.1, 0.5, 1, 2], fontproperties='Arial', fontsize=7)
        plt.yticks([ 0.6, 0.7, 0.8, 0.9, 1.0], [ 0.6, 0.7, 0.8, 0.9, 1.0],
                   fontproperties='Arial', fontsize=7)
    # 设置对数刻度

    # 设置标签字体和字号
    plt.xlabel("Stress $\sigma$ (MPa)", fontname="Arial", fontsize=7)
    plt.ylabel("Normalized Plastic Strain Rate", fontname="Arial", fontsize=7)

    plt.legend(prop={"family": "Arial", "size": 6}, ncol=2)
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格

    plt.savefig(f"plot_save/{save_name}.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(f"plot_save/{save_name}.png", bbox_inches='tight', dpi=300)

    plt.show()

def plot_pred_obs_DIF(strain_rate,predicted_DIF,actual_DIF,data_name,plot=True,save=False):
    plt.figure(figsize=(2.5, 2.5), dpi=300)

    # 绘制预测数据：scatter
    plt.scatter(strain_rate, predicted_DIF, label='Predicted', color='black', marker='^',s=5,alpha=0.7)

    # 绘制实际数据：虚线（dotted line）
    plt.plot(strain_rate, actual_DIF, linestyle='--', linewidth=1,label='Actual', color='red')

    # 设置标签与字体
    font_settings = {'family': 'Arial', 'size': 7}
    plt.xscale('log')
    plt.xlabel('Strain Rate', fontdict=font_settings)
    plt.ylabel('DIF', fontdict=font_settings)
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格
    # 设置刻度字体
    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')

    R2=1-((actual_DIF-predicted_DIF)**2).sum()/((actual_DIF-np.mean(actual_DIF))**2).sum()
    plt.title(f'{data_name}:  R2={R2:.3f}',fontdict={'family': 'Arial', 'fontsize': 6})
    # 设置图例字体
    plt.legend(prop={"family": "Arial", "size": 6})

    plt.tight_layout()
    if save==True:
        plt.savefig(f"plot_save/plot_pred_strain_rate_DIF/{data_name}.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"plot_save/plot_pred_strain_rate_DIF/{data_name}.png", bbox_inches='tight', dpi=300)
    if plot==True:
        plt.show()

def plot_pred_obs_strain_hardening(strain,predicted_stress,actual_stress,data_name,plot=True,save=False):
    plt.figure(figsize=(2.5, 2.5), dpi=300)

    # 绘制预测数据：scatter
    plt.scatter(strain, predicted_stress, label='Predicted', color='black', marker='^',s=5,alpha=0.7)

    # 绘制实际数据：虚线（dotted line）
    plt.plot(strain, actual_stress, linestyle='--', linewidth=1,label='Actual', color='red')

    # 设置标签与字体
    font_settings = {'family': 'Arial', 'size': 7}
    plt.xlabel('True Plastic Strain', fontdict=font_settings)
    plt.ylabel('True Stress', fontdict=font_settings)
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格
    # 设置刻度字体
    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')

    R2=1-((actual_stress-predicted_stress)**2).sum()/((actual_stress-np.mean(actual_stress))**2).sum()
    plt.title(f'{data_name[0:-5]}:  R2={R2:.3f}',fontdict={'family': 'Arial', 'fontsize': 6})
    # 设置图例字体
    plt.legend(prop={"family": "Arial", "size": 6})
    ax = plt.gca()
    # 使用 MaxNLocator 限制 x 轴主刻度数量为 4
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.tight_layout()
    if save==True:
        plt.savefig(f"plot_save/#1plot_strain_harderning_obs_predict/{data_name}.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"plot_save/#1plot_strain_harderning_obs_predict/{data_name}.png", bbox_inches='tight', dpi=300)
    if plot==True:
        plt.show()

def plot_pred_obs_hardening_strain_rate(strain,predicted_stress,actual_stress,strain_rate,data_name,plot=True,save=False):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]
    plt.figure(figsize=(2.25, 2.5), dpi=300)
    relative_error=[]
    #plt.ylim([0,1200])
    for i in range(len(predicted_stress)):
        plt.scatter(strain[i],actual_stress[i],color=colors[i], marker='^',s=5,alpha=0.8)
    # 绘制预测数据：scatter
        plt.plot(strain[i], predicted_stress[i],label=f"{strain_rate[i][0]} s$^{{-1}}$",linestyle='--', linewidth=1, color=colors[i])
        relative_error.append(np.mean(np.abs((actual_stress[i] - predicted_stress[i]) / actual_stress[i])))




    # 设置标签与字体
    font_settings = {'family': 'Arial', 'size': 7}
    # plt.xlabel('True Plastic Strain', fontdict=font_settings)
    # plt.ylabel('True Stress', fontdict=font_settings)
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格
    # 设置刻度字体
    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')
    R2 = 1 - ((np.concatenate(actual_stress) - np.concatenate(predicted_stress)) ** 2).sum() / (
                (np.concatenate(actual_stress) - np.mean(np.concatenate(actual_stress))) ** 2).sum()

    if data_name=='Liu(Q460JSC)_2020.pkl':
        plt.xticks([0,0.1,0.2,0.3],[0,0.1,0.2,0.3],fontsize=7, fontname='Arial')
        plt.yticks([600,800,1000,1200],[600,800,1000,1200],fontsize=7, fontname='Arial')
        print(relative_error)
        print(R2)
    elif data_name == 'Yang_2020.pkl':
        plt.ylim([675,None])
        plt.yticks([700, 800, 900, 1000], [700, 800, 900, 1000], fontsize=7, fontname='Arial')
    #plt.title(f'{data_name[0:-4]}:  R2={R2:.3f}',fontdict={'family': 'Arial', 'fontsize': 6})
    # 设置图例字体
    plt.legend(prop={"family": "Arial", "size": 5.5},ncol=1)

    plt.tight_layout()
    if save==True:
        #plt.savefig(f"plot_save/plot_pred_hardening_strain_rate/{data_name}.pdf", bbox_inches='tight', dpi=300)
        plt.savefig(f"plot_save/plot_pred_hardening_strain_rate/{data_name}_modified.pdf", bbox_inches='tight', dpi=300)
        #plt.savefig(f"plot_save/plot_pred_hardening_strain_rate/{data_name}.png", bbox_inches='tight', dpi=300)
    if plot==True:
        plt.show()

def plot_pred_obs_hardening_strain_rate_JC(strain,predicted_stress,actual_stress,strain_rate,data_name,plot=True,save=True):
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]
    plt.figure(figsize=(2.25, 2.5), dpi=300)
    #plt.ylim([0,1200])
    relative_error=[]
    for i in range(len(predicted_stress)):
        plt.scatter(strain[i], actual_stress[i],color=colors[i], marker='^', s=5, alpha=0.8)
        # 绘制预测数据：scatter
        plt.plot(strain[i], predicted_stress[i], label=f"{strain_rate[i]} s$^{{-1}}$", linestyle='--', linewidth=1,
                 color=colors[i])
        relative_error.append(np.mean(np.abs((actual_stress[i]-predicted_stress[i])/actual_stress[i])))

    # 设置标签与字体
    font_settings = {'family': 'Arial', 'size': 7}
    # plt.xlabel('True Plastic Strain', fontdict=font_settings)
    # plt.ylabel('True Stress', fontdict=font_settings)
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格
    # 设置刻度字体
    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')
    if data_name == 'Liu(Q460JSC)_2020.pkl':

        plt.xticks([0, 0.1, 0.2, 0.3], [0, 0.1, 0.2, 0.3], fontsize=7, fontname='Arial')
        plt.yticks([600, 800, 1000, 1200], [600, 800, 1000, 1200], fontsize=7, fontname='Arial')
        print(relative_error)
    elif data_name == 'Yang_2020.pkl':
        plt.yticks([700,800, 900, 1000], [700,800, 900, 1000], fontsize=7, fontname='Arial')
    R2=1-((np.concatenate(actual_stress)-np.concatenate(predicted_stress))**2).sum()/((np.concatenate(actual_stress)-np.mean(np.concatenate(actual_stress)))**2).sum()

    #plt.title(f'{data_name[0:-4]}:  R2={R2:.3f}',fontdict={'family': 'Arial', 'fontsize': 6})
    # 设置图例字体
    plt.legend(prop={"family": "Arial", "size": 5.5},ncol=1)

    plt.tight_layout()
    if save == True:
        plt.savefig(f"plot_save/plot_pred_hardening_strain_rate/{data_name}_JC.pdf", bbox_inches='tight', dpi=300)
        #plt.savefig(f"plot_save/plot_pred_hardening_strain_rate/{data_name}_JC.png", bbox_inches='tight', dpi=300)
    if plot==True:
        plt.show()

def find_min_max(list_of_arrays):
    """
    接受一个包含多个 numpy 数组的列表，
    返回所有数组元素中的最小值和最大值。

    参数:
        list_of_arrays: list of numpy.ndarray

    返回:
        min_val: 所有数组元素的最小值
        max_val: 所有数组元素的最大值
    """
    # 将列表中的所有数组合并为一个一维数组
    all_values = np.concatenate(list_of_arrays)

    # 求最小值和最大值
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    return min_val, max_val