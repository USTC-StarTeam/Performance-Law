import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import os
from collections import defaultdict
import heapq
Titles=['Max Sequence Length = 50','Max Sequence Length = 100','Max Sequence Length = 150','Max Sequence Length = 200']
def normalize_row(row):
    min_val = np.min(row)
    max_val = np.max(row)
    return (row - min_val) / (max_val - min_val) if max_val != min_val else row


def prepare_data(p2):
    x1_data = np.tile([4, 8, 12, 16, 24, 32], (4, 4))
    x2_data = np.tile(np.repeat([25, 50, 75, 100], 6),(4,1))
    
    y_data = np.array([
    #paste your result matrix on ../General_Transformer
    ], dtype=float)
    for iidx in range(y_data.shape[0]):
        for jidx in range(y_data.shape[1]):
            y_data[iidx][jidx]=y_data[iidx][jidx]/p2
    return x1_data,x2_data,y_data

def total_main(pf1,pf2,p2):
    def model_func(X, w1, w2, D, b, w3, w4,w5):
        N, h = X
        u1 = (pf1*N**w3)
        u2 = (pf1*h**w4)
        u3 = (pf2*D**w5)
        return w1 * (-1/u1-w3*np.log(N)) + w2 * (-1/u2-w4*np.log(h)) + -1/u3-w5*np.log(D) + b

    def paint2_main():
        x1_data,x2_data,y_data=prepare_data(p2)
        x1_data_new=[]
        x2_data_new=[]
        y_data_new=[]
        for i in range(x1_data.shape[0]):
            x1_data_new.append(x1_data[i].reshape(-1, 24))
            x2_data_new.append(x2_data[i].reshape(-1,24))
            y_data_new.append(y_data[i].reshape(-1,24))
        x1_data_new=np.array(x1_data_new)
        x2_data_new=np.array(x2_data_new)
        y_data_new=np.array(y_data_new)
        initial_guess = [1,1,1,1,1,1,1]
        
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            fitted_params_list = []
            fig = plt.figure(figsize=(20, 5))
            axs = [fig.add_subplot(1, 4, i+1, projection='3d') for i in range(4)]
            all_r2=[]
            all_param=[]
            all_mean_D=[]
            for j in range(x1_data_new.shape[0]):
                x1_data=x1_data_new[j]
                x2_data=x2_data_new[j]
                y_data=y_data_new[j]
                mean_d=[]
                for i in range(x1_data.shape[0]):
                    try:
                        tmp_x1=np.array([x1_data[i]])
                        tmp_x2=np.array([x2_data[i]])
                        tmp_x1x2=np.vstack((tmp_x1,tmp_x2))
                        params, params_covariance = curve_fit(
                            model_func,tmp_x1x2, y_data[i], p0=initial_guess,
                            bounds=([-np.inf,-np.inf,0,-np.inf,-np.inf,-np.inf,-np.inf], [np.inf,np.inf, np.inf, np.inf, np.inf, np.inf,np.inf]),
                            maxfev=500000, ftol=1e-10, xtol=1e-10
                        )
                        fitted_params_list.append(params)

                        w1,w2,D,b,w3,w4,w5 = params
                        y_fitted = model_func(tmp_x1x2,w1,w2,D,b,w3,w4,w5)
                        mse = np.mean((y_data[i] - y_fitted)**2)
                        ss_res = np.sum((y_data[i] - y_fitted) ** 2)
                        ss_tot = np.sum((y_data[i] - np.mean(y_data[i])) ** 2)
                        r2 = 1 - (ss_res / ss_tot)
                        all_r2.append(r2)
                        all_param.append([w1,w2,D,b,w3,w4,w5])
                        points=100
                        x1_fit = np.linspace(min(x1_data[i]), max(x1_data[i]), points)
                        x2_fit = np.linspace(min(x2_data[i])-2, max(x2_data[i])+2, points)
                        X_grid = np.meshgrid(x1_fit, x2_fit)
                        y_fit = model_func(X_grid, w1,w2,D,b,w3,w4,w5)
                        ax = axs[j]
                        x1_grid, x2_grid=X_grid
                        ax.plot_surface(x1_grid, x2_grid, y_fit, alpha=0.7, cmap='viridis',label=f'Fit:HD={(i+1)*25},w1={w1:.2f},w2={w2:.2f},D={D:.2f},b={b:.2f}')
                        ax.scatter(x1_data[i], x2_data[i], y_data[i], color='red')
                        mean_d.append(D)
                        ax.set_xlabel('N')
                        ax.set_ylabel('D')
                        ax.set_zlabel('HR')
                        ax.set_title(Titles[j] + ' MD = ' + str(round(np.average(mean_d), 4)))
                        ax.legend(loc='upper right')
                        ax.grid(True)

                    except OptimizeWarning as e:
                        print(f"Optimization warning for dataset {i}: {e}")
                    except RuntimeError as e:
                        print(f"Runtime error for dataset {i}: {e}")
                
                all_mean_D.append(np.average(mean_d))
            plt.tight_layout()
            plt.close()
            fitted_params_array = np.array(fitted_params_list)
            return all_param,all_r2,all_mean_D
    all_param,all_r2,all_mean_D=paint2_main()
    return all_param,all_r2,all_mean_D

def paint_main():
    x1_data,x2_data,y_data=prepare_data(1)
    data_list=y_data
    data_list_normalized = []
    for data in data_list:
        rows, cols = 4, 6
        array_2d = []
        for i in range(rows):
            row = data[i * cols:(i + 1) * cols]
            array_2d.append(row)
        normalized_array_2d = []
        for row in array_2d:
            normalized_row = normalize_row(np.array(row))
            normalized_array_2d.append(normalized_row)
        
        data_list_normalized.append(np.array(normalized_array_2d))
    xticks = [4, 8, 12, 16, 24, 32]
    yticks = [25, 50, 75, 100]
    plot_heatmap(data_list_normalized, '1.pdf', xticks, yticks)

def grid_search_and_fit(k):
    dataset_results = defaultdict(list)
    pf1=2
    pf2=2
    p2=1
    all_param, all_r2,all_mean_d = total_main(pf1, pf2, p2)
    print(all_param, all_r2,all_mean_d)
    for idx, (param, r2,mean_d) in enumerate(zip(all_param, all_r2,all_mean_d)):
        dataset_results[idx].append((r2, param,mean_d, pf1, pf2, p2))
    param_list=[]
    pf1_list=[]
    pf2_list=[]
    p2_list=[]
    chosen=[0,0,0,0]
    dataset_idx=0
    for dataset, results in dataset_results.items():
        top_k_results = heapq.nlargest(k, results, key=lambda x: x[0])
        print(f'Dataset {dataset} top {k} results:')
        idx=0
        for r2, param, md, pf1, pf2, p2 in top_k_results:
            if idx==chosen[dataset_idx]:
                pf1_list.append(pf1)
                pf2_list.append(pf2)
                p2_list.append(p2)
                param_list.append(param)
            print(f'R2: {r2}, Parameters: {param}, md: {md} pf1: {pf1}, pf2: {pf2}, p2: {p2}')
            idx+=1
        dataset_idx+=1
        print('------------------------')
    print(param_list)
    only_paint(param_list,pf1_list,pf2_list,p2_list)
grid_search_and_fit(k=30)