import os

import matplotlib.pyplot as plt
import numpy as np
from plot_data import *
from graph_regression_rubber import *


def get_value_from_expr(expr,all_x,all_y):
    def lambdify_expression(expr, symbols):
        return lambdify(symbols, expr, 'numpy')

    all_pred=[]
    lambdified_expr = lambdify_expression(expr, x)
    for n in range(len(all_x)):
    # Use the lambdified function to calculate y_data for each x in x_data
        pred = lambdified_expr(all_x[n])
        all_pred.append(pred)
    print(all_pred)
    print(all_y)
    plt.scatter(all_pred,all_y)
    plt.show()

if __name__ == '__main__':
    data_name=os.listdir('data_rubber')
    mode='Train'
    all_x=[]
    all_y=[]
    print(data_name)
    for name in data_name:
        file_path = f"data_rubber/{name}"
        df = pd.read_excel(file_path)
        # 确保列名正确
        strain_values = df['nominal strain'].values
        stress_values = df['nominal stress'].values
        lamda_values=1+strain_values

        all_x.append(lamda_values)
        all_y.append(stress_values)


    model = Random_graph_for_expr()
    model_sympy = Graph_to_sympy()
    GA = Genetic_algorithm(x_data=all_x,y_data=all_y)


    if mode=='Train':
        GA.max_unconstant=3
        GA.evolution(save_dir='constitutive_rubber')