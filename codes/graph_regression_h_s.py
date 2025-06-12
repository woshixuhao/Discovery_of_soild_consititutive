'''
graph regression for any expression with unknown form and coefficient---More variables
formal version
by HaoXu
'''

import os
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy
from torch_geometric.data import Data
import torch
from scipy.optimize import minimize,curve_fit
from tqdm import tqdm
import time
import heapq
from sympy import symbols, sin, cos, tan, log, ln, sqrt, exp, csc, sec, cot, sinh, tanh, cosh, atan, asin, acos, atanh, \
    asinh, acosh, sympify,pi, lambdify,E,I
from copy import deepcopy
import warnings
import sympy as sp
import pickle
from concurrent.futures import ProcessPoolExecutor
import traceback

warnings.filterwarnings("ignore", category=RuntimeWarning)
NODE_FEATURE_MAP = {
    "1":1,
    'add': 2,
    'mul': 3,
    'exp': 4,
    'div': 5,
    "log": 6,
    "ln": 7,
    "sqrt": 8,
    "abs": 9,
    "sub": 10,
    "sin": 11,
    "cos": 12,
    "tan":13,
    "csc":14,
    "sec":15,
    "cot":16,
    "sinh":17,
    "tanh":18,
    "cosh":19,
    "atan":20,
    "asin":21,
    "acos":22,
    "atanh":23,
    "asinh":24,
    "acosh":25,
    'x': 26,  # Variable
    'a':27,
    'b':28,
    'c':29,
    'd':30,
    'n':31,
    'm':32,
    'E':33,
    'pi':34,
    'x1':35
}
Binary_Operator=['add','mul']
Unary_Operator_ln=["log"]
Unary_Operator_exp=['exp']
Triangle_Operator=["sin", "cos","tan"]
Arctriangle_Operator=["csc","sec","cot","sinh", "tanh","cosh","atan","asin","acos","atanh","asinh","acosh" ]
Variable=['x']
Constant=['1','pi']
polynomial_integral={'value':[-2,-1,-0.5,0,0.5,1,2,-1e8],'prob':[2,3,2,1,2,3,2,1]} #-1e8 indicate a varying coefficient to be determined
x,x1,x2,x3,x4, C,C1,C2,C3,C4,C5,A,B = symbols('x x1 x2 x3 x4 C C1 C2 C3 C4 C5 A B')
use_var='x'
sp_symbol_list=[x,x1]

def set_random_seeds(rand_seed=1101, np_rand_seed=525):
    random.seed(rand_seed)
    np.random.seed(np_rand_seed)

#set_random_seeds(435)

def convert_graph_to_pyG(graph):
    graph_nodes = graph['nodes']
    graph_edges = graph['edges']
    graph_edge_attr = graph['edge_attr']
    x=[NODE_FEATURE_MAP[node] for node in graph_nodes]
    x=torch.from_numpy(np.array(x).astype(np.int64))
    edge_index=torch.from_numpy(np.array(graph_edges).astype(np.int64)).T
    edge_attr=torch.from_numpy(np.array(graph_edge_attr).astype(np.float32))
    pyG_graph=Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return pyG_graph


class Random_graph_for_expr():
    def concate_subgraph_to_node(self, graph, subgraph, concate_node_index, concate_node,set_maximum_node_num=-1,with_node_indice=[]):
        '''
        :param graph: a tuple of [nodes, edges, edge_attr] for the existing graph
        :param concate_node: concate the graph to which node
        :param subgraph: a tuple of [nodes, edges, edge_attr] for the generated subgraph
        :return: new graph
        '''
        if set_maximum_node_num==-1:
            maximun_node_num = len(graph['nodes'])
        else:
            maximun_node_num =set_maximum_node_num
        subgraph_nodes = subgraph['nodes']
        subgraph_edges = subgraph['edges']
        subgraph_edge_attr = subgraph['edge_attr']

        graph_nodes = graph['nodes']
        graph_edges = graph['edges']
        graph_edge_attr = graph['edge_attr']

        for sublist in subgraph_edges:
            for i in range(len(sublist)):
                sublist[i] += maximun_node_num
        graph_nodes += subgraph_nodes
        graph_edges.append([concate_node_index, maximun_node_num])
        graph_edges += subgraph_edges
        if concate_node =='exp':
            graph_edge_attr.append(random.choices([-2,-1,-0.5,0.5,1,2,-1e8],[0.1,0.25,0.1,0.1,0.25,0.1,0.2],k=1)[0])
        elif concate_node=='add':
            graph_edge_attr.append(random.choices([1,-1,-1e8],[0.45,0.45,0.1],k=1)[0])
        elif concate_node=='mul':
            graph_edge_attr.append(random.choices([1, -1, -1e8], [0.3, 0.3,0.4], k=1)[0])
        elif concate_node=='E_exp':
            graph_edge_attr.append(random.choices([1, -1, -1e8], [0.3, 0.3,0.4], k=1)[0])
        else:
            graph_edge_attr.append(1)
        graph_edge_attr += subgraph_edge_attr
        graph = {'nodes': graph_nodes, 'edges': graph_edges, 'edge_attr': graph_edge_attr}

        if len(with_node_indice)==0:
            return graph
        else:
            with_node_indice=[maximun_node_num+i for i in range(len(subgraph_nodes))]
            return graph,with_node_indice
    def generate_single_poly(self,polynomial_integral=polynomial_integral,var='x'):
        poly_integ =random.choices(polynomial_integral['value'],weights=np.array(polynomial_integral['prob'])/sum(polynomial_integral['prob']),k=1)[0]
        with_bias=random.choice(['bias','no_bias'])
        if with_bias=='no_bias':
            if poly_integ == 0:
                subgraph = {'nodes': ['1'], 'edges': [], 'edge_attr': []}
            elif poly_integ == 1:
                    subgraph = {'nodes': [var], 'edges': [], 'edge_attr': []}
            else:
                    subgraph = {'nodes': ['exp', var], 'edges': [[0, 1]], 'edge_attr': [poly_integ]}
        if with_bias=='bias':
            if poly_integ == 0:
                subgraph = {'nodes': ['1'], 'edges': [], 'edge_attr': []}
            elif poly_integ == 1:
                    subgraph = {'nodes': ['add',var,'1'], 'edges': [[0,1],[0,2]], 'edge_attr': [1,-1e8]}
            else:
                    subgraph = {'nodes': ['exp','add', var,'1'], 'edges': [[0, 1],[1,2],[1,3]], 'edge_attr': [poly_integ,1,-1e8]}
        return subgraph
    def generate_polynomial_template(self,polynomial_integral=polynomial_integral,max_poly_term=1,var='x'):
        n_terms=random.randint(1,max_poly_term)

        if n_terms==1:
           subgraph=self.generate_single_poly(polynomial_integral,var=var)
        else:
            subgraph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
            for i in range(n_terms):
                subgraph=self.concate_subgraph_to_node(subgraph,self.generate_single_poly(var=var),0,'add')
        return subgraph

    def generate_log_template(self,var='x'):
        '''
               The function to generate subgraph for the node 'log' and 'ln'
               define Template:
               log(A/B)
               log(A+B)
               log(A*B)
               :return:
               '''

        Template = ['log(A+B)', 'ln(A+B)','log(A)','ln(A)']
        use_template = random.choice(Template)
        if use_template == 'log(A/B)':
            initial_graph = {'nodes': ['log','mul'], 'edges': [[0,1]], 'edge_attr': [10]}
            A_graph=self.generate_polynomial_template(polynomial_integral={'value':[0,1,2,3],'prob':[1,2,1,1]},var=var)
            B_graph=self.generate_polynomial_template(polynomial_integral={'value':[0,1,2,3],'prob':[1,2,1,1]},var=var)
            subgraph=self.concate_subgraph_to_node(initial_graph,A_graph,1,'mul')
            inverse_B={'nodes': ['exp'], 'edges': [], 'edge_attr': []}
            B_graph=self.concate_subgraph_to_node(inverse_B,B_graph,0,'exp')
            B_graph['edge_attr'][0]=-1
            graph=self.concate_subgraph_to_node(subgraph,B_graph,1,'mul')
        if use_template == 'log(A+B)':
            initial_graph = {'nodes': ['log'], 'edges': [], 'edge_attr': []}
            A_graph = self.generate_polynomial_template(
                polynomial_integral={'value': [0, 1, 2, 3], 'prob': [1, 2, 1, 1]},var=var)
            graph = self.concate_subgraph_to_node(initial_graph, A_graph, 0, 'log')
            graph['edge_attr'][0]=10
        if use_template == 'ln(A/B)':
            initial_graph = {'nodes': ['log', 'mul'], 'edges': [[0, 1]], 'edge_attr': [math.e]}
            A_graph = self.generate_polynomial_template(
                polynomial_integral={'value': [0, 1, 2, 3], 'prob': [1, 2, 1, 1]},var=var)
            B_graph = self.generate_polynomial_template(
                polynomial_integral={'value': [0, 1, 2, 3], 'prob': [1, 2, 1, 1]},var=var)
            subgraph = self.concate_subgraph_to_node(initial_graph, A_graph, 1, 'mul')
            inverse_B = {'nodes': ['exp'], 'edges': [], 'edge_attr': []}
            B_graph = self.concate_subgraph_to_node(inverse_B, B_graph, 0, 'exp')
            B_graph['edge_attr'][0] = -1
            graph = self.concate_subgraph_to_node(subgraph, B_graph, 1, 'mul')
        if use_template == 'ln(A+B)':
            initial_graph = {'nodes': ['log'], 'edges': [], 'edge_attr': []}
            A_graph = self.generate_polynomial_template(
                polynomial_integral={'value': [0, 1, 2, 3], 'prob': [1, 2, 1, 1]},var=var)
            graph = self.concate_subgraph_to_node(initial_graph, A_graph, 0, 'log')
            graph['edge_attr'][0] = math.e
        if use_template=='log(A)':
            graph= {'nodes': ['log',var], 'edges': [[0,1]], 'edge_attr': [10]}
        if use_template == 'ln(A)':
            graph = {'nodes': ['log', var], 'edges': [[0, 1]], 'edge_attr': [math.e]}
        return graph

    def generate_triangle_template(self,var='x'):
        Template = ['tri(Cx)', 'tri(C*pi*x)', 'tri(pi*x+C)', 'tri(2*pi*x+C)']
        use_template = random.choice(Template)
        Operator=random.choice(Triangle_Operator)
        if use_template == 'tri(Cx)':
            graph = {'nodes': [Operator,var], 'edges': [[0,1]], 'edge_attr': [random.choice([-2,-1,1,2,-1e8])]}
        if use_template == 'tri(C*pi*x)':
            graph = {'nodes': [Operator,'mul',var], 'edges': [[0,1],[1,2]], 'edge_attr': [math.pi,random.choice([-2,-1,1,2,-1e8])]}
        if use_template == 'tri(pi*x+C)':
            graph = {'nodes': [Operator,'add', 'mul', var, '1'], 'edges': [[0,1],[1,2],[2,3],[1,4]],
                     'edge_attr': [1,1,math.pi,-1e8]}
        if use_template == 'tri(2*pi*x+C)':
            graph = {'nodes': [Operator, 'add', 'mul', var, '1'], 'edges': [[0, 1], [1, 2], [2, 3], [1, 4]],
                     'edge_attr': [1, 1, 2*math.pi, -1e8]}
        return graph

    def generate_sinh_template(self,var='x'):
        Template = ['sinh(Cx)']
        use_template = random.choice(Template)
        if use_template == 'sinh(Cx)':
            graph = {'nodes': ['sinh', var], 'edges': [[0, 1]], 'edge_attr': [random.choice([-2, -1, 1, 2, -1e8])]}
        return graph
    def generate_e_exponential_template(self,var='x'):
        Template=['E_exp(x)','E_exp(Cx)']
        use_template=random.choice(Template)
        if use_template=='E_exp(x)':
            graph = {'nodes': ['E_exp'], 'edges': [], 'edge_attr': []}
            template = self.generate_single_poly(var=var)
            graph = self.concate_subgraph_to_node(graph, template, 0, 'E_exp')
        if use_template=='E_exp(Cx)':
            graph = {'nodes': ['E_exp','mul'], 'edges': [[0,1]], 'edge_attr': [random.choice([-1, 1, -1e8])]}
            template = self.generate_single_poly(var=var)
            graph = self.concate_subgraph_to_node(graph, template, 1, 'mul')
        return graph
    def generate_exp_template(self,var='x'):
        '''
        Here, exp is x^n
        :param var:
        :return:
        '''
        graph = {'nodes': ['exp'], 'edges': [], 'edge_attr': []}
        #Template = ['exp-log', 'exp-tri','exp-x']
        #use_template = random.choices(Template, weights=[0.2, 0.2, 0.6], k=1)[0]
        use_template='exp-x'

        if use_template=='exp-x':
            template=self.generate_polynomial_template(var=var)
        elif use_template=='exp-log':
            template = self.generate_log_template(var=var)
        elif use_template=='exp-tri':
            template = self.generate_triangle_template(var=var)
        graph = self.concate_subgraph_to_node(graph, template, 0, 'exp')
        return graph

    def generate_strain_rate_graph_template(self):
        '''
        Generate from template
        In this project, x is stress, x1 is temperature
        Now we have template including:
        [x]:'x','x*x','x/x','expx','log','tri','x*expx','x*tri','x*log','x+expx','x+log','x+tri','E_expx'
        [x1]:'x1','x1/x1','expx1','logx1','E_expx1','x1*E_expx1'
        '''
        var='x'
        template=['x','log','expx','E_expx','x/x']
        choice_prob=np.array([1,1,1,1,1])
        temp=random.choices(template,weights=choice_prob/np.sum(choice_prob),k=1)[0]

        if temp=='x':
            graph=self.generate_polynomial_template()
        elif temp=='x*x':
            graph= {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph=self.concate_subgraph_to_node(graph,self.generate_polynomial_template(),0,'mul')
            graph=self.concate_subgraph_to_node(graph,self.generate_polynomial_template(),0,'mul')
        elif temp=='x*tri':
            graph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'mul')
            graph = self.concate_subgraph_to_node(graph, self.generate_triangle_template(), 0, 'mul')
        elif temp == 'x*log':
            graph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'mul')
            graph = self.concate_subgraph_to_node(graph, self.generate_log_template(), 0, 'mul')
        elif temp=='expx':
            graph=self.generate_exp_template()
        elif temp=='x/x':
            subgraph = {'nodes': ['exp'], 'edges': [], 'edge_attr': []}
            template = self.generate_polynomial_template()
            subgraph = self.concate_subgraph_to_node(subgraph, template, 0, 'exp')
            subgraph['edge_attr'][0]=-1
            graph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'mul')
            graph = self.concate_subgraph_to_node(graph, subgraph, 0, 'mul')
        elif temp=='x*expx':
            graph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'mul')
            graph = self.concate_subgraph_to_node(graph, self.generate_exp_template(), 0, 'mul')
        elif temp=='x+expx':
            graph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'add')
            graph = self.concate_subgraph_to_node(graph, self.generate_exp_template(), 0, 'add')
        elif temp=='x+log':
            graph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'add')
            graph = self.concate_subgraph_to_node(graph, self.generate_log_template(), 0, 'add')
        elif temp == 'x+tri':
            graph = {'nodes': ['add'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(), 0, 'add')
            graph = self.concate_subgraph_to_node(graph, self.generate_triangle_template(), 0, 'add')
        elif temp=='log':
            graph=self.generate_log_template()
        elif temp=='tri':
            graph=self.generate_triangle_template()
        elif temp=='E_expx':
            graph = self.generate_e_exponential_template()
        elif temp=='sinh':
            graph=self.generate_sinh_template()
        elif 'x*E_expx':
            graph = {'nodes': ['mul'], 'edges': [], 'edge_attr': []}
            graph = self.concate_subgraph_to_node(graph, self.generate_polynomial_template(var=var), 0, 'mul')
            graph = self.concate_subgraph_to_node(graph, self.generate_e_exponential_template(var=var), 0, 'mul')
        return graph

    def generate_random_graph(self):
        '''
        stress=f1(epi)*dyn_stress+f2(epi)
        '''
        f1_epi=self.generate_strain_rate_graph_template()
        f2_epi=self.generate_strain_rate_graph_template()
        graph= {'nodes': ['add'], 'edges': [], 'edge_attr': []}
        subgraph_f1={'nodes': ['mul'], 'edges': [], 'edge_attr': []}
        subgraph_f1=self.concate_subgraph_to_node(subgraph_f1,f1_epi,concate_node='mul',concate_node_index=0)
        subgraph_f1=self.concate_subgraph_to_node(subgraph_f1, {'nodes': ['x1'], 'edges': [], 'edge_attr': []}
                                                  ,concate_node='mul',concate_node_index=0)
        graph=self.concate_subgraph_to_node(graph,subgraph_f1,concate_node='add',concate_node_index=0)
        graph = self.concate_subgraph_to_node(graph, f2_epi, concate_node='add', concate_node_index=0)
        return graph


class Graph_to_sympy():

    def get_nodes_to_subgraphs(self,graph):
        operator_map = {
            'add': lambda a, b: a + b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b,
            'log': lambda a: log(a, 10),
            'ln': lambda a: ln(a),
            'sqrt': lambda a: sqrt(a),
            'exp': lambda a: a,  # exp(x) -> x^C, where C is an unknown coefficient
            'sin': lambda a: sin(a),
            'cos': lambda a: cos(a),
            'tan': lambda a: tan(a),
            'csc': lambda a: csc(a),
            'sec': lambda a: sec(a),
            'cot': lambda a: cot(a),
            'sinh': lambda a: sinh(a),
            'tanh': lambda a: tanh(a),
            'cosh': lambda a: cosh(a),
            'atan': lambda a: atan(a),
            'asin': lambda a: asin(a),
            'acos': lambda a: acos(a),
            'atanh': lambda a: atanh(a),
            'asinh': lambda a: asinh(a),
            'acosh': lambda a: acosh(a),
            'E_exp': lambda a: E ** a
        }

        nodes, edges, edge_attr = graph['nodes'], graph['edges'], graph['edge_attr']
        # Create a dictionary to hold the expressions for each node
        expressions = {}
        visited = set()  # To track nodes that are currently being processed
        symbol_index = 0  # To track how many unknown coefficients we've generated

        # Assuming new symbols are generated for edge_attr = -1
        unknown_symbol_dict = {i: symbols(f'C{i}') for i in range(len(edges) + 5)}  #

        # Function to evaluate the expression for a given node
        def evaluate_node(node_index):
            # If the node has already been evaluated, return its expression
            if node_index in expressions:
                return expressions[node_index]

            # If the node is currently being processed, we've detected a cycle
            if node_index in visited:
                raise RecursionError(f"Cyclic dependency detected at node {node_index}")

            # Mark the current node as being processed
            visited.add(node_index)

            # Get the current node's operation or variable
            node_value = nodes[node_index]
            # If the node is a constant (e.g., pi or 1)
            if node_value == 'pi':
                expressions[node_index] = pi
            elif node_value == 'x':
                expressions[node_index] = x
            elif node_value == 'x1':
                expressions[node_index] = x1
            elif node_value == 'x2':
                expressions[node_index] = x2
            elif node_value == 'x3':
                expressions[node_index] = x3
            elif node_value == '1':
                expressions[node_index] = 1
            # If the node is a unary operator
            elif node_value in Unary_Operator_ln + Unary_Operator_exp + Triangle_Operator + Arctriangle_Operator + [
                'E_exp']:
                # For unary operators, only one edge should point to it
                child_node_index = [edges[i][1] for i in range(len(edges)) if edges[i][0] == node_index][
                    0]  # Find the parent node
                edge_index = [i for i in range(len(edges)) if edges[i][0] == node_index][0]
                if node_value == 'log':
                    if evaluate_node(child_node_index) in [0, -1]:
                        expressions[node_index] = 1
                    elif (type(evaluate_node(child_node_index)).__name__.lower() in ['int', 'float', 'integer',
                                                                                     'rational']) and (
                            float(evaluate_node(child_node_index)) < 0):
                        expressions[node_index] = 1
                    else:
                        if abs(edge_attr[edge_index] - math.e) < 1e-5:
                            expressions[node_index] = operator_map['ln'](evaluate_node(child_node_index))
                        else:
                            expressions[node_index] = operator_map[node_value](evaluate_node(child_node_index))

                elif node_value in Triangle_Operator + Arctriangle_Operator:
                    if edge_attr[edge_index] == -1e8:
                        expressions[node_index] = operator_map[node_value](
                            unknown_symbol_dict[edge_index] * evaluate_node(child_node_index))
                    else:
                        expressions[node_index] = operator_map[node_value](
                            edge_attr[edge_index] * evaluate_node(child_node_index))
                else:
                    expressions[node_index] = operator_map[node_value](evaluate_node(child_node_index))
            # If the node is a binary operator (add, mul)
            elif node_value in Binary_Operator:
                # Collect the child nodes of this node
                child_nodes = [edges[i][1] for i in range(len(edges)) if edges[i][0] == node_index]
                child_egdes = [i for i in range(len(edges)) if edges[i][0] == node_index]

                # Apply the appropriate operation (add, mul, div)
                if node_value == 'add':
                    addition = 0
                    for iter, child in enumerate(zip(child_nodes, child_egdes)):
                        if edge_attr[child[1]] == -1e8:
                            addition += evaluate_node(child[0]) * unknown_symbol_dict[child[1]]
                        else:
                            addition += evaluate_node(child[0]) * edge_attr[child[1]]
                    expressions[node_index] = addition
                elif node_value == 'mul':
                    product = 1
                    unknown_flag = 0
                    for iter, child in enumerate(zip(child_nodes, child_egdes)):
                        if edge_attr[child[1]] == -1e8:
                            unknown_flag += 1
                            product *= (evaluate_node(child[0]))
                        else:
                            product *= (evaluate_node(child[0]) * edge_attr[child[1]])
                    if unknown_flag == 0:
                        expressions[node_index] = product
                    else:
                        expressions[node_index] = product * unknown_symbol_dict[child[1]]

            # If the edge_attr is -1, we introduce an unknown coefficient (C)
            if nodes[node_index] == 'exp':  # For non-exponentiation nodes
                child_egdes = [i for i in range(len(edges)) if edges[i][0] == node_index]

                if edge_attr[child_egdes[0]] != -1e8:
                    if expressions[node_index] == 0 and edge_attr[child_egdes[0]] <= 0:
                        expressions[node_index] = 1
                    elif expressions[node_index] == -1 and edge_attr[child_egdes[0]] in [0.5, -0.5]:
                        expressions[node_index] = 1
                    elif type(expressions[node_index]).__name__.lower() in ['int', 'float', 'integer',
                                                                            'rational'] and float(
                            evaluate_node(child_node_index)) < 0 and edge_attr[child_egdes[0]] in [0.5, -0.5]:
                        expressions[node_index] = 1
                    else:
                        expressions[node_index] = expressions[node_index] ** edge_attr[child_egdes[0]]
                else:
                    if expressions[node_index] == 0:
                        expressions[node_index] = 1
                    else:
                        expressions[node_index] = expressions[node_index] ** unknown_symbol_dict[child_egdes[0]]

            # Remove the node from the visited set as we're done processing it
            visited.remove(node_index)

            # Return the evaluated expression for the node
            return expressions[node_index]

        # Start the evaluation from the root nodes (ones that do not have incoming edges)
        for node_index in range(len(nodes)):
            if node_index not in [edge[1] for edge in edges]:
                evaluate_node(node_index)

        if type(expressions[0]).__name__.lower() not in ['int', 'float', 'integer', 'rational']:
            if expressions[0].has(I) == True:
                expressions[0] = 1
        # The root node will be the last evaluated expression
        return expressions
    def graph_to_sympy(self,graph):
        expressions=self.get_nodes_to_subgraphs(graph)
        # The root node will be the last evaluated expression
        return expressions[0]#expressions[0]*A+B

        # Example input (nodes, edges, edge_attr as described)

    def filter_nodes_by_variable(self, graph, vars):
        """
        筛选出仅包含给定变量 var 的节点。

        参数：
            expressions: dict, 节点编号到 sympy 表达式的映射
            var: sympy 符号，例如 x2

        返回：
            过滤后的字典，仅保留 free_symbols 集合恰好等于 {var} 的节点。
        """
        expressions = self.get_nodes_to_subgraphs(graph)
        filtered = {}
        for node, expr in expressions.items():
            if type(expr).__name__.lower() not in ['int', 'float', 'integer', 'rational']:
                free_symbols = expr.free_symbols
                reserve_list = [s for s in sp_symbol_list if s not in vars]
                # 若表达式中的自由符号恰好为 {var}，则保留该节点
                if set(vars).issubset(free_symbols) and set(reserve_list).isdisjoint(free_symbols):
                    filtered[node] = expr
        return filtered


class Genetic_algorithm(Random_graph_for_expr,Graph_to_sympy):
    def __init__(self,x_data,y_data):
        super().__init__()  # 调用父类的构造函数
        self.x_data=x_data
        self.y_data=y_data
        self.size_pop=500
        self.generation_num=150
        self.distinction_epoch=5
        self.max_edge_num=30
        self.max_variable_num=7
        self.use_parallel_computing=True
        self.seek_best_initial=True
        self.epi=0.2
        self.max_unconstant=3


    def renumber_subgraph(self,graph,node_indice):
        """
        Renumber the nodes of a subgraph starting from 0, and adjust the edges accordingly.

        Args:
        nodes (list): A list of nodes in the subgraph.
        edges (list): A list of edges, where each edge is represented by a pair of node indices.

        Returns:
        tuple: A tuple containing:
            - The renumbered nodes (list)
            - The renumbered edges (list)
        """
        nodes=node_indice
        edges=graph['edges']
        # Step 1: Create a mapping from the original nodes to the new renumbered nodes.
        node_mapping = {node: i for i, node in enumerate(sorted(nodes))}
        # Step 2: Renumber the nodes in the edges based on the new node_mapping
        renumbered_edges = [[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in edges]

        # Step 3: Return the renumbered nodes (sorted and starting from 0) and edges.
        renumbered_nodes = list(node_mapping.values())  # New nodes are just the renumbered indices

        graph['edges']=renumbered_edges
        return graph

    def extract_subgraph(self,graph, root):
        nodes, edges, edge_attr = graph['nodes'], graph['edges'], graph['edge_attr']
        subgraph_nodes = []
        subgraph_edges = []
        subgraph_edge_attr = []

        queue = [root]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            subgraph_nodes.append(current)

            for edge, attr in zip(edges, edge_attr):
                if edge[0] == current:
                    subgraph_edges.append(edge)
                    subgraph_edge_attr.append(attr)
                    queue.append(edge[1])
        subgraph_nodes=sorted(subgraph_nodes)
        return subgraph_nodes, subgraph_edges, subgraph_edge_attr

    def delete_subgraph_from_node(self, graph, node_index):
        subgraph_nodes, subgraph_edges, subgraph_edge_attr = self.extract_subgraph(graph, node_index)
        graph_node_index = [i for i in range(len(graph['nodes']))]
        graph['nodes'] = [element for index, element in enumerate(graph['nodes']) if index not in subgraph_nodes]
        graph_node_index = [element for index, element in enumerate(graph_node_index) if index not in subgraph_nodes]
        new_edges = []
        new_edge_attr = []
        for index, edge_info in enumerate(zip(graph['edges'], graph['edge_attr'])):
            if (edge_info[0] not in subgraph_edges) and (edge_info[0][1] not in subgraph_nodes):
                new_edges.append(edge_info[0])
                new_edge_attr.append(edge_info[1])
        graph['edges'] = new_edges
        graph['edge_attr'] = new_edge_attr
        return graph,graph_node_index

    def cross_over(self, graph1, graph2):
        """
        Perform crossover by exchanging subgraphs between two graphs.
        """
        graph1 = deepcopy(graph1)
        graph2 = deepcopy(graph2)
        # print(graph1)
        # print(graph2)
        # Select a random node in graph1 as the root of the subgraph to replace

        # x for x and [x1,x2,x3] for [x1,x2,x3]
        filter_param_graph_1 = list(self.filter_nodes_by_variable(graph1, [x]).keys())
        filter_param_graph_2 = list(self.filter_nodes_by_variable(graph2, [x]).keys())
        if len(filter_param_graph_1) == 0 or len(filter_param_graph_2) == 0:
            return graph1
        node1 = random.choice(filter_param_graph_1)
        max_node1_num = len(graph1['nodes'])
        if node1 != 0:
            parent_node_index = \
            [graph1['edges'][i][0] for i in range(len(graph1['edges'])) if graph1['edges'][i][1] == node1][0]
            parent_node_attr = \
            [graph1['edge_attr'][i] for i in range(len(graph1['edges'])) if graph1['edges'][i][1] == node1][0]

        # Select a random node in graph2 as the root of the subgraph to use
        node2 = random.choice(filter_param_graph_2)
        # print(node1,node2)
        # Extract subgraph from graph2 starting at node2

        # Extract subgraph from graph2
        sub_nodes_2, sub_edges_2, sub_edge_attr_2 = self.extract_subgraph(graph2, node2)
        sub_nodes_1, sub_edges_1, sub_edge_attr_1 = self.extract_subgraph(graph1, node1)

        # Delete the subgraph
        graph1_node_index = [i for i in range(len(graph1['nodes']))]
        graph1['nodes'] = [element for index, element in enumerate(graph1['nodes']) if index not in sub_nodes_1]
        graph1_node_index = [element for index, element in enumerate(graph1_node_index) if index not in sub_nodes_1]
        new_edges_1 = []
        new_edge_attr_1 = []
        for index, edge_info in enumerate(zip(graph1['edges'], graph1['edge_attr'])):
            if edge_info[0] not in sub_edges_1:
                if edge_info[0][1] not in sub_nodes_1:
                    new_edges_1.append(edge_info[0])
                    new_edge_attr_1.append(edge_info[1])
        graph1['edges'] = new_edges_1
        graph1['edge_attr'] = new_edge_attr_1
        # print(graph1['nodes'], graph1['edges'])
        # print(sub_nodes_2, sub_edges_2, sub_edge_attr_2)
        # print('cross_over nodes:',node1)
        if node1 == 0:
            graph1_nodes = sub_nodes_2
            graph1['nodes'] = [graph2['nodes'][i] for i in graph1_nodes]
            graph1['edges'] = sub_edges_2
            graph1['edge_attr'] = sub_edge_attr_2
            graph1 = self.renumber_subgraph(graph1, graph1_nodes)

        else:

            if len(sub_edges_2) != 0:
                sub_edges_2 = (np.array(sub_edges_2) + max_node1_num).tolist()
                nodes2_min = np.min(np.array(sub_edges_2))
                graph1['edges'].append([parent_node_index, nodes2_min])
                graph1_node_index.extend((np.array(sub_nodes_2) + max_node1_num).tolist())
            else:
                graph1['edges'].append([parent_node_index, parent_node_index + max_node1_num])
                graph1_node_index.extend([parent_node_index + max_node1_num])

            graph1['edge_attr'].append(parent_node_attr)

            graph1['nodes'].extend([graph2['nodes'][i] for i in sub_nodes_2])
            graph1['edges'].extend(sub_edges_2)
            graph1['edge_attr'].extend(sub_edge_attr_2)

            # print(graph1_node_index,graph1['edges'])
            graph1 = self.renumber_subgraph(graph1, graph1_node_index)
        # print('after cross over:',graph1)
        # pyG_graph = convert_graph_to_pyG(graph1)
        # plot_graph_with_features(pyG_graph)
        # plt.show()
        return graph1

    def mutate(self, graph, node_mutation_rate=0.3, graph_mutation_rate=0.5, graph_delete_graph=0.2,
               mutate_edge_attr_prob=0.5):
        """
        Perform mutation by modifying nodes and edges randomly.
        """
        graph = deepcopy(graph)
        # print(graph)
        num_nodes = len(graph['nodes'])
        edges = graph['edges']

        # Mutate nodes
        for i in range(num_nodes):
            if random.random() < node_mutation_rate:
                # print('mutate nodes')
                if graph['nodes'][i] == 'log':
                    edge_index = [j for j in range(len(edges)) if edges[j][0] == i][0]
                    graph['edge_attr'][edge_index] = random.choice([10, math.e])
                if graph['nodes'][i] in Triangle_Operator:
                    graph['nodes'][i] = random.choice(Triangle_Operator)


        # Mutate subgraphs--search more complex forms:
        filter_param_graph = list(self.filter_nodes_by_variable(graph, [x]).keys())
        if (random.random() < graph_mutation_rate) and (len(graph['nodes'])) > 1 and len(filter_param_graph) != 0:
            # print('mutate subgraphs')
            num_nodes = len(graph['nodes'])
            node_indices = [i for i in range(len(graph['nodes']))]

            # mutate_node= random.randint(1, len(graph['nodes']) - 1)
            mutate_node = random.choice(filter_param_graph)
            if mutate_node != 0:
                parent_node_index = \
                    [graph['edges'][i][0] for i in range(len(graph['edges'])) if graph['edges'][i][1] == mutate_node][0]

                if graph['nodes'][mutate_node] in ['mul', 'add']:
                    child_edge_indice = [i for i in range(len(edges)) if edges[i][0] == mutate_node]
                    mutate_edge_index = random.choice(child_edge_indice)
                    mutate_node_index = edges[mutate_edge_index][1]
                    graph, node_indices = self.delete_subgraph_from_node(graph, mutate_node_index)

                    select_var = 'x'

                    if select_var == 'x':
                        select_template = random.choices(['exp', 'x', 'log'], weights=[0.3, 0.4, 0.3], k=1)[0]

                    if select_template == 'log':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_log_template(
                            var=select_var),
                                                                                    mutate_node,
                                                                                    graph['nodes'][mutate_node],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    elif select_template == 'tri':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                    self.generate_triangle_template(
                                                                                        var=select_var),
                                                                                    mutate_node,
                                                                                    graph['nodes'][mutate_node],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    elif select_template == 'poly':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                    self.generate_polynomial_template(
                                                                                        var=select_var),
                                                                                    mutate_node,
                                                                                    graph['nodes'][mutate_node],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    elif select_template == 'exp':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                    self.generate_exp_template(
                                                                                        var=select_var),
                                                                                    mutate_node,
                                                                                    graph['nodes'][mutate_node],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    elif select_template == 'x':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                    self.generate_single_poly(
                                                                                        var=select_var),
                                                                                    mutate_node,
                                                                                    graph['nodes'][mutate_node],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    elif select_template == 'sinh':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                    self.generate_sinh_template(
                                                                                        var=select_var),
                                                                                    mutate_node,
                                                                                    graph['nodes'][mutate_node],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    elif select_template == 'E_expx':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                    self.generate_e_exponential_template(
                                                                                        var=select_var),
                                                                                    mutate_node,
                                                                                    graph['nodes'][mutate_node],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    node_indices = node_indices + concate_node_indices
                    graph = self.renumber_subgraph(graph, node_indices)

                elif graph['nodes'][mutate_node] in ['log'] + Triangle_Operator + ['E_exp']:
                    graph, node_indices = self.delete_subgraph_from_node(graph, mutate_node)
                    select_var = 'x'

                    if select_var == 'x':
                        select_template = random.choices(['exp', 'x', 'log'], weights=[0.3, 0.4, 0.3], k=1)[0]

                    if select_template == 'log':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_log_template(
                            var=select_var), parent_node_index,
                                                                                    graph['nodes'][parent_node_index],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    if select_template == 'poly':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                    self.generate_polynomial_template(
                                                                                        var=select_var),
                                                                                    parent_node_index,
                                                                                    graph['nodes'][parent_node_index],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    if select_template == 'tri':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                    self.generate_triangle_template(
                                                                                        var=select_var),
                                                                                    parent_node_index,
                                                                                    graph['nodes'][parent_node_index],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    if select_template == 'exp':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_exp_template(
                            var=select_var), parent_node_index,
                                                                                    graph['nodes'][parent_node_index],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    if select_template == 'x':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_single_poly(
                            var=select_var), parent_node_index,
                                                                                    graph['nodes'][parent_node_index],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    if select_template == 'sinh':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_sinh_template(
                            var=select_var), parent_node_index,
                                                                                    graph['nodes'][parent_node_index],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    if select_template == 'E_expx':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph,
                                                                                    self.generate_e_exponential_template(
                                                                                        var=select_var),
                                                                                    parent_node_index,
                                                                                    graph['nodes'][parent_node_index],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    node_indices = node_indices + concate_node_indices
                    graph = self.renumber_subgraph(graph, node_indices)
                elif graph['nodes'][mutate_node] == 'exp':
                    graph, node_indices = self.delete_subgraph_from_node(graph, mutate_node)
                    select_var = 'x'

                    if select_var == 'x':
                        graph, concate_node_indices = self.concate_subgraph_to_node(graph, self.generate_single_poly(),
                                                                                    parent_node_index,
                                                                                    graph['nodes'][parent_node_index],
                                                                                    set_maximum_node_num=num_nodes,
                                                                                    with_node_indice=node_indices)
                    node_indices = node_indices + concate_node_indices
                    graph = self.renumber_subgraph(graph, node_indices)

            # delete edges
            filter_param_graph = list(self.filter_nodes_by_variable(graph, [x]).keys())
            if random.random() < graph_delete_graph and len(filter_param_graph) != 0:
                mutate_node = random.choice(filter_param_graph)
                # print('delete edges')
                if len(graph['nodes']) > 1 and mutate_node != 0:
                    # mutate_node= random.randint(1, len(graph['nodes']) - 1)
                    parent_node_index = \
                        [graph['edges'][i][0] for i in range(len(graph['edges'])) if
                         graph['edges'][i][1] == mutate_node][0]
                    parent_node_attr = \
                        [graph['edge_attr'][i] for i in range(len(graph['edges'])) if
                         graph['edges'][i][1] == mutate_node][0]
                    mutate_node_value = graph['nodes'][mutate_node]
                    max_node_num = len(graph['nodes'])
                    subgraph_nodes, subgraph_edges, subgraph_edge_attr = self.extract_subgraph(graph, mutate_node)
                    graph_node_index = [i for i in range(len(graph['nodes']))]
                    graph['nodes'] = [element for index, element in enumerate(graph['nodes']) if
                                      index not in subgraph_nodes]
                    graph_node_index = [element for index, element in enumerate(graph_node_index) if
                                        index not in subgraph_nodes]
                    new_edges_1 = []
                    new_edge_attr_1 = []
                    for index, edge_info in enumerate(zip(graph['edges'], graph['edge_attr'])):
                        if edge_info[0] not in subgraph_edges:
                            if edge_info[0][1] not in subgraph_nodes:
                                new_edges_1.append(edge_info[0])
                                new_edge_attr_1.append(edge_info[1])
                    graph['edges'] = new_edges_1
                    graph['edge_attr'] = new_edge_attr_1
                    graph['nodes'].append('1')
                    graph_node_index.append(max_node_num + 1)
                    graph['edges'].append([parent_node_index, max_node_num + 1])
                    graph['edge_attr'].append(1)
                    graph = self.renumber_subgraph(graph, graph_node_index)

        # mutate edge_attr
        filter_param_graph = list(self.filter_nodes_by_variable(graph, [x]).keys())
        for mutate_edge_attr_index in range(len(graph['edge_attr'])):
            if random.random() < mutate_edge_attr_prob:
                # print('mutate edge_attr')
                mutate_edge = graph['edges'][mutate_edge_attr_index]
                begin_node = graph['nodes'][mutate_edge[0]]
                if begin_node == 'add':
                    graph['edge_attr'][mutate_edge_attr_index] = random.choices([1, -1, -1e8], [0.45, 0.45, 0.1], k=1)[
                        0]
                elif begin_node == 'exp':
                    if begin_node in filter_param_graph:
                        graph['edge_attr'][mutate_edge_attr_index] = \
                        random.choices([-2, -1, 0, 1, 2,-1e8], [0.1, 0.15, 0.1, 0.15, 0.1,0.1], k=1)[0]
                elif begin_node == 'log':
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([10, math.e])
                elif begin_node in Triangle_Operator:
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([1, 2, math.pi, 2 * math.pi, -1e8])
                elif begin_node == 'mul':
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([1, -1, -1e8])
                elif begin_node == 'E_exp':
                    graph['edge_attr'][mutate_edge_attr_index] = random.choice([1, -1, -1e8])

        # print(graph)
        # print('--------------------')
        return graph

    def generate_initial_guesses(self,len_variables, num_samples=50):
        """
        Generate initial guesses for parameters.
        :param bounds: List of (min, max) tuples for each parameter.
        :param num_samples: Number of initial guesses to generate.
        :return: List of initial guesses.
        """
        guesses = []
        for _ in range(num_samples):
            guess=[1,1]
            for _ in range(len_variables):
                guess.append(np.random.choice([-10,-2,-1,1,2,10]))
            guesses.append(guess)
        return guesses

    # Find the best initial guess
    def find_best_initial_guess(self,len_variables, lambdified_expr,x_data, y_data):
        """
        Find the best initial guess for optimization.
        :param bounds: List of (min, max) tuples for each parameter.
        :param lambdified_expr: Lambdified sympy expression.
        :param x_data: Array of x values.
        :param y_data: Array of y values.
        :param num_samples: Number of guesses to evaluate.
        :return: Best initial guess.
        """
        guesses = self.generate_initial_guesses(len_variables)

        best_guess = None
        best_score = float('inf')

        def objective(params, lambdified_expr, x_data, y_data):
            y_pred = lambdified_expr(*x_data, *params)
            mse = np.mean((y_pred - y_data) ** 2)
            return mse

        for guess in guesses:
            result = minimize(objective, guess, args=(lambdified_expr, x_data, y_data), method='BFGS',
                              options={'disp': False})
            score= result.fun
            #score = objective(guess, lambdified_expr, x_data, y_data)
            #print('guess:',guess,score)
            if score < best_score:
                best_score = score
                best_guess = guess
        if best_guess==None:
            best_guess=guesses[0]
        # print('best_score:',best_score)
        # print('best_guess:',best_guess)
        return best_guess

    def get_fitness_from_graph(self,graph):
        expr=self.graph_to_sympy(graph)
        #print(expr)
        variables = list(expr.free_symbols)

        variables=[x,x1,x2,x3,x4,A,B]+[sym for sym in variables if sym not in [x,x1,x2,x3,x4,A,B]]
        def lambdify_expression(expr, symbols):
            return lambdify(symbols, expr, 'numpy')
        lambdified_expr=lambdify_expression(expr, variables)

        def objective(params, lambdified_expr, x_data, y_data):
            # For each expression, calculate the corresponding y values and sum them
            y_pred = lambdified_expr(*x_data, *params)  # Sum the contributions from each expression
            MSE = np.mean((y_pred - y_data) ** 2)
            return MSE
        #print(variables)
        fitness=0
        optimal_params=[]

        for x_d, y_d in zip(self.x_data, self.y_data):
            if self.seek_best_initial == True:
                best_guess = self.find_best_initial_guess(len(variables) -7, lambdified_expr,
                                                          x_d, y_d)
            else:
                best_guess = np.ones([len(variables) - 1])
            # print('best_guess_use:',best_guess)
            result = minimize(objective, best_guess, args=(lambdified_expr, x_d, y_d),
                              method='BFGS', options={'disp': False})

            # print('optimized result:',result.x)
            optimal_params.append(result.x)
            fitness += result.fun
        fitness=fitness/len(self.y_data)#*(1+self.epi*len(variables)+self.epi*len(graph['nodes']))
        if isinstance(fitness, np.complex128)==True:
            fitness=1e8
        #print(optimal_params,fitness)
        #y_pred=prediction(optimal_params,lambdified_expr,x_data)
        # print(y_pred)
        # print(y_data)

        return fitness,optimal_params

    def get_fitness_from_expr(self,expr,x_data,y_data):
        '''
        Used for high-performance parallel computing
        since graph_to_sympy cannot be parallelled
        '''
        try:
            variables = list(expr.free_symbols)
            variables = [x, x1, x2, x3, x4, A, B] + [sym for sym in variables if sym not in [x, x1, x2, x3, x4, A, B]]
        except AttributeError:
            return 1e8,0

        def lambdify_expression(expr, symbols):
            return lambdify(symbols, expr, 'numpy')

        lambdified_expr = lambdify_expression(expr, variables)

        def objective(params, lambdified_expr, x_data, y_data):
            # For each expression, calculate the corresponding y values and sum them
            y_pred = lambdified_expr(*x_data, *params)  # Sum the contributions from each expression
            MSE = np.mean((y_pred - y_data) ** 2)
            return MSE
            # print(variables)

        fitness = 0
        optimal_params = []

        for x_d, y_d in zip(self.x_data, self.y_data):
            if self.seek_best_initial == True:
                best_guess = self.find_best_initial_guess(len(variables) -7, lambdified_expr,
                                                          x_d, y_d)
            else:
                best_guess = np.ones([len(variables) - 1])
            # print('best_guess_use:',best_guess)
            result = minimize(objective, best_guess, args=(lambdified_expr, x_d, y_d),
                              method='BFGS', options={'disp': False})

            # print('optimized result:',result.x)
            optimal_params.append(result.x)
            fitness += result.fun
        fitness = fitness / len(y_data)#*(1+self.epi*len(variables))
        if isinstance(fitness, np.complex128)==True:
            fitness=1e8
        #fitness=total_relative_error/len(y_data)*(1+self.epi*len(variables))

        # print(optimal_params,fitness)
        # y_pred=prediction(optimal_params,lambdified_expr,x_data)
        # print(y_pred)
        # print(y_data)

        return fitness, optimal_params

    def get_regressed_function(self,graph):
        expr = self.graph_to_sympy(graph)
        variables = list(expr.free_symbols)
        variables = [A, B] + [sym for sym in variables if sym not in [x,x1,x2,x3,x4, A, B]]
        fitness,optimal_params=self.get_fitness_from_graph(graph)
        simplified_expr=[]
        for j in range(len(self.y_data)):
            param_dict={}
            for i in range(len(variables)):
                param_dict[variables[i]]=optimal_params[j][i]
            substituted_expr = expr.subs(param_dict)
            simplified_expr.append(sp.simplify(substituted_expr))
        return simplified_expr
    def get_regressed_function_from_expr(self,expr,optimal_params):
        variables = list(expr.free_symbols)
        variables = [A, B] + [sym for sym in variables if sym not in [x,x1,x2,x3,x4, A, B]]
        simplified_expr=[]
        for j in range(len(self.y_data)):
            param_dict={}
            for i in range(len(variables)):
                param_dict[variables[i]]=optimal_params[j][i]
            substituted_expr = expr.subs(param_dict)
            simplified_expr.append(sp.simplify(substituted_expr))
        return simplified_expr
    def get_function_prediction(self,expr,x_data):
        def lambdify_expression(expr, symbols):
            return lambdify(symbols, expr, 'numpy')

        # Create the lambda function for the expression
        lambdified_expr = lambdify_expression(expr, [x,x1,x2,x3,x4])

        # Use the lambdified function to calculate y_data for each x in x_data
        pred = lambdified_expr(*x_data)
        return pred
    def sorted(self,graph_list,fitness_list):
        combined = list(zip(graph_list, fitness_list))

        combined_sorted = sorted(combined, key=lambda x: x[1])

        graph_sorted, fitness_sorted = zip(*combined_sorted)
        graph_sorted = list(graph_sorted)
        fitness_sorted = list(fitness_sorted)
        return graph_sorted,fitness_sorted

    def distinction(self,graphs):
        for i in range(1,len(graphs)):
            graphs[i]=self.generate_random_graph()
        return graphs

    def elimiate_length(self,graph):
        flag=1
        max_iter_num=10
        iter_num=0
        while flag==1:
            edge_length=len(graph['edges'])
            expr = self.graph_to_sympy(graph)
            try:
                variables = list(expr.free_symbols)
            except AttributeError:
                return graph
            len_variable=len(variables)
            graph=self.mutate(graph)
            if (edge_length<self.max_edge_num) and (len_variable<self.max_variable_num):
                flag=0
            if iter_num==max_iter_num:
                flag=0
            iter_num+=1
        return graph

    def elimiate_var_num(self,graph):
        edge_attr=graph['edge_attr']
        # Find indices where the value is -1e8
        neg_inf_indices = [i for i, val in enumerate(edge_attr) if val == -1e8]

        # If we already meet the requirement, no change needed
        if len(neg_inf_indices) <= self.max_unconstant:
            return graph

        # Randomly choose which indices to keep as -1e8
        keep_indices = set(random.sample(neg_inf_indices, self.max_unconstant))

        # Create a new list with modifications
        new_edge_attr = []
        for i, val in enumerate(edge_attr):
            if val == -1e8 and i not in keep_indices:
                # Replace with either 1 or -1
                new_edge_attr.append(random.choice([1, -1]))
            else:
                new_edge_attr.append(val)
        graph['edge_attr']=new_edge_attr
        return graph

    def parallel_get_fitness(self,exprs,x_data,y_data, get_fitness_from_expr):
        """
        Compute fitness for all graphs in parallel using multiprocessing.
        x_data,y_data should be amplified
        """
        max_workers = max(os.cpu_count() - 1, 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(get_fitness_from_expr, exprs, x_data, y_data))
        return results
    def evolution(self,save_dir='default'):
        self.graphs=[]
        self.fitnesses=[]
        self.best_graphs_record=[]
        self.best_fitness_record=[]
        self.exprs=[]
        all_x_data=[]
        all_y_data=[]

        if self.use_parallel_computing==True:
            for i in range(self.size_pop):
                graph=self.generate_random_graph()
                graph=self.elimiate_var_num(graph)
                self.graphs.append(graph)
                expr = self.graph_to_sympy(graph)
                self.exprs.append(expr)
                all_x_data.append(self.x_data)
                all_y_data.append(self.y_data)
            results = self.parallel_get_fitness(self.exprs, all_x_data, all_y_data, self.get_fitness_from_expr)
            self.fitnesses = [1e18 if pd.isna(f[0]) else f[0] for f in results]

        if self.use_parallel_computing==False:
            for i in range(self.size_pop):
                graph = self.generate_random_graph()
                self.graphs.append(graph)
                fitness = self.get_fitness_from_graph(graph)[0]
                if pd.isna(fitness) == True:
                    fitness = 1e18
                self.fitnesses.append(fitness)

        self.graphs,self.fitnesses=self.sorted(self.graphs,self.fitnesses)
        print(self.fitnesses[0:5])
        print([self.graph_to_sympy(graph) for graph in self.graphs[0:5]])
        best_graph = {0: self.graphs[0], 1: self.graphs[1], 2: self.graphs[2], 3: self.graphs[3], 4: self.graphs[4]}
        best_fitness = {0: self.fitnesses[0], 1: self.fitnesses[1], 2: self.fitnesses[2], 3: self.fitnesses[3],
                        4: self.fitnesses[4]}
        self.best_graphs_record.append(best_graph)
        self.best_fitness_record.append(best_fitness)

        distinction_flag=0
        for iter_num in tqdm(range(self.generation_num)):
            new_graphs=list(best_graph.values())
            new_fitness_list=list(best_fitness.values())

            for i in range(self.size_pop):
                parent1=self.graphs[i]
                parent2=self.graphs[random.randint(0,self.size_pop-1)]

                # Perform crossover
                offspring = self.cross_over(parent1, parent2)
                #print(self.graph_to_sympy(offspring['nodes'],offspring['edges'],offspring['edge_attr']))
                # Perform mutation
                offspring = self.mutate(offspring)
                offspring=self.elimiate_length(offspring)
                offspring=self.elimiate_var_num(offspring)
                #print(self.graph_to_sympy(offspring['nodes'], offspring['edges'], offspring['edge_attr']))

                if self.use_parallel_computing==False:
                    fitness,coef = self.get_fitness_from_graph(offspring)

                    if pd.isna(fitness)==True:
                        fitness=1e8
                    new_fitness_list.append(fitness)

                new_graphs.append(offspring)

            if self.use_parallel_computing==True:
                new_exprs=[self.graph_to_sympy(graph) for graph in new_graphs[5:]]
                pickle.dump(new_graphs, open(f'result_save/strain_rate_DIF_normal/new_graphs.pkl', 'wb'))
                results = self.parallel_get_fitness(new_exprs, all_x_data, all_y_data, self.get_fitness_from_expr)
                new_fitness_list += [1e18 if pd.isna(f[0]) else f[0] for f in results]



            #sort
            re1 = list(map(new_fitness_list.index, heapq.nsmallest(int(self.size_pop / 2), new_fitness_list)))

            sorted_graph=[]
            sorted_fitness= []
            for index in re1:
                if new_fitness_list[index] not in sorted_fitness:
                    sorted_graph.append(new_graphs[index])
                    sorted_fitness.append(new_fitness_list[index])
            for index in range(self.size_pop-len(sorted_fitness)):
                sorted_graph.append(self.generate_random_graph())
            self.graphs=sorted_graph
            self.fitnesses=sorted_fitness
            print(self.fitnesses[0:5])
            #print([self.graph_to_sympy(graph) for graph in self.graphs[0:5]])
            print([self.graph_to_sympy(graph) for graph in self.graphs[0:5]])
            print('best graph:',self.graphs[0])
            if self.fitnesses[0]==best_fitness[0]:
                distinction_flag+=1
            else:
                distinction_flag=0

            best_graph={0:self.graphs[0],1:self.graphs[1],2:self.graphs[2],3:self.graphs[3],4:self.graphs[4]}
            best_fitness={0:self.fitnesses[0],1:self.fitnesses[1],2:self.fitnesses[2],3:self.fitnesses[3],4:self.fitnesses[4]}
            self.best_graphs_record.append(best_graph)
            self.best_fitness_record.append(best_fitness)
            if distinction_flag==self.distinction_epoch:
                #print('disctinction happens!')
                distinction_flag=0
                self.graphs=self.distinction(self.graphs)

            try:
                os.makedirs(f'result_save/{save_dir}/')
            except OSError:
                pass
            if iter_num==0:
                params = {
                    "size_pop": self.size_pop,
                    "generation_num": self.generation_num,
                    "distinction_epoch": self.distinction_epoch,
                    "max_edge_num": self.max_edge_num,
                    "max_variable_num": self.max_variable_num,
                    "use_parallel_computing": self.use_parallel_computing,
                    "seek_best_initial": self.seek_best_initial,
                    "epi": self.epi,
                    "max_unconstant": self.max_unconstant,
                    "use_var":use_var
                }

                with open(f'result_save/{save_dir}/params.txt', "w") as f:
                    for key, value in params.items():
                        f.write(f"{key}: {value}\n")
                f.close()
            if (iter_num+1)%10==0:
                pickle.dump(self.best_graphs_record,open(f'result_save/{save_dir}/best_graphs.pkl', 'wb'))
                pickle.dump(self.best_fitness_record,open(f'result_save/{save_dir}/best_fitness.pkl', 'wb'))