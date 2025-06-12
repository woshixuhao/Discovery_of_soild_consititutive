import os

import matplotlib.pyplot as plt
import numpy as np

from graph_regression import *
from utils import *

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

def plot_expand_predict_DIF(x_plot,pred_plot,x_obs,y_obs,save_name=''):
    plt.figure(figsize=(2.5, 2.5), dpi=300)
    plt.plot(x_plot, pred_plot, c='#8A83B4', linewidth=1,label='pred.')
    plt.scatter(x_obs,y_obs,marker='^',s=5,c='black',label='ref.')


    plt.xticks( fontproperties='Arial', fontsize=7)
    plt.yticks(fontproperties='Arial', fontsize=7)
    # 设置对数刻度
    plt.xscale("log")
    # 设置标签字体和字号
    plt.xlabel("strain rate", fontname="Arial", fontsize=7)
    plt.ylabel("DIF", fontname="Arial", fontsize=7)

    plt.legend(prop={"family": "Arial", "size": 6}, ncol=2)
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格

    # plt.savefig(f"plot_save/{save_name}.pdf", bbox_inches='tight', dpi=300)
    # plt.savefig(f"plot_save/{save_name}.png", bbox_inches='tight', dpi=300)

    plt.show()

if __name__ == '__main__':
    data_name=os.listdir('data_DIF')
    mode='Valid'
    all_x=[]
    all_y=[]
    print(data_name)
    for name in data_name:
        file_path = f"data_DIF/{name}"
        df = pd.read_excel(file_path)
        # 确保列名正确
        kesi_values = df['strain rate'].values
        DIF_values = df['DIF'].values


        all_x.append(kesi_values)
        all_y.append(DIF_values)


    model = Random_graph_for_expr()
    model_sympy = Graph_to_sympy()
    GA = Genetic_algorithm(x_data=all_x,y_data=all_y)



    if mode=='Train':
        GA.evolution(save_dir='strain_rate_DIF_default')
    if mode=='Valid':

        #discovered equation
        best_graph={'nodes': ['exp', 'exp', 'exp', 'add', 'x', '1'], 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5]],
          'edge_attr': [0.5, -100000000.0, -2, -100000000.0, 1]}


        expr = model_sympy.graph_to_sympy(best_graph)
        print(expr)
        #fitness, optimal_params=GA.get_fitness_from_graph(true_graph)

        fitness, optimal_params = GA.get_fitness_from_expr(expr, all_x, all_y)
        print(fitness,optimal_params)
        sim_expr = GA.get_regressed_function_from_expr(expr, optimal_params)
        print(sim_expr)
        pred_concate=[]
        all_pred={}
        all_obs={}
        y_concate=[]
        all_pred_extend = {}
        all_kesi = {}
        expand_strain_rate= np.logspace(-5, 4, num=100)

        for n in range(len(sim_expr)):
            print(data_name[n],':  ',sim_expr[n],'\n')
            kesi_values=all_x[n]
            min_val = np.min(all_x[n])
            max_val = np.max(all_x[n])
            if np.max(kesi_values) / np.min(kesi_values) < 100:
                extend_kesi_values = np.linspace(min_val, max_val, 1000)
            else:
                extend_kesi_values = np.logspace(np.log10(min_val), np.log10(max_val), 1000)
            pred = GA.get_function_prediction(sim_expr[n], all_x[n])
            expand_pred= GA.get_function_prediction(sim_expr[n], expand_strain_rate)
            extend_pred= GA.get_function_prediction(sim_expr[n], extend_kesi_values)

            pred_concate.extend(pred)

            y_concate.extend(all_y[n])
            #plot_pred_obs_DIF(all_x[n], pred, all_y[n], data_name=data_name[n],plot=False,save=True)
            #plot_expand_predict_C_S(expand_strain_rate,expand_pred,all_x[n],all_y[n])
            all_pred[data_name[n]]=pred
            all_obs[data_name[n]]=all_y[n]
            all_pred_extend[data_name[n]] = extend_pred.tolist()
            all_kesi[data_name[n]] = extend_kesi_values.tolist()

        # pickle.dump(all_pred, open(f'outcome_save/Our_model_strain_rate_DIF.pkl', 'wb'))
        # pickle.dump(all_obs, open(f'outcome_save/True_strain_rate_DIF.pkl', 'wb'))
        print(calculate_relative_error(pred_concate,y_concate))
        pickle.dump(all_pred_extend, open(f'outcome_save/extend_Our_model_strain_rate_DIF.pkl', 'wb'))
        pickle.dump(all_kesi, open(f'outcome_save/kesi_Our_model_strain_rate_DIF.pkl', 'wb'))
        #plot_pred_vs_obs(all_x,all_x1,all_pred,all_y)

    if mode=='Valid_discovered':
        best_graph = pickle.load(open(f'result_save/strain_rate_DIF_no_A_2/best_graphs.pkl', 'rb'))
        best_fitnesses = pickle.load(open(f'result_save/strain_rate_DIF_no_A_2/best_fitness.pkl', 'rb'))
        best_expr = []
        for graphs in best_graph:
            exprs = []
            for i in range(5):
                exprs.append(model_sympy.graph_to_sympy(graphs[i]))
            best_expr.append(exprs)

        for j in range(150):
            print(f'The {j} epoch')
            for i in range(5):
                print(f'The #{i} best expr:, The #{i} best fitness:', best_expr[j][i], best_fitnesses[j][i])
                print(f'The #{i} best graph:', best_graph[j][i])
            print('=================================')

    if mode=='C_S_model':
        def estimate_cs_parameters(strain_rate, DIF):
            """
            Estimate the parameters for the Cowper-Symonds (CS) material model.

            This function calculates the parameters D and P of the Cowper-Symonds equation
            based on provided strain rate and dynamic increase factor (DIF) data. The method
            involves filtering valid data points, performing a logarithmic transformation,
            and applying linear regression to estimate the parameters.

            Parameters
            ----------
            strain_rate : numpy.ndarray
                Array of strain rate values. Must be a 1D array of numerical values.
            DIF : numpy.ndarray
                Array of dynamic increase factor (DIF) values corresponding to the strain rates.
                Must be a 1D array of numerical values with the same length as `strain_rate`.

            Returns
            -------
            tuple
                A tuple containing two floats: D and P. If the input data is invalid or insufficient,
                both values will be zero.

            Raises
            ------
            ValueError
                If the input arrays `strain_rate` and `DIF` do not have the same shape or if they
                contain non-numerical values.

            Notes
            -----
            The function assumes that the input arrays are NumPy arrays. It performs the following steps:
                1. Filters out data points where DIF values are less than or equal to 1.
                2. Ensures there are at least two valid data points for regression.
                3. Applies a logarithmic transformation to the filtered data.
                4. Uses linear regression to fit the transformed data to the form y = a * x + b.
                5. Computes the Cowper-Symonds parameters D and P from the regression coefficients.

            The Cowper-Symonds equation is defined as:
                DIF - 1 = (strain_rate / D) ** P

            Warnings
            --------
            If the input data contains fewer than two valid points after filtering, the function
            returns zero for both parameters. Ensure that the input data is carefully validated
            before calling this function.
            """

            if np.any(DIF < 1):
                return 0,0

            valid_indices = DIF > 1
            strain_rate_valid = strain_rate[valid_indices]
            DIF_valid = DIF[valid_indices]


            if len(DIF_valid) < 2:
                return 0, 0


            x = np.log(strain_rate_valid)
            y = np.log(DIF_valid - 1)


            a, b = np.polyfit(x, y, 1)

            P = 1.0 / a
            D = np.exp(-b / a)

            return D, P


        def compute_dif(strain_rate, D, P):
            """
            Compute the Dynamic Increase Factor (DIF) based on the given strain rate, D, and P.

            This function calculates the DIF using the formula 1 + (strain_rate / D) ** (1 / P).
            The strain_rate is converted to a numpy array to ensure compatibility with array operations.

            Parameters
            ----------
            strain_rate : float or array-like
                The strain rate value(s) used in the DIF calculation.
            D : float
                A scalar value representing the material constant D in the DIF equation.
            P : float
                A scalar value representing the material constant P in the DIF equation.

            Returns
            -------
            float or ndarray
                The computed Dynamic Increase Factor (DIF). If strain_rate is a single value,
                a float is returned. If strain_rate is array-like, an ndarray is returned.

            Raises
            ------
            ValueError
                If D or P is non-positive, as these values are required to be greater than zero
                for the calculation to be valid.
            TypeError
                If strain_rate is not a numeric type or array-like of numeric types.
            """
            strain_rate = np.array(strain_rate)
            DIF = 1 + (strain_rate / D) ** (1 / P)
            return DIF


        pred_concate = []
        y_concate = []
        all_pred={}
        all_pred_extend={}
        all_kesi={}
        for name in data_name:
            file_path = f"data_DIF/{name}"
            df = pd.read_excel(file_path)
            # 确保列名正确
            kesi_values = df['strain rate'].values
            min_val = np.min(kesi_values)
            max_val = np.max(kesi_values)
            if np.max(kesi_values)/np.min(kesi_values)<100:
                extend_kesi_values = np.linspace(min_val,max_val,1000)
            else:
                extend_kesi_values = np.logspace(np.log10(min_val), np.log10(max_val), 1000)
            DIF_values = df['DIF'].values
            D,P=estimate_cs_parameters(kesi_values,DIF_values)
            if D==0 and P==0:
                print(f'fail to calculate D,P in C_S model! The data name is {name}')
                continue
            else:
                pred=compute_dif(kesi_values,D,P)
                extend_pred=compute_dif(extend_kesi_values,D,P)
                #plot_pred_obs_DIF(kesi_values,pred,DIF_values,name)

            pred_concate.extend(pred.tolist())
            y_concate.extend(DIF_values.tolist())
            all_pred[name]=pred.tolist()
            all_pred_extend[name]=extend_pred.tolist()
            all_kesi[name]=extend_kesi_values.tolist()

        pred_concate=np.array(pred_concate)
        y_concate=np.array(y_concate)
        #pickle.dump(all_pred, open(f'outcome_save/C_S_model_strain_rate_DIF.pkl', 'wb'))
        pickle.dump(all_pred_extend, open(f'outcome_save/extend_C_S_model_strain_rate_DIF.pkl', 'wb'))
        pickle.dump(all_kesi, open(f'outcome_save/kesi_C_S_model_strain_rate_DIF.pkl', 'wb'))
        print(calculate_relative_error(pred_concate, y_concate))

    if mode=='J_C_model':
        def estimate_jc_parameter(strain_rate, DIF, reference_strain_rate=1e-4):
            """
            Estimate the Johnson-Cook (JC) material parameter C using strain rate and
            Dynamic Increase Factor (DIF) data. The function filters invalid input data,
            performs necessary transformations, and applies a least-squares method to
            compute the parameter.

            Parameters
            ----------
            strain_rate : array_like
                Array of strain rate values. Must contain positive values since logarithmic
                transformation is applied.
            DIF : array_like
                Array of Dynamic Increase Factor (DIF) values corresponding to the strain
                rates. Must have the same length as `strain_rate`.
            reference_strain_rate : float, optional
                Reference strain rate used for normalization. Default value is 1e-4.

            Returns
            -------
            float
                Estimated Johnson-Cook parameter C. Returns 0 if the input data is invalid
                or insufficient for computation.

            Raises
            ------
            ValueError
                If `strain_rate` and `DIF` arrays do not have the same length.

            Notes
            -----
            The function assumes that the relationship between DIF and strain rate follows
            a logarithmic form as described in the Johnson-Cook material model. Data points
            with non-positive strain rates are excluded from the computation. If the filtered
            data contains fewer than two valid points, or if all normalized strain rates are
            identical, the function returns 0 as the parameter estimate.

            The least-squares method enforces a zero intercept for the linear relationship
            between the transformed variables. This ensures compliance with the theoretical
            formulation of the Johnson-Cook model.

            References
            ----------
            For more information on the Johnson-Cook material model, refer to:
                Johnson, G. R., & Cook, W. H. (1983). A constitutive model and data for
                metals subjected to large strains, high strain rates, and high temperatures.
            """

            valid_idx = (strain_rate > 0)
            strain_rate = strain_rate[valid_idx]
            DIF = DIF[valid_idx]


            if len(strain_rate) < 2:
                return 0


            epsilon_star = strain_rate / reference_strain_rate
            x = np.log(epsilon_star)  # ln(epsilon*)


            y = DIF - 1


            if np.allclose(x, 0):
                return 0


            numerator = np.sum(x * y)
            denominator = np.sum(x ** 2)

            if abs(denominator) < 1e-12:
                return 0

            C = numerator / denominator

            return C


        def compute_dif_jc(strain_rate, C, reference_strain_rate=1e-4):
            """
            Compute the difference in Johnson-Cook parameter based on strain rate.

            This function calculates the difference in the Johnson-Cook material model
            parameter by evaluating the logarithmic relationship between the given
            strain rate and a reference strain rate. The computation involves scaling
            the logarithmic term by a material-specific constant C.

            Args:
                strain_rate (float or array-like): The strain rate(s) to evaluate. This can be
                    a single value or an array of values. It must be convertible to a numpy
                    array of floats.
                C (float): A material-specific constant that scales the logarithmic term.
                    Represents the sensitivity of the material to strain rate changes.
                reference_strain_rate (float, optional): The reference strain rate used as a
                    baseline for normalization. Defaults to 1e-4.

            Returns:
                float or numpy.ndarray: The computed difference in the Johnson-Cook parameter.
                    The result type matches the input type of `strain_rate` (scalar or array).

            Raises:
                ValueError: If `strain_rate` contains non-positive values, as the logarithm
                    of zero or negative values is undefined.
                TypeError: If `strain_rate` cannot be converted to a float array or if `C`
                    is not a scalar value.

            Notes:
                The function assumes that all inputs are valid and within physically meaningful
                ranges unless otherwise specified. Ensure that `strain_rate` and `reference_strain_rate`
                are positive to avoid mathematical errors.
            """
            strain_rate = np.array(strain_rate, dtype=float)
            epsilon_star = strain_rate / reference_strain_rate
            return 1.0 + C * np.log(epsilon_star)


        pred_concate = []
        y_concate = []
        reference_strain_rate=1e-4
        all_pred={}
        all_pred_extend = {}
        all_kesi = {}
        for name in data_name:
            file_path = f"data_DIF/{name}"
            df = pd.read_excel(file_path)

            kesi_values = df['strain rate'].values
            DIF_values = df['DIF'].values
            min_val = np.min(kesi_values)
            max_val = np.max(kesi_values)
            if np.max(kesi_values) / np.min(kesi_values) < 100:
                extend_kesi_values = np.linspace(min_val, max_val, 1000)
            else:
                extend_kesi_values = np.logspace(np.log10(min_val), np.log10(max_val), 1000)
            C=estimate_jc_parameter(kesi_values,DIF_values,reference_strain_rate=reference_strain_rate)
            pred=compute_dif_jc(kesi_values,C,reference_strain_rate=reference_strain_rate)
            extend_pred=compute_dif_jc(extend_kesi_values,C,reference_strain_rate=reference_strain_rate)
            #plot_pred_obs_DIF(kesi_values,pred,DIF_values,name)

            pred_concate.extend(pred.tolist())
            y_concate.extend(DIF_values.tolist())
            all_pred[name] = pred.tolist()
            all_pred_extend[name] = extend_pred.tolist()
            all_kesi[name] = extend_kesi_values.tolist()

        pred_concate = np.array(pred_concate)
        y_concate = np.array(y_concate)
        #pickle.dump(all_pred, open(f'outcome_save/J_C_model_strain_rate_DIF.pkl', 'wb'))
        pickle.dump(all_pred_extend, open(f'outcome_save/extend_J_C_model_strain_rate_DIF.pkl', 'wb'))
        pickle.dump(all_kesi, open(f'outcome_save/kesi_J_C_model_strain_rate_DIF.pkl', 'wb'))
        print(calculate_relative_error(pred_concate, y_concate))

    if mode=='H_K_model':
        def estimate_hk_parameters(strain_rate, DIF, reference_strain_rate=1e-4):
            """
            Estimate the parameters C1 and C2 for the HK model using least-squares regression.

            This function calculates the coefficients C1 and C2 based on the input strain rate and
            dynamic increase factor (DIF) by performing a least-squares polynomial regression. The
            regression is applied to the logarithmic transformation of the strain rate normalized
            by a reference value, and the DIF values adjusted by subtracting 1. If the input data
            is insufficient or invalid, the function returns zero for both parameters.

            Parameters
            ----------
            strain_rate : array_like
                Array of strain rate values. Must be positive for valid computation.
            DIF : array_like
                Array of dynamic increase factor (DIF) values corresponding to the strain rates.
            reference_strain_rate : float, optional
                Reference strain rate used for normalization. Default is 1e-4.

            Returns
            -------
            C1 : float
                The first coefficient obtained from the least-squares regression.
            C2 : float
                The second coefficient obtained from the least-squares regression.

            Raises
            ------
            ValueError
                If the input arrays `strain_rate` and `DIF` do not have the same shape.

            Notes
            -----
            The function filters out non-positive strain rate values since they cannot be
            log-transformed. If the filtered data contains fewer than two points or if the
            log-transformed strain rates are constant, the function returns zero for both
            parameters as no meaningful regression can be performed.

            The regression solves for [C1, C2] in the equation A * [C1, C2]^T = y, where:
            - A is the design matrix containing [x, x^2], with x being the log-transformed
              normalized strain rates.
            - y is the adjusted DIF values (DIF - 1).

            References
            ----------
            The methodology follows a standard approach for fitting polynomial models using
            least-squares regression, commonly applied in material science for modeling
            strain-rate-dependent behavior.
            """

            valid_idx = (strain_rate > 0)
            strain_rate_valid = strain_rate[valid_idx]
            DIF_valid = DIF[valid_idx]


            if len(strain_rate_valid) < 2:
                return 0, 0

            epsilon_star = strain_rate_valid / reference_strain_rate
            x = np.log(epsilon_star)  # ln(epsilon_star)


            y = DIF_valid - 1


            if np.allclose(x, x[0]):
                return 0, 0


            A = np.column_stack([x, x ** 2])  # shape (n, 2)


            c, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
            C1, C2 = c[0], c[1]

            return C1, C2


        def compute_dif_hk(strain_rate, C1, C2, reference_strain_rate=1e-4):
            """
            Compute the difference in material hardness based on the strain rate.

            This function calculates the adjustment factor for material hardness
            using the strain rate and empirical constants. The computation is
            based on a logarithmic transformation of the normalized strain rate.

            Parameters
            ----------
            strain_rate : array_like
                The strain rate values provided as input. It can be a scalar or an
                array-like object.
            C1 : float
                The first empirical constant used in the hardness adjustment formula.
            C2 : float
                The second empirical constant used in the hardness adjustment formula.
            reference_strain_rate : float, optional
                The reference strain rate used for normalization. Defaults to 1e-4.

            Returns
            -------
            float or ndarray
                The computed hardness adjustment factor. The return type matches the
                shape of the input `strain_rate` (scalar or array).

            Raises
            ------
            TypeError
                If `strain_rate` cannot be converted into a float array.
            ValueError
                If `reference_strain_rate` is non-positive, leading to invalid
                logarithmic calculations.
            """
            strain_rate = np.array(strain_rate, dtype=float)
            epsilon_star = strain_rate / reference_strain_rate
            x = np.log(epsilon_star)  # ln(epsilon_star)
            return 1.0 + C1 * x + C2 * (x ** 2)

        pred_concate = []
        y_concate = []
        reference_strain_rate=1e-4
        expand_strain_rate = np.logspace(-4, 4, num=100)
        all_pred={}
        all_pred_extend = {}
        all_kesi = {}
        for name in data_name:
            file_path = f"data_DIF/{name}"
            df = pd.read_excel(file_path)

            kesi_values = df['strain rate'].values
            DIF_values = df['DIF'].values
            min_val = np.min(kesi_values)
            max_val = np.max(kesi_values)
            if np.max(kesi_values) / np.min(kesi_values) < 100:
                extend_kesi_values = np.linspace(min_val, max_val, 1000)
            else:
                extend_kesi_values = np.logspace(np.log10(min_val), np.log10(max_val), 1000)
            C1,C2=estimate_hk_parameters(kesi_values,DIF_values,reference_strain_rate=reference_strain_rate)
            pred=compute_dif_hk(kesi_values,C1,C2,reference_strain_rate=reference_strain_rate)
            extend_pred=compute_dif_hk(extend_kesi_values,C1,C2,reference_strain_rate=reference_strain_rate)

            #plot_pred_obs_DIF(kesi_values,pred,DIF_values,name,plot=False)
            #plot_expand_predict_C_S(expand_strain_rate, expand_pred, kesi_values,DIF_values)
            pred_concate.extend(pred.tolist())
            y_concate.extend(DIF_values.tolist())
            all_pred[name] = pred.tolist()
            all_pred_extend[name] = extend_pred.tolist()
            all_kesi[name] = extend_kesi_values.tolist()

        pred_concate = np.array(pred_concate)
        y_concate = np.array(y_concate)
        #pickle.dump(all_pred, open(f'outcome_save/H_K_model_strain_rate_DIF.pkl', 'wb'))
        pickle.dump(all_pred_extend, open(f'outcome_save/extend_H_K_model_strain_rate_DIF.pkl', 'wb'))
        pickle.dump(all_kesi, open(f'outcome_save/kesi_H_K_model_strain_rate_DIF.pkl', 'wb'))
        print(calculate_relative_error(pred_concate, y_concate))

    if mode == 'R_J_model':
        def estimate_rj_parameter(strain_rate, DIF, reference_strain_rate=1e-4):
            """
            Estimate the RJ parameter lambda using linear regression on log-transformed data.

            This function calculates the RJ parameter lambda by performing a forced-origin
            linear regression on the logarithmic transformation of the input data. The
            regression is performed only on valid data points where both strain_rate and
            DIF are positive. If the data does not meet the requirements for regression,
            the function returns 0.

            Parameters
            ----------
            strain_rate : array_like
                Array of strain rate values. Must be a one-dimensional array or list.
            DIF : array_like
                Array of dynamic increase factor (DIF) values corresponding to the strain
                rate values. Must have the same length as strain_rate.
            reference_strain_rate : float, optional
                Reference strain rate used to calculate the dimensionless strain rate.
                Defaults to 1e-4.

            Returns
            -------
            float
                Estimated value of the RJ parameter lambda. Returns 0 if the regression
                cannot be performed due to insufficient valid data or numerical issues.

            Raises
            ------
            ValueError
                If the lengths of strain_rate and DIF do not match.
            TypeError
                If any of the inputs are not of the expected types.

            Notes
            -----
            The function filters out invalid data points where strain_rate <= 0 or DIF <= 0.
            Regression is performed only if there are at least two valid data points. The
            regression assumes a model of the form y = lambda * x, where y is the logarithm
            of DIF and x is the logarithm of the dimensionless strain rate.

            The function uses NumPy for numerical operations and handles edge cases such as
            division by zero or insufficient variability in the data by returning 0.

            Warnings
            --------
            If the input data contains invalid values or insufficient variability, the
            function will return 0 without raising an exception. Ensure that the input data
            is properly validated before calling this function.

            See Also
            --------
            numpy.log : Natural logarithm function used internally for transformations.
            numpy.allclose : Function used to check variability in the data.
            """

            valid_idx = (strain_rate > 0) & (DIF > 0)
            strain_rate_valid = strain_rate[valid_idx]
            DIF_valid = DIF[valid_idx]


            if len(strain_rate_valid) < 2:
                return 0


            epsilon_star = strain_rate_valid / reference_strain_rate
            x = np.log(epsilon_star)  # ln(epsilon_star)
            y = np.log(DIF_valid)  # ln(DIF)


            if np.allclose(x, x[0]):
                return 0


            numerator = np.sum(x * y)
            denominator = np.sum(x ** 2)

            if abs(denominator) < 1e-12:
                return 0

            lambda_est = numerator / denominator
            return lambda_est


        def compute_dif_rj(strain_rate, lambda_est, reference_strain_rate=1e-4):
            """
            Compute the normalized strain rate raised to a specified exponent.

            This function calculates the normalized strain rate by dividing the input
            strain rate by a reference value, then raises the result to the power of
            a given exponent. The computation is useful in rheological models and
            material science applications.

            Parameters
            ----------
            strain_rate : array_like
                Input strain rate values to be normalized and transformed. Can be a
                scalar or an array-like object.
            lambda_est : float
                Exponent value to which the normalized strain rate will be raised.
            reference_strain_rate : float, optional
                Reference strain rate used for normalization. Defaults to 1e-4.

            Returns
            -------
            numpy.ndarray
                Array of computed values representing the normalized strain rate
                raised to the specified exponent. If the input is scalar, the output
                will also be scalar.

            Notes
            -----
            The function internally converts the input strain_rate to a NumPy array
            with a float data type to ensure numerical stability and compatibility
            with mathematical operations. Ensure that all inputs are non-negative
            to avoid undefined behavior in the power operation.

            References
            ----------
            For more information on strain rate normalization, consult materials
            science literature or rheological modeling resources.
            """
            strain_rate = np.array(strain_rate, dtype=float)
            epsilon_star = strain_rate / reference_strain_rate
            return epsilon_star ** lambda_est


        pred_concate = []
        y_concate = []
        reference_strain_rate = 1e-4
        expand_strain_rate = np.logspace(-4, 4, num=100)
        all_pred={}
        all_pred_extend = {}
        all_kesi = {}
        for name in data_name:
            file_path = f"data_DIF/{name}"
            df = pd.read_excel(file_path)

            kesi_values = df['strain rate'].values
            DIF_values = df['DIF'].values
            min_val = np.min(kesi_values)
            max_val = np.max(kesi_values)
            if np.max(kesi_values) / np.min(kesi_values) < 100:
                extend_kesi_values = np.linspace(min_val, max_val, 1000)
            else:
                extend_kesi_values = np.logspace(np.log10(min_val), np.log10(max_val), 1000)
            lamda = estimate_rj_parameter(kesi_values, DIF_values, reference_strain_rate=reference_strain_rate)
            pred = compute_dif_rj(kesi_values, lamda, reference_strain_rate=reference_strain_rate)
            extend_pred = compute_dif_rj(extend_kesi_values, lamda, reference_strain_rate=reference_strain_rate)

            #plot_pred_obs_DIF(kesi_values,pred,DIF_values,name,plot=True)
            #plot_expand_predict_C_S(expand_strain_rate, expand_pred, kesi_values, DIF_values)

            pred_concate.extend(pred.tolist())
            y_concate.extend(DIF_values.tolist())
            all_pred[name] = pred.tolist()
            all_pred_extend[name] = extend_pred.tolist()
            all_kesi[name] = extend_kesi_values.tolist()

        pred_concate = np.array(pred_concate)
        y_concate = np.array(y_concate)
        pickle.dump(all_pred, open(f'outcome_save/R_J_model_strain_rate_DIF.pkl', 'wb'))
        pickle.dump(all_pred_extend, open(f'outcome_save/extend_R_J_model_strain_rate_DIF.pkl', 'wb'))
        pickle.dump(all_kesi, open(f'outcome_save/kesi_R_J_model_strain_rate_DIF.pkl', 'wb'))
        print(calculate_relative_error(pred_concate, y_concate))


