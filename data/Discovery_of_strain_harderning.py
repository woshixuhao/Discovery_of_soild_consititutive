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

def plot_expand_predict_strain_hardening(x_plot,pred_plot,x_obs,y_obs,save_name=''):
    plt.figure(figsize=(2.5, 2.5), dpi=300)
    plt.plot(x_plot, pred_plot, c='#8A83B4', linewidth=1,label='pred.')
    plt.scatter(x_obs,y_obs,marker='^',s=5,c='black',label='ref.')


    plt.xticks( fontproperties='Arial', fontsize=7)
    plt.yticks(fontproperties='Arial', fontsize=7)
    # 设置对数刻度
    # 设置标签字体和字号
    plt.xlabel("True plastic strain", fontname="Arial", fontsize=7)
    plt.ylabel("True plastic stress", fontname="Arial", fontsize=7)

    plt.legend(prop={"family": "Arial", "size": 6}, ncol=2)
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格

    # plt.savefig(f"plot_save/{save_name}.pdf", bbox_inches='tight', dpi=300)
    # plt.savefig(f"plot_save/{save_name}.png", bbox_inches='tight', dpi=300)

    plt.show()

if __name__ == '__main__':
    data_name=os.listdir('data_strain_stress')
    mode='Valid_discovered'
    all_x=[]
    all_y=[]
    print(data_name)
    for name in data_name:
        file_path = f"data_strain_stress/{name}"
        df = pd.read_excel(file_path)
        # 确保列名正确
        strain = df['true plastic strain'].values
        stress = df['true plastic stress'].values

        # all_x.append(kesi_values)
        # all_y.append(DIF_values)
        all_x.append(strain)
        all_y.append(stress)


    model = Random_graph_for_expr()
    model_sympy = Graph_to_sympy()
    GA = Genetic_algorithm(x_data=all_x,y_data=all_y)


    if mode=='Train':
        GA.evolution(save_dir='strain_hardening')
    if mode=='Valid':

        #discovered equation
        best_graph={'nodes': ['mul', 'mul', '1', '1', 'E_exp', 'exp', 'add', 'x', '1'],'edges': [[0, 1], [1, 2], [0, 3], [1, 4], [4, 5], [5, 6], [6, 7], [6, 8]],'edge_attr': [-1, -1, 1, -100000000.0, -1, -1, -100000000.0, -100000000.0]}


        expr = model_sympy.graph_to_sympy(best_graph)

        print(expr)

        fitness, optimal_params = GA.get_fitness_from_expr(expr, all_x, all_y)
        print(fitness, optimal_params)
        sim_expr = GA.get_regressed_function_from_expr(expr, optimal_params)
        print(sim_expr)
        pred_concate = []
        all_pred ={}
        all_obs={}
        y_concate = []
        expand_strain=np.linspace(0,2,100)
        for n in range(len(sim_expr)):
            print(data_name[n], ':  ', sim_expr[n])
            print('optimal_params:',optimal_params[n])
            pred = GA.get_function_prediction(sim_expr[n], all_x[n])
            expand_predict=GA.get_function_prediction(sim_expr[n],expand_strain)

            pred_concate.extend(pred)
            all_pred[data_name[n]] = pred
            all_obs[data_name[n]] = all_y[n]

            y_concate.extend(all_y[n])
            #plot_pred_obs_strain_hardening(all_x[n], pred, all_y[n], data_name=data_name[n], plot=True,save=True)
            #plot_expand_predict_strain_hardening(expand_strain, expand_predict, all_x[n], all_y[n])
        # pickle.dump(all_pred, open(f'outcome_save/Our_model_strain_hardening.pkl', 'wb'))
        # pickle.dump(all_obs, open(f'outcome_save/True_strain_hardening.pkl', 'wb'))
        print(calculate_relative_error(pred_concate, y_concate))
    if mode == 'Valid_discovered':
        best_graph = pickle.load(open(f'result_save/strain_hardening_max_3/best_graphs.pkl', 'rb'))
        best_fitnesses = pickle.load(open(f'result_save/strain_hardening_max_3/best_fitness.pkl', 'rb'))
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
    if mode=='Hollomn_model':
        def estimate_hollomon_parameters(epsilon, sigma):
            """
            Estimate Hollomon parameters K and n from strain-stress data using linear regression on log-transformed values.

            This function takes strain (epsilon) and stress (sigma) data, filters out invalid points,
            and performs a logarithmic transformation to fit the Hollomon equation. The parameters K
            and n are then derived from the regression coefficients.

            Parameters
            ----------
            epsilon : array-like of float
                Strain values as an array or list of floats. Must be positive for valid processing.
            sigma : array-like of float
                Stress values corresponding to the strain values as an array or list of floats.
                Must be positive for valid processing.

            Returns
            -------
            K : float
                The strength coefficient in the Hollomon equation, derived from the intercept of
                the regression line after exponentiation.
            n : float
                The strain-hardening exponent in the Hollomon equation, corresponding to the slope
                of the regression line.

            Raises
            ------
            ValueError
                If the input arrays are not of the same length or if they contain non-numeric values.
            TypeError
                If the inputs cannot be converted into numpy arrays of floats.

            Notes
            -----
            The Hollomon equation is defined as sigma = K * epsilon^n, where sigma is stress,
            epsilon is strain, K is the strength coefficient, and n is the strain-hardening exponent.
            Logarithmic transformation is applied to linearize the equation for regression analysis.

            The function excludes any data points where epsilon <= 0 or sigma <= 0 before performing
            the regression. If fewer than two valid data points remain, the function returns (0, 0).

            References
            ----------
            1. Hollomon, J. H. (1945). Tensile deformation. Transactions of the American Institute
               of Mining, Metallurgical and Petroleum Engineers, 162, 268-290.
            """

            epsilon = np.array(epsilon, dtype=float)
            sigma = np.array(sigma, dtype=float)


            valid_idx = (epsilon > 0) & (sigma > 0)
            epsilon_valid = epsilon[valid_idx]
            sigma_valid = sigma[valid_idx]


            if len(epsilon_valid) < 2:
                return 0, 0


            X = np.log(epsilon_valid)
            Y = np.log(sigma_valid)


            b, a = np.polyfit(X, Y, 1)

            # a = ln(K), b = n
            K = np.exp(a)
            n = b

            return K, n


        def compute_hollomon(epsilon, K, n):
            """
            Compute the Hollomon equation for given strain, strength coefficient, and
            strain-hardening exponent.

            The function calculates the stress based on the Hollomon equation, which is
            widely used in material science to describe the stress-strain behavior of
            materials that exhibit power-law hardening.

            Parameters
            ----------
            epsilon : array_like
                Strain values. Can be a scalar or an array-like object. The input will
                be converted to a NumPy array of type float.
            K : float
                Strength coefficient representing the material's resistance to plastic
                deformation.
            n : float
                Strain-hardening exponent describing how the material's strength increases
                with strain.

            Returns
            -------
            ndarray
                Computed stress values corresponding to the input strain values. The result
                is returned as a NumPy array.

            Raises
            ------
            TypeError
                If epsilon cannot be converted into a float array or if K and n are not
                numerical values.
            ValueError
                If any value in epsilon is negative, as strain values must be non-negative.
            """
            epsilon = np.array(epsilon, dtype=float)
            return K * epsilon ** n

        pred_concate = []
        y_concate = []
        all_pred={}
        for name in data_name:
            file_path = f"data_strain_stress/{name}"
            df = pd.read_excel(file_path)

            strain_values = df['true plastic strain'].values
            stress_values = df['true plastic stress'].values
            K,n = estimate_hollomon_parameters(strain_values, stress_values)

            pred = compute_hollomon(strain_values, K, n)

            #plot_pred_obs_strain_hardening(strain_values, pred, stress_values, name)

            pred_concate.extend(pred.tolist())
            y_concate.extend(stress_values.tolist())
            all_pred[name] = pred.tolist()
        pred_concate = np.array(pred_concate)
        y_concate = np.array(y_concate)
        print(calculate_relative_error(pred_concate, y_concate))
        pickle.dump(all_pred, open(f'outcome_save/Hollomn_model_strain_hardening.pkl', 'wb'))

    if mode=='Voce_model':
        def voce_model(epsilon, sigma0, A, B):
            """
            Voce model function for material stress-strain behavior.

            This function models the stress-strain relationship of a material based on
            the Voce equation. It is commonly used to describe the hardening behavior of
            materials under deformation. The model incorporates an exponential decay term
            to capture the non-linear stress response as strain increases.

            Parameters
            ----------
            epsilon : float
                Strain value at which the stress is to be calculated.
            sigma0 : float
                Initial yield stress or stress offset parameter.
            A : float
                Coefficient controlling the magnitude of the exponential decay term.
            B : float
                Exponential decay rate parameter that determines how quickly the
                stress approaches its asymptotic value.

            Returns
            -------
            float
                Calculated stress value based on the Voce model equation.

            Notes
            -----
            The Voce model assumes isotropic hardening and is suitable for monotonic
            loading conditions. It may not accurately represent complex cyclic or
            non-proportional loading scenarios.

            References
            ----------
            For further details on the Voce model, refer to materials science textbooks
            or research articles on constitutive modeling of metals.
            """
            return sigma0 * (1.0 - A * np.exp(B * epsilon))

        def estimate_voce_parameters(epsilon, sigma, p0=(300,0.5,-5),method='lm'):
            """
            Estimate the parameters for the Voce constitutive model using curve fitting.

            This function takes in strain and stress data, filters out invalid values, and
            applies a curve-fitting procedure to estimate the parameters of the Voce model.
            The estimation is performed using the Levenberg-Marquardt algorithm by default,
            but other methods can be specified. If the fitting process fails or insufficient
            valid data points are available, default zeroed parameters are returned.

            Parameters
            ----------
            epsilon : array_like
                Strain values as an iterable (list, tuple, numpy array).
            sigma : array_like
                Stress values corresponding to the strain values as an iterable
                (list, tuple, numpy array).
            p0 : tuple, optional
                Initial guess for the Voce model parameters. Defaults to (300, 0.5, -5).
            method : str, optional
                Method used for optimization in the curve_fit function. Defaults to 'lm'.

            Returns
            -------
            tuple
                A tuple containing two elements:
                - popt : array_like
                    Optimal estimated parameters for the Voce model if fitting succeeds,
                    otherwise (0, 0, 0).
                - pcov : array_like or None
                    Estimated covariance matrix of the parameters if fitting succeeds,
                    otherwise None.

            Raises
            ------
            ValueError
                If `epsilon` and `sigma` do not have the same length.
            RuntimeError
                If the curve fitting procedure encounters an internal error.

            Notes
            -----
            The function filters out stress values that are negative before performing
            the curve fitting. At least three valid data points are required for the
            fitting process to proceed; otherwise, default parameters are returned.

            The Voce model is assumed to be defined elsewhere and passed to the
            curve_fit function as `voce_model`.

            Warnings
            --------
            If the curve fitting fails due to insufficient iterations or numerical issues,
            default parameters (0, 0, 0) are returned without raising an exception.

            See Also
            --------
            scipy.optimize.curve_fit : The underlying function used for non-linear
                                       least squares fitting.
            """

            epsilon = np.array(epsilon, dtype=float)
            sigma = np.array(sigma, dtype=float)


            valid_idx = (sigma >= 0)
            epsilon_valid = epsilon[valid_idx]
            sigma_valid = sigma[valid_idx]


            if len(epsilon_valid) < 3:
                return (0, 0, 0), None

            try:
                popt, pcov = curve_fit(
                    f=voce_model,
                    xdata=epsilon_valid,
                    ydata=sigma_valid,
                    p0=p0,
                    method=method,
                    maxfev=10000
                )
                return popt, pcov
            except RuntimeError:
                return (0, 0, 0), None

        def compute_voce(epsilon, sigma0, A, B):
            """
            Compute the Voce equation for given parameters.

            The Voce equation is commonly used to model stress-strain behavior in materials.
            This function calculates the stress values based on the provided strain values
            and material-specific constants.

            Parameters
            ----------
            epsilon : array_like
                Strain values as an array or array-like object. It will be converted to a
                numpy array of type float internally.
            sigma0 : float
                The initial yield stress or scaling factor for the material.
            A : float
                Dimensionless parameter representing the deviation from linearity in the
                stress-strain curve.
            B : float
                Exponential parameter controlling the rate of saturation in the
                stress-strain curve.

            Returns
            -------
            ndarray
                Computed stress values corresponding to the input strain values, returned
                as a numpy array.

            Raises
            ------
            TypeError
                If epsilon cannot be converted into a valid numpy array, or if sigma0, A,
                or B are not of type float.
            ValueError
                If the computation involves invalid mathematical operations, such as
                overflow in exponentiation.
            """
            epsilon = np.array(epsilon, dtype=float)
            return sigma0 * (1.0 - A * np.exp(B * epsilon))

        pred_concate = []
        y_concate = []
        all_pred = {}
        for name in data_name:
            file_path = f"data_strain_stress/{name}"
            df = pd.read_excel(file_path)

            strain_values = df['true plastic strain'].values
            stress_values = df['true plastic stress'].values
            (sigma0_est, A_est, B_est), pcov = estimate_voce_parameters(strain_values, stress_values)
            if (sigma0_est, A_est, B_est)==(0,0,0):
                print(name)
                continue

            pred = compute_voce(strain_values, sigma0_est, A_est, B_est)

            #plot_pred_obs_strain_hardening(strain_values, pred, stress_values, name,plot=False)

            pred_concate.extend(pred.tolist())
            y_concate.extend(stress_values.tolist())
            all_pred[name] = pred.tolist()
        pred_concate = np.array(pred_concate)
        y_concate = np.array(y_concate)
        print(calculate_relative_error(pred_concate, y_concate))
        pickle.dump(all_pred, open(f'outcome_save/Voce_model_strain_hardening.pkl', 'wb'))

    if mode=='Ludwik_model':
        def ludwik_model(epsilon, sigma0, K, n):
            """
            sigma = sigma0 + K * epsilon^n
            """
            return sigma0 + K * epsilon ** n


        def estimate_ludwik_parameters(epsilon, sigma, p0=(100.0, 500.0, 0.3)):
            """
            Estimate Ludwik hardening model parameters by fitting experimental stress-strain data.

            This function fits the Ludwik hardening model to provided stress and strain data
            using a non-linear least squares approach. It handles invalid data points, ensures
            sufficient valid data for regression, and manages potential fitting errors gracefully.

            Parameters
            ----------
            epsilon : array-like
                Strain values as a list or array. Must be of the same length as `sigma`.
            sigma : array-like
                Stress values corresponding to the strain values in `epsilon`. Must be of the
                same length as `epsilon`.
            p0 : tuple, optional
                Initial guesses for the Ludwik model parameters (sigma_0, K, n). Defaults to
                (100.0, 500.0, 0.3).

            Returns
            -------
            tuple
                A tuple containing two elements:
                - popt : tuple or None
                    Optimized parameters (sigma_0, K, n) if fitting is successful, otherwise
                    (0, 0, 0).
                - pcov : ndarray or None
                    The estimated covariance matrix of the optimized parameters if fitting is
                    successful, otherwise None.

            Raises
            ------
            ValueError
                If `epsilon` and `sigma` do not have the same length.
            RuntimeError
                If the curve fitting process fails to converge despite sufficient valid data.

            Notes
            -----
            The function filters out invalid data points where stress is negative. At least three
            valid data points are required to perform the regression. If fewer than three valid
            points exist, the function returns default values.

            The Ludwik model used for fitting is defined elsewhere (e.g., in `ludwik_model`). Ensure
            this function is correctly implemented and accessible in the scope of this function.

            Warnings
            --------
            If the fitting process encounters issues such as insufficient data or convergence
            problems, the function will return default values without raising exceptions beyond
            critical runtime errors.
            """

            epsilon = np.array(epsilon, dtype=float)
            sigma = np.array(sigma, dtype=float)


            valid_idx = (sigma >= 0)
            eps_valid = epsilon[valid_idx]
            sigma_valid = sigma[valid_idx]


            if len(eps_valid) < 3:
                return (0, 0, 0), None

            try:
                popt, pcov = curve_fit(
                    f=ludwik_model,
                    xdata=eps_valid,
                    ydata=sigma_valid,
                    p0=p0,
                    maxfev=10000
                )
                return popt, pcov
            except RuntimeError:

                return (0, 0, 0), None


        def compute_ludwik(epsilon, sigma0, K, n):
            """
            Compute the Ludwik equation for stress-strain relationship.

            The Ludwik equation models the stress as a function of strain using a power-law
            relationship. This function evaluates the stress based on the provided parameters.

            Parameters
            ----------
            epsilon : array_like
                Strain values. Can be a single value or an array of values.
            sigma0 : float
                Yield strength or initial stress value.
            K : float
                Strength coefficient in the Ludwik equation.
            n : float
                Strain-hardening exponent in the Ludwik equation.

            Returns
            -------
            ndarray
                Computed stress values corresponding to the input strain values.

            Notes
            -----
            The Ludwik equation is defined as sigma = sigma0 + K * epsilon ** n, where:
            - sigma is the stress,
            - epsilon is the strain,
            - sigma0 is the yield strength,
            - K is the strength coefficient,
            - n is the strain-hardening exponent.

            This implementation converts `epsilon` to a NumPy array internally to ensure
            compatibility with both scalar and array inputs.
            """
            epsilon = np.array(epsilon, dtype=float)
            return sigma0 + K * epsilon ** n


        pred_concate = []
        y_concate = []
        all_pred = {}
        for name in data_name:
            file_path = f"data_strain_stress/{name}"
            df = pd.read_excel(file_path)

            strain_values = df['true plastic strain'].values
            stress_values = df['true plastic stress'].values
            (sigma0_est, K_est, n_est), pcov = estimate_ludwik_parameters(strain_values, stress_values)
            if (sigma0_est, K_est, n_est) == (0, 0, 0):
                print(name)
                continue

            pred = compute_ludwik(strain_values, sigma0_est, K_est, n_est)

            #plot_pred_obs_strain_hardening(strain_values, pred, stress_values, name, plot=True)

            pred_concate.extend(pred.tolist())
            y_concate.extend(stress_values.tolist())
            all_pred[name] = pred.tolist()
        pred_concate = np.array(pred_concate)
        y_concate = np.array(y_concate)
        print(calculate_relative_error(pred_concate, y_concate))
        pickle.dump(all_pred, open(f'outcome_save/Ludwik_model_strain_hardening.pkl', 'wb'))
    if mode=='H_S_model':
        def hs_model(epsilon, sigma0, K, eps0, n):
            """
            sigma = sigma0 + K * (epsilon + eps0)^n
            """
            return sigma0 + K * (epsilon + eps0) ** n


        def estimate_hs_parameters(epsilon, sigma, p0=(50.0, 500.0, 0.01, 0.3)):
            """
            Estimate parameters for the H-S model using stress-strain data.

            This function processes input stress and strain data, filters invalid entries,
            and performs a curve fitting procedure to estimate the parameters of the H-S model.
            If the fitting process fails or there are insufficient valid data points, default
            values are returned.

            Parameters
            ----------
            epsilon : array-like
                Strain data provided as a list, tuple, or numpy array.
            sigma : array-like
                Stress data corresponding to the strain data, provided as a list, tuple,
                or numpy array.
            p0 : tuple[float, float, float, float], optional
                Initial guess for the parameters of the H-S model. Defaults to
                (50.0, 500.0, 0.01, 0.3).

            Returns
            -------
            tuple[tuple[float, float, float, float], Optional[numpy.ndarray]]
                A tuple containing the estimated parameters and the covariance matrix.
                If the fitting process fails or there are insufficient valid data points,
                the parameters will be (0, 0, 0, 0) and the covariance matrix will be None.

            Raises
            ------
            RuntimeError
                Raised internally during the curve fitting process if the algorithm fails
                to converge. This exception is caught within the function, resulting in
                default values being returned.

            Notes
            -----
            The function filters out invalid data points where stress is negative.
            At least four valid data points are required for the curve fitting process.
            If fewer valid points exist, the function returns default values.

            The `hs_model` function is expected to be defined elsewhere and represents
            the mathematical model used for fitting.

            Warnings
            --------
            If the curve fitting process does not converge, the function silently handles
            the error by returning default values. Ensure that the input data is well-formed
            to avoid unexpected results.
            """

            epsilon = np.array(epsilon, dtype=float)
            sigma = np.array(sigma, dtype=float)


            valid_idx = (sigma >= 0)
            eps_valid = epsilon[valid_idx]
            sigma_valid = sigma[valid_idx]


            if len(eps_valid) < 4:
                return (0, 0, 0, 0), None

            try:
                popt, pcov = curve_fit(
                    f=hs_model,
                    xdata=eps_valid,
                    ydata=sigma_valid,
                    p0=p0,
                    maxfev=10000
                )
                return popt, pcov
            except RuntimeError:
                return (0, 0, 0, 0), None


        def compute_hs(epsilon, sigma0, K, eps0, n):
            """
            Compute the hardening stress (hs) based on the given parameters.

            This function calculates the hardening stress using a power-law relationship
            that depends on strain, initial stress, material constants, and exponents. The
            function ensures that the input strain values are converted to a NumPy array of
            type float for consistent computation.

            Parameters
            epsilon: array-like
                Strain values provided as input. Can be a scalar or an iterable.
            sigma0: float
                Initial yield stress of the material.
            K: float
                Material constant representing the strength coefficient.
            eps0: float
                Pre-strain offset in the material's stress-strain curve.
            n: float
                Strain-hardening exponent defining the rate of hardening.

            Returns
            numpy.ndarray
                An array of computed hardening stress values corresponding to each strain
                value in epsilon.

            Raises
            TypeError
                If epsilon cannot be converted into a valid numerical array.
            ValueError
                If any of sigma0, K, eps0, or n is not a valid numerical value.
            """
            epsilon = np.array(epsilon, dtype=float)
            return sigma0 + K * (epsilon + eps0) ** n


        pred_concate = []
        y_concate = []
        all_pred={}
        for name in data_name:
            file_path = f"data_strain_stress/{name}"
            df = pd.read_excel(file_path)

            strain_values = df['true plastic strain'].values
            stress_values = df['true plastic stress'].values
            (sigma0_est, K_est,eps0_est, n_est), pcov = estimate_hs_parameters(strain_values, stress_values)
            if  (sigma0_est, K_est,eps0_est, n_est) == (0, 0, 0,0):
                print(name)
                continue

            pred = compute_hs(strain_values, sigma0_est, K_est, eps0_est,n_est)

            plot_pred_obs_strain_hardening(strain_values, pred, stress_values, name, plot=False)

            pred_concate.extend(pred.tolist())
            y_concate.extend(stress_values.tolist())
            all_pred[name] = pred.tolist()
        pred_concate = np.array(pred_concate)
        y_concate = np.array(y_concate)
        print(calculate_relative_error(pred_concate, y_concate))
        pickle.dump(all_pred, open(f'outcome_save/H_S_model_strain_hardening.pkl', 'wb'))
    if mode=='Ludwigson_model':
        def ludwigson_model(epsilon, K1, n1, K2, n2):
            """
            sigma = K1 * epsilon^n1 + exp(K2 + n2 * epsilon)
            """
            return K1 * epsilon ** n1 + np.exp(K2 + n2 * epsilon)


        def estimate_ludwigson_parameters(epsilon, sigma, p0=(500.0, 0.3, 2.0, 0.1)):
            """
            Estimate the parameters of the Ludwigson model using nonlinear least squares fitting.

            This function fits experimental stress-strain data to the Ludwigson model by performing
            a curve-fitting operation. It handles invalid data points (e.g., negative stress or
            non-positive strain) and ensures sufficient valid data is available for regression.
            If the fitting process fails or there are insufficient valid data points, default
            values are returned.

            Parameters
            ----------
            epsilon : array-like of float
                Strain values as input data for the model. Must be a one-dimensional sequence
                of numerical values.
            sigma : array-like of float
                Stress values corresponding to the strain data. Must have the same length as
                `epsilon`.
            p0 : tuple of float, optional
                Initial guess for the parameters of the Ludwigson model. Defaults to
                (500.0, 0.3, 2.0, 0.1).

            Returns
            -------
            tuple
                A tuple containing two elements:
                    - popt : tuple of float
                        The optimized parameters of the Ludwigson model if fitting is successful.
                        If fitting fails or there are insufficient data points, this will be
                        (0, 0, 0, 0).
                    - pcov : ndarray or None
                        The estimated covariance matrix of the optimized parameters. Returns None
                        if fitting fails or there are insufficient data points.

            Raises
            ------
            ValueError
                If the lengths of `epsilon` and `sigma` do not match.
            RuntimeError
                If the curve-fitting algorithm encounters an internal error during optimization.

            Notes
            -----
            The function filters out invalid data points where stress is negative or strain is
            non-positive. At least four valid data points are required to perform regression.
            If fewer valid points are available, the function returns default values without
            attempting to fit.

            The Ludwigson model is assumed to be implemented in a separate function named
            `ludwigson_model`. Ensure this function is defined and compatible with the input data.

            Warnings
            --------
            A warning may be issued internally by `curve_fit` if the fitting process does not
            converge within the allowed number of iterations (`maxfev=10000`). In such cases,
            default values are returned.

            See Also
            --------
            scipy.optimize.curve_fit : Function used internally for nonlinear least squares fitting.
            numpy.array : Used internally for handling and filtering input data arrays.
            """

            epsilon = np.array(epsilon, dtype=float)
            sigma = np.array(sigma, dtype=float)


            valid_idx = (sigma >= 0) & (epsilon > 0)
            eps_valid = epsilon[valid_idx]
            sigma_valid = sigma[valid_idx]


            if len(eps_valid) < 4:
                return (0, 0, 0, 0), None

            try:
                popt, pcov = curve_fit(
                    f=ludwigson_model,
                    xdata=eps_valid,
                    ydata=sigma_valid,
                    p0=p0,
                    maxfev=10000
                )
                return popt, pcov
            except RuntimeError:
                return (0, 0, 0, 0), None


        def compute_ludwigson(epsilon, K1, n1, K2, n2):
            """
            Compute the Ludwigson equation given the parameters.

            This function evaluates the Ludwigson equation, which is commonly used in
            material science to model stress-strain relationships. It combines a power-law
            term and an exponential term to capture complex material behavior under loading.

            Parameters
            ----------
            epsilon : array_like
                Strain values at which the Ludwigson equation is evaluated. The input will
                be converted to a NumPy array of type float.
            K1 : float
                Coefficient for the power-law term. Controls the magnitude of the
                contribution from the power-law component.
            n1 : float
                Exponent for the power-law term. Determines the nonlinearity of the
                power-law component.
            K2 : float
                Coefficient for the exponential term. Scales the exponential component's
                contribution.
            n2 : float
                Exponent multiplier for the exponential term. Modulates the rate of
                growth of the exponential component.

            Returns
            -------
            ndarray
                Computed values of the Ludwigson equation at the given strain values.

            Notes
            -----
            The Ludwigson equation is defined as:
                f(epsilon) = K1 * epsilon ** n1 + exp(K2 + n2 * epsilon)

            Ensure that the input `epsilon` is provided in a format compatible with NumPy
            array conversion. The computation relies on NumPy for efficient numerical
            evaluation.

            References
            ----------
            For more information on the Ludwigson equation, refer to standard material
            science textbooks or research articles discussing stress-strain modeling.
            """
            epsilon = np.array(epsilon, dtype=float)
            return K1 * epsilon ** n1 + np.exp(K2 + n2 * epsilon)


        pred_concate = []
        y_concate = []
        all_pred={}
        for name in data_name:
            file_path = f"data_strain_stress/{name}"
            df = pd.read_excel(file_path)

            strain_values = df['true plastic strain'].values
            stress_values = df['true plastic stress'].values
            (K1_est, n1_est, K2_est, n2_est), pcov = estimate_ludwigson_parameters(strain_values, stress_values)
            if (K1_est, n1_est, K2_est, n2_est)== (0, 0, 0, 0):
                print(name)
                continue

            pred = compute_ludwigson(strain_values,K1_est, n1_est, K2_est, n2_est)

            plot_pred_obs_strain_hardening(strain_values, pred, stress_values, name, plot=False)

            pred_concate.extend(pred.tolist())
            y_concate.extend(stress_values.tolist())
            all_pred[name] = pred.tolist()
        pred_concate = np.array(pred_concate)
        y_concate = np.array(y_concate)
        print(calculate_relative_error(pred_concate, y_concate))
        pickle.dump(all_pred, open(f'outcome_save/Ludwigson_model_strain_hardening.pkl', 'wb'))
    if mode=='Bargar_model':
        def bargar_model(epsilon, sigma0, c, d, e):
            """
            sigma = sigma0 + c * epsilon^0.4 + d * epsilon^0.8 + e * epsilon^1.2
            """
            return sigma0 + c * epsilon ** 0.4 + d * epsilon ** 0.8 + e * epsilon ** 1.2


        def estimate_bargar_parameters(epsilon, sigma):
            """
            Estimate parameters for the Bargar model using linear regression.

            This function performs parameter estimation for a model where sigma is
            expressed as a function of epsilon. It uses a design matrix to fit the data
            to a polynomial-like form and applies least-squares regression to compute
            the coefficients.

            Parameters
            ----------
            epsilon : array-like of float
                Input values representing epsilon in the model. Must be non-negative.
            sigma : array-like of float
                Input values representing sigma in the model. Must be non-negative.

            Returns
            -------
            tuple
                A tuple containing the following elements:
                - coeffs : tuple of float
                    Coefficients (sigma0, c, d, e) derived from the regression.
                - residuals : ndarray or None
                    Residuals of the least-squares fit. None if the fit fails.
                - rank : int or None
                    Rank of the design matrix. None if the fit fails.
                - s : ndarray or None
                    Singular values of the design matrix. None if the fit fails.

            Raises
            ------
            ValueError
                If the input arrays `epsilon` and `sigma` have incompatible shapes or
                contain invalid data types.

            Notes
            -----
            The function filters out invalid data points where epsilon or sigma is
            negative. If fewer than four valid data points remain, the regression
            cannot be performed, and default values are returned.

            The design matrix includes columns corresponding to constant, epsilon^0.4,
            epsilon^0.8, and epsilon^1.2 terms, allowing for a flexible fit to the data.
            """
            epsilon = np.array(epsilon, dtype=float)
            sigma = np.array(sigma, dtype=float)


            valid_idx = (epsilon >= 0) & (sigma >= 0)
            eps_valid = epsilon[valid_idx]
            sigma_valid = sigma[valid_idx]


            if len(eps_valid) < 4:
                return (0, 0, 0, 0), None, None, None


            A = np.column_stack([
                np.ones_like(eps_valid),
                eps_valid ** 0.4,
                eps_valid ** 0.8,
                eps_valid ** 1.2
            ])


            coeffs, residuals, rank, s = np.linalg.lstsq(A, sigma_valid, rcond=None)
            sigma0, c, d, e = coeffs
            return (sigma0, c, d, e), residuals, rank, s


        def compute_bargar(epsilon, sigma0, c, d, e):
            """
            Compute the Bargar model using the provided parameters.

            This function takes in several parameters related to the Bargar model and computes
            the result by converting epsilon into a NumPy array and passing all arguments to the
            bargar_model function. Ensure that all inputs are properly defined and compatible
            with the underlying model implementation.

            Parameters
            ----------
            epsilon : array-like
                Input parameter representing epsilon values. It will be converted to a NumPy
                array of type float.
            sigma0 : float
                Initial sigma value required for the Bargar model computation.
            c : float
                Coefficient or constant used in the Bargar model.
            d : float
                Additional coefficient or constant for the model.
            e : float
                Extra parameter influencing the Bargar model calculation.

            Returns
            -------
            float or ndarray
                The computed result from the Bargar model based on the input parameters.

            Notes
            -----
            Ensure that the bargar_model function is correctly implemented and accessible in the
            current environment. The function relies on NumPy for handling epsilon as an array.
            """
            epsilon = np.array(epsilon, dtype=float)
            return bargar_model(epsilon, sigma0, c, d, e)


        pred_concate = []
        y_concate = []
        all_pred={}
        for name in data_name:
            file_path = f"data_strain_stress/{name}"
            df = pd.read_excel(file_path)

            strain_values = df['true plastic strain'].values
            stress_values = df['true plastic stress'].values
            (sigma0, c, d, e), _,_,_ = estimate_bargar_parameters(strain_values, stress_values)

            pred = compute_bargar(strain_values,sigma0, c, d, e)

            plot_pred_obs_strain_hardening(strain_values, pred, stress_values, name, plot=False)

            pred_concate.extend(pred.tolist())
            y_concate.extend(stress_values.tolist())
            all_pred[name] = pred.tolist()
        pred_concate = np.array(pred_concate)
        y_concate = np.array(y_concate)
        print(calculate_relative_error(pred_concate, y_concate))
        pickle.dump(all_pred, open(f'outcome_save/Bargar_model_strain_hardening.pkl', 'wb'))