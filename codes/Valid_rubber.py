import os

import matplotlib.pyplot as plt
import numpy as np
from plot_data import *
from graph_regression import *


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
    data_name=os.listdir('test_data_rubber')
    mode='Valid'
    all_x=[]
    all_y=[]
    print(data_name)
    for name in data_name:
        file_path = f"test_data_rubber/{name}"
        df = pd.read_excel(file_path)

        strain_values = df['nominal strain'].values
        stress_values = df['nominal stress'].values
        lamda_values=1+strain_values

        all_x.append(lamda_values)
        all_y.append(stress_values)


    model = Random_graph_for_expr()
    model_sympy = Graph_to_sympy()
    GA = Genetic_algorithm(x_data=all_x,y_data=all_y)


    if mode=='Train':
        GA.evolution(save_dir='constitutive_rubber_2')
    if mode=='Valid':

        #discovered equation
        #best_graph={'nodes': ['mul', 'exp', 'x', 'add', 'log', 'x', 'exp', 'x'], 'edges': [[0, 1], [1, 2], [0, 3], [3, 4], [3, 6], [4, 5], [6, 7]], 'edge_attr': [1, -100000000.0, -100000000.0, -100000000.0, -1, 2.718281828459045, -100000000.0]}
        best_graph_2={'nodes': ['add', 'E_exp', 'mul', 'exp', 'x', 'exp', 'x'], 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6]], 'edge_attr': [1, 1, -100000000.0, -100000000.0, -1, -2]}
        best_graph_3={'nodes': ['add', 'mul', 'exp', 'add', 'x', '1', 'mul', 'exp', 'x', 'exp', 'exp', 'add', 'x', '1', 'mul', 'exp', 'add', 'x', '1', 'add', 'x', '1', '1'], 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [3, 5], [0, 6], [6, 7], [7, 8], [6, 9], [9, 10], [10, 11], [11, 12], [11, 13], [0, 14], [14, 15], [15, 16], [16, 17], [16, 18], [1, 19], [19, 20], [19, 21], [14, 22]], 'edge_attr': [-1, -1, 1, -1, 1, 1, 1, -100000000.0, -100000000.0, -1, -1, 1, -1, -1, 1, 1, 1, -1, -100000000.0, 1, -1, -1]}
        best_graph_4={'nodes': ['mul', 'exp', 'exp', 'add', 'mul', 'add', 'x', '1', 'exp', 'add', 'x', '1', 'log', 'x', 'exp', 'add', 'x', '1'], 'edges': [[0, 1], [1, 2], [0, 3], [3, 4], [3, 12], [4, 5], [4, 8], [12, 13], [5, 6], [5, 7], [8, 9], [9, 10], [9, 11], [2, 14], [14, 15], [15, 16], [15, 17]], 'edge_attr': [1, 1, -100000000.0, -100000000.0, 1, 1, 1, 2.718281828459045, 1, 1, -100000000.0, 1, -1, -100000000.0, -0.5, 1, -1]}

        best_graph=best_graph_3
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

        expand_lamda= np.linspace(1,10.6,100)

        for n in range(len(sim_expr)):
            print(data_name[n],':  ',sim_expr[n],'\n')
            lamda_values=all_x[n]
            pred = GA.get_function_prediction(sim_expr[n], all_x[n])
            expand_pred= GA.get_function_prediction(sim_expr[n], expand_lamda)

            pred_concate.extend(pred)
            y_concate.extend(all_y[n])

            all_pred[data_name[n]]=pred
            all_obs[data_name[n]]=all_y[n]
            all_pred_extend[data_name[n]] = expand_pred.tolist()

        all_x_dict = {data_name[i]: all_x[i] for i in range(len(data_name))}
        expand_x = expand_lamda  # 你已有
        MSE,R2=calculate_relative_error(pred_concate, y_concate)
        print('MSE:',MSE,'  R2:',R2)


    if mode=='Valid_discovered':
        def plot_discovery_process():
            best_fitness = pickle.load(open(f'result_save/constitutive_rubber_3/best_fitness.pkl', 'rb'))

            fitness_lists = [list(epoch.values()) for epoch in best_fitness]

            fig, ax = plt.subplots(figsize=(1.8,1.8), dpi=300)

            plt.rcParams['font.family'] = 'Arial'
            plt.rcParams['font.size'] = 7

            y = fitness_lists[0]

            x = np.arange(1, len(y) + 1)
            ax.scatter(x, y, color='#6495ED', s=5, label='Initial', zorder=10)

            for i in range(1, 149):
                y = fitness_lists[i]
                x = np.arange(1, len(y) + 1)
                ax.scatter(x, y, color=str(0.81 - 0.4 * i / 149), s=5, zorder=1)  # 灰度从0.7到1变化
            history_plot = ax.scatter([], [], color='gray', label='Process', s=5)

            #plt.ylim([None, 0.003])
            y = fitness_lists[149]
            x = np.arange(1, len(y) + 1)
            ax.scatter(x, y, color='#CD5C5C', s=5, label='Final', zorder=10)
            plt.xticks([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], fontsize=7, fontname='Arial')
            # plt.yticks([0.0005,0.001,0.0015,0.002,0.0025,0.003],[0.0005,'0.0010',0.0015,'0.0020',0.0025,'0.0030'],fontsize=7, fontname='Arial')
            plt.yticks([0,0.002,0.004,0.006,0.008],[0,0.002,0.004,0.006,0.008],fontsize=7, fontname='Arial')
            plt.legend(loc='upper left', fontsize=6, ncol=2, columnspacing=0.5,handlelength=1.1, handletextpad=0.6)


            # ax.set_xlabel('Top-k', fontsize=7)
            # ax.set_ylabel('Fitness', fontsize=7)
            # ax.tick_params(axis='both', labelsize=7)
            plt.savefig('plot_save/process_rubber.pdf', bbox_inches='tight', dpi=300)
            plt.savefig('plot_save/process_rubber.png', bbox_inches='tight', dpi=300)
            plt.tight_layout()
            plt.show()

        best_graph = pickle.load(open(f'result_save/constitutive_rubber_3/best_graphs.pkl', 'rb'))
        best_fitnesses = pickle.load(open(f'result_save/constitutive_rubber_3/best_fitness.pkl', 'rb'))
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
        plot_discovery_process()


    if mode=='Yeoh_model':
        def estimate_yeoh_parameters_lambda(lam, f):
            """
            Estimate (C10, C20, C30) for Yeoh model from (lambda, nominal_stress).

            Yeoh (uniaxial, incompressible):
                f = 2(λ^2 - 1/λ) * [ C10 + 2*C20*(I1-3) + 3*C30*(I1-3)^2 ]
                I1 = λ^2 + 2/λ

            This is linear in (C10, C20, C30):
                f = g0*C10 + g1*C20 + g2*C30
                g0 = 2(λ^2 - 1/λ)
                g1 = g0 * 2*(I1-3)
                g2 = g0 * 3*(I1-3)^2
            """
            lam = np.asarray(lam, dtype=float).ravel()
            f = np.asarray(f, dtype=float).ravel()

            if lam.shape != f.shape:
                raise ValueError("lam and f must have the same shape.")

            mask = np.isfinite(lam) & np.isfinite(f) & (lam > 1e-12)
            lam = lam[mask]
            f = f[mask]

            if lam.size < 3:
                return 0.0, 0.0, 0.0

            I1 = lam ** 2 + 2.0 / lam
            A = (I1 - 3.0)

            g0 = 2.0 * (lam ** 2 - 1.0 / lam)
            g1 = g0 * (2.0 * A)
            g2 = g0 * (3.0 * A ** 2)

            G = np.vstack([g0, g1, g2]).T  # N x 3

            theta, *_ = np.linalg.lstsq(G, f, rcond=None)
            C10, C20, C30 = theta

            return float(C10), float(C20), float(C30)


        def compute_yeoh_nominal_stress_lambda(lam, C10, C20, C30):
            """
            Compute Yeoh nominal stress f from lambda.
            """
            lam = np.asarray(lam, dtype=float)
            lam = np.clip(lam, 1e-12, None)

            I1 = lam ** 2 + 2.0 / lam
            A = (I1 - 3.0)

            bracket = (C10 + 2.0 * C20 * A + 3.0 * C30 * (A ** 2))
            f = 2.0 * (lam ** 2 - 1.0 / lam) * bracket
            return f


        # =========================
        # Main loop (CS-like style)
        # =========================
        pred_concate = []
        y_concate = []
        all_pred = {}
        all_obs = {}
        all_pred_extend = {}
        all_lam_extend = {}
        all_params = {}

        data_dir = "test_data_rubber"  # 修改成你的路径
        x_col = "nominal strain"  # 你的 λ 列名（常见：lambda/lamda/λ）
        y_col = "nominal stress"  # 你的名义应力列名（单位自洽即可）

        # data_name 例子：
        # data_name = ["C00_293.xlsx", "C00_313.xlsx", ...]
        # 或 data_name = ["C00_293", "C00_313", ...]
        for name in data_name:
            file_name = name if str(name).lower().endswith(".xlsx") else f"{name}.xlsx"
            file_path = os.path.join(data_dir, file_name)

            df = pd.read_excel(file_path)

            if x_col not in df.columns or y_col not in df.columns:
                raise KeyError(
                    f"Missing columns in {file_name}. Need '{x_col}' and '{y_col}'. "
                    f"Columns={list(df.columns)}"
                )

            lam_values = df[x_col].values+1
            f_values = df[y_col].values

            # extend λ grid
            min_val = float(np.nanmin(lam_values))
            max_val = float(np.nanmax(lam_values))
            extend_lam_values = np.linspace(min_val, max_val, 1000)

            C10, C20, C30 = estimate_yeoh_parameters_lambda(lam_values, f_values)

            if C10 == 0.0 and C20 == 0.0 and C30 == 0.0:
                print(f"fail to calculate (C10,C20,C30) in Yeoh model! The data name is {name}")
                continue

            pred = compute_yeoh_nominal_stress_lambda(lam_values, C10, C20, C30)
            extend_pred = compute_yeoh_nominal_stress_lambda(extend_lam_values, C10, C20, C30)

            pred_concate.extend(pred.tolist())
            y_concate.extend(f_values.tolist())

            key = os.path.splitext(file_name)[0]
            all_pred[key] = pred.tolist()
            all_obs[key] = f_values.tolist()
            all_pred_extend[key] = extend_pred.tolist()
            all_lam_extend[key] = extend_lam_values.tolist()
            all_params[key] = {"C10": C10, "C20": C20, "C30": C30}

            print(f"{key}: C10={C10:.6g}, C20={C20:.6g}, C30={C30:.6g}")

        pred_concate = np.asarray(pred_concate, dtype=float)
        y_concate = np.asarray(y_concate, dtype=float)

        mse,r2=calculate_relative_error(pred_concate, y_concate)
        print("Overall MSE =", mse)
        print("Overall R2  =", r2)

        os.makedirs("outcome_save", exist_ok=True)
        pickle.dump(all_pred, open("outcome_save/Yeoh_pred.pkl", "wb"))
        pickle.dump(all_obs, open("outcome_save/Yeoh_obs.pkl", "wb"))
        pickle.dump(all_pred_extend, open("outcome_save/Yeoh_pred_extend.pkl", "wb"))
        pickle.dump(all_lam_extend, open("outcome_save/Yeoh_lam_extend.pkl", "wb"))
        pickle.dump(all_params, open("outcome_save/Yeoh_params.pkl", "wb"))

    if mode=='Ogden_model':
        # =========================
        # Ogden model (incompressible, uniaxial) - helper funcs
        # =========================

        def compute_ogden_nominal_stress_lambda(lam, mu, alpha):
            """
            Ogden nominal stress f(λ) for uniaxial incompressible:
                f = Σ μ_i ( λ^(α_i - 1) - λ^(-α_i/2 - 1) )
            lam: array-like
            mu, alpha: 1D arrays of length N
            """
            lam = np.asarray(lam, dtype=float)
            lam = np.clip(lam, 1e-12, None)

            mu = np.asarray(mu, dtype=float).ravel()
            alpha = np.asarray(alpha, dtype=float).ravel()
            if mu.size != alpha.size:
                raise ValueError("mu and alpha must have the same length.")

            # B: (len(lam), N)
            # b_i(λ) = λ^(α_i-1) - λ^(-α_i/2 - 1)
            B = np.stack([lam ** (a - 1.0) - lam ** (-0.5 * a - 1.0) for a in alpha], axis=1)
            f = B @ mu
            return f


        def estimate_ogden_parameters_lambda(
                lam, f,
                n_terms=3,
                n_restarts=25,
                alpha_bounds=(0.1, 30.0),
                mu_bounds=(-1e8, 1e8),
                loss="linear",  # 可选: "soft_l1", "huber"
                f_scale=1.0,
                random_seed=0
        ):
            """
            Nonlinear least squares for Ogden parameters (mu_i, alpha_i).
            Multi-start: random alpha init + solve mu by linear LS, then refine with NLLS.

            Returns:
                mu_opt, alpha_opt
            """
            try:
                from scipy.optimize import least_squares
            except Exception as e:
                raise ImportError("SciPy is required for Ogden fitting. Please install scipy.") from e

            lam = np.asarray(lam, dtype=float).ravel()
            f = np.asarray(f, dtype=float).ravel()
            if lam.shape != f.shape:
                raise ValueError("lam and f must have the same shape.")

            mask = np.isfinite(lam) & np.isfinite(f) & (lam > 1e-12)
            lam = lam[mask]
            f = f[mask]

            if lam.size < max(2 * n_terms, 6):
                return np.zeros(n_terms), np.ones(n_terms)

            a_lo, a_hi = alpha_bounds
            m_lo, m_hi = mu_bounds

            rng = np.random.default_rng(random_seed)

            best_cost = np.inf
            best_mu = None
            best_alpha = None

            # ---- helper: build linear basis for given alpha
            def basis_matrix(lam_, alpha_):
                return np.stack([lam_ ** (a - 1.0) - lam_ ** (-0.5 * a - 1.0) for a in alpha_], axis=1)

            # ---- residual for NLLS
            def residual(p):
                mu_ = p[:n_terms]
                alpha_ = p[n_terms:]
                pred = compute_ogden_nominal_stress_lambda(lam, mu_, alpha_)
                return pred - f

            # bounds for least_squares
            lb = np.array([m_lo] * n_terms + [a_lo] * n_terms, dtype=float)
            ub = np.array([m_hi] * n_terms + [a_hi] * n_terms, dtype=float)

            # ---- deterministic first guess (mild)
            init_alpha0 = np.linspace(1.0, 6.0, n_terms)
            B0 = basis_matrix(lam, init_alpha0)
            mu0, *_ = np.linalg.lstsq(B0, f, rcond=None)
            p0 = np.concatenate([mu0, init_alpha0])

            res0 = least_squares(
                residual, p0, bounds=(lb, ub),
                loss=loss, f_scale=f_scale, max_nfev=20000
            )
            best_cost = res0.cost
            best_mu = res0.x[:n_terms].copy()
            best_alpha = res0.x[n_terms:].copy()

            # ---- multi-start: random alpha + linear solve mu + refine
            for _ in range(n_restarts):
                # log-uniform alpha init
                loga = rng.uniform(np.log(a_lo), np.log(a_hi), size=n_terms)
                alpha_init = np.exp(loga)
                alpha_init.sort()

                B = basis_matrix(lam, alpha_init)
                mu_init, *_ = np.linalg.lstsq(B, f, rcond=None)
                mu_init = np.clip(mu_init, m_lo, m_hi)

                p_init = np.concatenate([mu_init, alpha_init])

                res = least_squares(
                    residual, p_init, bounds=(lb, ub),
                    loss=loss, f_scale=f_scale, max_nfev=20000
                )

                if res.cost < best_cost and np.all(np.isfinite(res.x)):
                    best_cost = res.cost
                    best_mu = res.x[:n_terms].copy()
                    best_alpha = res.x[n_terms:].copy()

            # ---- sort terms by alpha for stable reporting
            order = np.argsort(best_alpha)
            best_alpha = best_alpha[order]
            best_mu = best_mu[order]

            return best_mu.astype(float), best_alpha.astype(float)




        pred_concate = []
        y_concate = []
        all_pred = {}
        all_obs = {}
        all_pred_extend = {}
        all_lam_extend = {}
        all_params = {}

        data_dir = "test_data_rubber"
        x_col = "nominal strain"  # 工程应变 -> λ = strain + 1
        y_col = "nominal stress"

        # Ogden settings
        N_TERMS = 3
        N_RESTARTS = 30
        ALPHA_BOUNDS = (0.1, 30.0)
        MU_BOUNDS = (-1e8, 1e8)  # 若希望 μ_i 强制为正，可改成 (0.0, 1e8)
        LOSS = "soft_l1"  # "linear"/"soft_l1"/"huber"

        for name in data_name:
            file_name = name if str(name).lower().endswith(".xlsx") else f"{name}.xlsx"
            file_path = os.path.join(data_dir, file_name)

            df = pd.read_excel(file_path)
            if x_col not in df.columns or y_col not in df.columns:
                raise KeyError(
                    f"Missing columns in {file_name}. Need '{x_col}' and '{y_col}'. "
                    f"Columns={list(df.columns)}"
                )

            lam_values = df[x_col].values.astype(float) + 1.0
            f_values = df[y_col].values.astype(float)

            min_val = float(np.nanmin(lam_values))
            max_val = float(np.nanmax(lam_values))
            extend_lam_values = np.linspace(min_val, max_val, 1000)

            mu_opt, alpha_opt = estimate_ogden_parameters_lambda(
                lam_values, f_values,
                n_terms=N_TERMS,
                n_restarts=N_RESTARTS,
                alpha_bounds=ALPHA_BOUNDS,
                mu_bounds=MU_BOUNDS,
                loss=LOSS,
                f_scale=np.nanstd(f_values) if np.nanstd(f_values) > 0 else 1.0,
                random_seed=0
            )

            pred = compute_ogden_nominal_stress_lambda(lam_values, mu_opt, alpha_opt)
            extend_pred = compute_ogden_nominal_stress_lambda(extend_lam_values, mu_opt, alpha_opt)

            pred_concate.extend(pred.tolist())
            y_concate.extend(f_values.tolist())

            key = os.path.splitext(file_name)[0]
            all_pred[key] = pred.tolist()
            all_obs[key] = f_values.tolist()
            all_pred_extend[key] = extend_pred.tolist()
            all_lam_extend[key] = extend_lam_values.tolist()

            # 保存参数（按项）
            all_params[key] = {
                "mu": mu_opt.tolist(),
                "alpha": alpha_opt.tolist()
            }

            # 打印
            msg = [f"{key}:"]
            for i, (m, a) in enumerate(zip(mu_opt, alpha_opt), start=1):
                msg.append(f"  term{i}: mu={m:.6g}, alpha={a:.6g}")
            print("\n".join(msg))

        pred_concate = np.asarray(pred_concate, dtype=float)
        y_concate = np.asarray(y_concate, dtype=float)

        mse, r2 = calculate_relative_error(pred_concate, y_concate)
        print("Overall MSE =", mse)
        print("Overall R2  =", r2)

        os.makedirs("outcome_save", exist_ok=True)
        pickle.dump(all_pred, open("outcome_save/Ogden_pred.pkl", "wb"))
        pickle.dump(all_obs, open("outcome_save/Ogden_obs.pkl", "wb"))
        pickle.dump(all_pred_extend, open("outcome_save/Ogden_pred_extend.pkl", "wb"))
        pickle.dump(all_lam_extend, open("outcome_save/Ogden_lam_extend.pkl", "wb"))
        pickle.dump(all_params, open("outcome_save/Ogden_params.pkl", "wb"))

    if mode == "ArrudaBoyce_model":
        def compute_arruda_boyce_nominal_stress_lambda(lam, C_R, N):
            lam = np.asarray(lam, dtype=float)
            lam = np.clip(lam, 1e-12, None)

            I1 = lam ** 2 + 2.0 / lam

            N = float(N)
            if N <= 0:
                raise ValueError("N must be positive.")

            poly = (
                    3.0
                    + (3.0 / (5.0 * N)) * I1
                    + (33.0 / (175.0 * N ** 2)) * (I1 ** 2)
                    + (57.0 / (875.0 * N ** 3)) * (I1 ** 3)
                    + (1557.0 / (67375.0 * N ** 4)) * (I1 ** 4)
            )

            g = (lam - 1.0 / (lam ** 2)) * poly
            return float(C_R) * g


        def estimate_arruda_boyce_parameters_lambda(
                lam, f,
                N_bounds=(0.1, 500.0),
        ):
            """
            Fit (C_R, N) from (lambda, nominal_stress).

            For fixed N, model is linear in C_R:
                f ≈ C_R * g(lam; N)
                C_R*(N) = (g^T f) / (g^T g)

            Then minimize SSE over N (1D).
            """
            try:
                from scipy.optimize import minimize_scalar
            except Exception as e:
                raise ImportError("SciPy is required for Arruda–Boyce fitting. Please install scipy.") from e

            lam = np.asarray(lam, dtype=float).ravel()
            f = np.asarray(f, dtype=float).ravel()
            if lam.shape != f.shape:
                raise ValueError("lam and f must have the same shape.")

            mask = np.isfinite(lam) & np.isfinite(f) & (lam > 1e-12)
            lam = lam[mask]
            f = f[mask]

            if lam.size < 3:
                return 0.0, 0.0

            N_lo, N_hi = map(float, N_bounds)
            if N_lo <= 0 or N_hi <= 0 or N_lo >= N_hi:
                raise ValueError("Invalid N_bounds; must satisfy 0 < N_lo < N_hi.")

            def g_of_N(N):
                I1 = lam ** 2 + 2.0 / lam
                poly = (
                        3.0
                        + (3.0 / (5.0 * N)) * I1
                        + (33.0 / (175.0 * N ** 2)) * (I1 ** 2)
                        + (57.0 / (875.0 * N ** 3)) * (I1 ** 3)
                        + (1557.0 / (67375.0 * N ** 4)) * (I1 ** 4)
                )
                return (lam - 1.0 / (lam ** 2)) * poly

            def best_CR_for_N(N):
                g = g_of_N(N)
                denom = float(np.dot(g, g))
                if denom <= 1e-30:
                    return 0.0
                return float(np.dot(g, f) / denom)

            # optimize in log(N) for numerical stability
            log_lo, log_hi = np.log(N_lo), np.log(N_hi)

            def objective(logN):
                N = float(np.exp(logN))
                C_R = best_CR_for_N(N)
                pred = C_R * g_of_N(N)
                r = pred - f
                return float(np.dot(r, r))  # SSE

            res = minimize_scalar(objective, bounds=(log_lo, log_hi), method="bounded")
            N_opt = float(np.exp(res.x))
            C_R_opt = best_CR_for_N(N_opt)

            return C_R_opt, N_opt
        pred_concate = []
        y_concate = []
        all_pred = {}
        all_obs = {}
        all_pred_extend = {}
        all_lam_extend = {}
        all_params = {}

        data_dir = "test_data_rubber"
        x_col = "nominal strain"  # λ = strain + 1
        y_col = "nominal stress"

        # N search bounds (可按材料经验收窄/放宽)
        N_BOUNDS = (0.1, 500.0)

        for name in data_name:
            file_name = name if str(name).lower().endswith(".xlsx") else f"{name}.xlsx"
            file_path = os.path.join(data_dir, file_name)

            df = pd.read_excel(file_path)
            if x_col not in df.columns or y_col not in df.columns:
                raise KeyError(
                    f"Missing columns in {file_name}. Need '{x_col}' and '{y_col}'. "
                    f"Columns={list(df.columns)}"
                )

            lam_values = df[x_col].values.astype(float) + 1.0
            f_values = df[y_col].values.astype(float)

            min_val = float(np.nanmin(lam_values))
            max_val = float(np.nanmax(lam_values))
            extend_lam_values = np.linspace(min_val, max_val, 1000)

            C_R, N = estimate_arruda_boyce_parameters_lambda(lam_values, f_values, N_bounds=N_BOUNDS)
            if C_R == 0.0 and N == 0.0:
                print(f"fail to calculate (C_R, N) in Arruda–Boyce model! The data name is {name}")
                continue

            pred = compute_arruda_boyce_nominal_stress_lambda(lam_values, C_R, N)
            extend_pred = compute_arruda_boyce_nominal_stress_lambda(extend_lam_values, C_R, N)

            pred_concate.extend(np.asarray(pred).tolist())
            y_concate.extend(np.asarray(f_values).tolist())

            key = os.path.splitext(file_name)[0]
            all_pred[key] = np.asarray(pred).tolist()
            all_obs[key] = np.asarray(f_values).tolist()
            all_pred_extend[key] = np.asarray(extend_pred).tolist()
            all_lam_extend[key] = np.asarray(extend_lam_values).tolist()
            all_params[key] = {"C_R": float(C_R), "N": float(N)}

            print(f"{key}: C_R={C_R:.6g}, N={N:.6g}")

        pred_concate = np.asarray(pred_concate, dtype=float)
        y_concate = np.asarray(y_concate, dtype=float)

        mse, r2 = calculate_relative_error(pred_concate, y_concate)
        print("Overall MSE =", mse)
        print("Overall R2  =", r2)

        os.makedirs("outcome_save", exist_ok=True)
        pickle.dump(all_pred, open("outcome_save/ArrudaBoyce_pred.pkl", "wb"))
        pickle.dump(all_obs, open("outcome_save/ArrudaBoyce_obs.pkl", "wb"))
        pickle.dump(all_pred_extend, open("outcome_save/ArrudaBoyce_pred_extend.pkl", "wb"))
        pickle.dump(all_lam_extend, open("outcome_save/ArrudaBoyce_lam_extend.pkl", "wb"))
        pickle.dump(all_params, open("outcome_save/ArrudaBoyce_params.pkl", "wb"))

    if mode=='pySR':
        # =========================
        # PySR model: (exp(x)-C1)*C2 + C3*log(x)
        # =========================


        def estimate_pysr_parameters_x(x, f):
            """
            Estimate (C1, C2, C3) for:
                f(x) = (exp(x) - C1)*C2 + C3*log(x)

            Linearization:
                f(x) = A*exp(x) + B + D*log(x)
                A = C2
                B = -C1*C2
                D = C3

            Solve least squares for (A,B,D), then recover:
                C2 = A
                C3 = D
                C1 = -B/A  (if |A|>eps)
            """
            x = np.asarray(x, dtype=float).ravel()
            f = np.asarray(f, dtype=float).ravel()

            if x.shape != f.shape:
                raise ValueError("x and f must have the same shape.")

            # log(x) requires x>0
            mask = np.isfinite(x) & np.isfinite(f) & (x > 1e-12)
            x = x[mask]
            f = f[mask]

            if x.size < 3:
                return 0.0, 0.0, 0.0

            # features
            # (若 x 很大 exp 会溢出，可按需加 clip；rubber 的 lambda 通常不大)
            phi0 = np.exp(x)  # exp(x)
            phi1 = np.ones_like(x)  # constant
            phi2 = np.log(x)  # log(x)

            G = np.vstack([phi0, phi1, phi2]).T  # N x 3

            theta, *_ = np.linalg.lstsq(G, f, rcond=None)
            A, B, D = theta

            eps = 1e-12
            if abs(A) < eps:
                # C1 不可识别（因为 A≈0 => C2≈0），给出失败标志
                return 0.0, 0.0, 0.0

            C2 = float(A)
            C3 = float(D)
            C1 = float(-B / A)
            return C1, C2, C3


        def compute_pysr_nominal_stress_x(x, C1, C2, C3):
            """
            Compute f(x) = (exp(x) - C1)*C2 + C3*log(x)
            """
            x = np.asarray(x, dtype=float)
            x = np.clip(x, 1e-12, None)  # for log safety
            return (np.exp(x) - C1) * C2 + C3 * np.log(x)


        # =========================
        # Main loop (CS-like style)
        # =========================
        pred_concate = []
        y_concate = []
        all_pred = {}
        all_obs = {}
        all_pred_extend = {}
        all_x_extend = {}
        all_params = {}

        data_dir = "test_data_rubber"
        x_col = "nominal strain"
        y_col = "nominal stress"

        for name in data_name:
            file_name = name if str(name).lower().endswith(".xlsx") else f"{name}.xlsx"
            file_path = os.path.join(data_dir, file_name)

            df = pd.read_excel(file_path)
            if x_col not in df.columns or y_col not in df.columns:
                raise KeyError(
                    f"Missing columns in {file_name}. Need '{x_col}' and '{y_col}'. "
                    f"Columns={list(df.columns)}"
                )

            # 你原来这里用的是 lambda = 1 + strain
            x_values = df[x_col].values + 1.0  # x = lambda
            f_values = df[y_col].values

            # extend grid
            min_val = float(np.nanmin(x_values))
            max_val = float(np.nanmax(x_values))
            extend_x_values = np.linspace(min_val, max_val, 1000)

            C1, C2, C3 = estimate_pysr_parameters_x(x_values, f_values)
            if C1 == 0.0 and C2 == 0.0 and C3 == 0.0:
                print(f"fail to calculate (C1,C2,C3) in PySR model! The data name is {name}")
                continue

            pred = compute_pysr_nominal_stress_x(x_values, C1, C2, C3)
            extend_pred = compute_pysr_nominal_stress_x(extend_x_values, C1, C2, C3)

            pred_concate.extend(pred.tolist())
            y_concate.extend(np.asarray(f_values, dtype=float).tolist())

            key = os.path.splitext(file_name)[0]
            all_pred[key] = pred.tolist()
            all_obs[key] = np.asarray(f_values, dtype=float).tolist()
            all_pred_extend[key] = extend_pred.tolist()
            all_x_extend[key] = extend_x_values.tolist()
            all_params[key] = {"C1": C1, "C2": C2, "C3": C3}

            print(f"{key}: C1={C1:.6g}, C2={C2:.6g}, C3={C3:.6g}")

        pred_concate = np.asarray(pred_concate, dtype=float)
        y_concate = np.asarray(y_concate, dtype=float)

        mse, r2 = calculate_relative_error(pred_concate, y_concate)
        print("Overall MSE =", mse)
        print("Overall R2  =", r2)

        os.makedirs("outcome_save", exist_ok=True)
        # pickle.dump(all_pred, open("outcome_save/PySR_pred.pkl", "wb"))
        # pickle.dump(all_obs, open("outcome_save/PySR_obs.pkl", "wb"))
        # pickle.dump(all_pred_extend, open("outcome_save/PySR_pred_extend.pkl", "wb"))
        # pickle.dump(all_x_extend, open("outcome_save/PySR_x_extend.pkl", "wb"))
        # pickle.dump(all_params, open("outcome_save/PySR_params.pkl", "wb"))

    # =========================
    # DSO model (3 params)
    # y(x) = log(C1*x + exp(C2*cos(x))) / (C3*x)
    # =========================
    if mode == "DSO":
        from scipy.optimize import least_squares
        def compute_dso_y(x, C1, C2, C3):
            """
            y(x) = log(C1*x + exp(C2*cos(x))) / (C3*x)
            with basic numerical safety.
            """
            x = np.asarray(x, dtype=float)
            x = np.clip(x, 1e-12, None)  # avoid divide-by-zero

            # prevent overflow in exp
            expo = np.clip(C2 * np.cos(x), -50.0, 50.0)
            term = C1 * x + np.exp(expo)

            # log needs positive
            term = np.clip(term, 1e-300, None)
            return np.log(term) / (C3 * x)

        def estimate_dso_parameters_x(x, y, C1_init=2.0, C2_init=1.0, C3_init=1.0):
            """
            Estimate (C1, C2, C3) by nonlinear least squares.

            Bounds (recommended for stability):
              C1 >= 0, C3 > 0, C2 in [-10, 10]
            """
            x = np.asarray(x, dtype=float).ravel()
            y = np.asarray(y, dtype=float).ravel()
            if x.shape != y.shape:
                raise ValueError("x and y must have the same shape.")

            mask = np.isfinite(x) & np.isfinite(y) & (x > 1e-12)
            x = x[mask]
            y = y[mask]
            if x.size < 5:
                return 0.0, 0.0, 0.0

            def residual(theta):
                C1, C2, C3 = theta
                pred = compute_dso_y(x, C1, C2, C3)
                return pred - y

            x0 = np.array([C1_init, C2_init, C3_init], dtype=float)
            lb = np.array([0.0, -10.0, 1e-8], dtype=float)
            ub = np.array([np.inf, 10.0, np.inf], dtype=float)

            try:
                res = least_squares(
                    residual, x0, bounds=(lb, ub),
                    loss="linear",  # 可改成 "soft_l1" 以增强鲁棒性
                    max_nfev=5000
                )
                if not res.success:
                    return 0.0, 0.0, 0.0
                C1, C2, C3 = res.x
                return float(C1), float(C2), float(C3)
            except Exception:
                return 0.0, 0.0, 0.0


        # =========================
        # Main loop (CS-like style)
        # =========================
        pred_concate = []
        y_concate = []
        all_pred = {}
        all_obs = {}
        all_pred_extend = {}
        all_x_extend = {}
        all_params = {}

        data_dir = "test_data_rubber"
        x_col = "nominal strain"
        y_col = "nominal stress"

        for name in data_name:
            file_name = name if str(name).lower().endswith(".xlsx") else f"{name}.xlsx"
            file_path = os.path.join(data_dir, file_name)

            df = pd.read_excel(file_path)
            if x_col not in df.columns or y_col not in df.columns:
                raise KeyError(
                    f"Missing columns in {file_name}. Need '{x_col}' and '{y_col}'. "
                    f"Columns={list(df.columns)}"
                )

            # 你之前的习惯：x = lambda = 1 + strain
            x_values = df[x_col].values + 1.0
            y_values = df[y_col].values

            # extend grid
            min_val = float(np.nanmin(x_values))
            max_val = float(np.nanmax(x_values))
            extend_x_values = np.linspace(min_val, max_val, 1000)

            C1, C2, C3 = estimate_dso_parameters_x(x_values, y_values)
            if C1 == 0.0 and C2 == 0.0 and C3 == 0.0:
                print(f"fail to calculate (C1,C2,C3) in DSO model! The data name is {name}")
                continue

            pred = compute_dso_y(x_values, C1, C2, C3)
            extend_pred = compute_dso_y(extend_x_values, C1, C2, C3)

            pred_concate.extend(np.asarray(pred, dtype=float).tolist())
            y_concate.extend(np.asarray(y_values, dtype=float).tolist())

            key = os.path.splitext(file_name)[0]
            all_pred[key] = np.asarray(pred, dtype=float).tolist()
            all_obs[key] = np.asarray(y_values, dtype=float).tolist()
            all_pred_extend[key] = np.asarray(extend_pred, dtype=float).tolist()
            all_x_extend[key] = np.asarray(extend_x_values, dtype=float).tolist()
            all_params[key] = {"C1": C1, "C2": C2, "C3": C3}

            print(f"{key}: C1={C1:.6g}, C2={C2:.6g}, C3={C3:.6g}")

        pred_concate = np.asarray(pred_concate, dtype=float)
        y_concate = np.asarray(y_concate, dtype=float)

        mse, r2 = calculate_relative_error(pred_concate, y_concate)
        print("Overall MSE =", mse)
        print("Overall R2  =", r2)

        os.makedirs("outcome_save", exist_ok=True)
        pickle.dump(all_pred, open("outcome_save/DSO_pred.pkl", "wb"))
        pickle.dump(all_obs, open("outcome_save/DSO_obs.pkl", "wb"))
        pickle.dump(all_pred_extend, open("outcome_save/DSO_pred_extend.pkl", "wb"))
        pickle.dump(all_x_extend, open("outcome_save/DSO_x_extend.pkl", "wb"))
        pickle.dump(all_params, open("outcome_save/DSO_params.pkl", "wb"))

