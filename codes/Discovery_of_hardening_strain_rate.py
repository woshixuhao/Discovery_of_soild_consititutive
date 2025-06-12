import math
import os

import matplotlib.pyplot as plt
import numpy as np

from graph_regression_h_s import *
from utils import *


class compute_strain_hardening():
    def model_func(self, x, C1, C2, C3):
        """
        模型: y = C3*(C1*x + 1) / (x - C2)
        """
        return C3 * (C1 * x + 1) / (x - C2)

    def generate_initial_guesses(self, len_variables=3, num_samples=10):
        """
        Generate initial guesses for parameters.
        :param bounds: List of (min, max) tuples for each parameter.
        :param num_samples: Number of initial guesses to generate.
        :return: List of initial guesses.
        """
        guesses = []
        for _ in range(num_samples):
            guess = []
            for _ in range(len_variables):
                guess.append(np.random.choice([-10, -2, -1, 1, 2, 10]))
            guesses.append(guess)
        return guesses

    def find_best_initial(self, x_data, y_data):
        guesses = self.generate_initial_guesses()

        best_guess = None
        best_score = float('inf')
        for guess in guesses:
            try:
                popt, pcov = curve_fit(
                    f=self.model_func,
                    xdata=x_data,
                    ydata=y_data,
                    p0=tuple(guess),  # 初始猜测
                    maxfev=10000  # 允许更多迭代次数
                )
                y_fit = self.model_func(x_data, *popt)

                # 计算残差
                residuals = y_data - y_fit

                # 计算均方根误差 (RMSE)
                rmse = np.sqrt(np.mean(residuals ** 2))
                if rmse < best_score:
                    best_score = rmse
                    best_guess = guess
            except RuntimeError:
                popt = (0, 0, 0)
                rmse = 1e8

        if best_guess == None:
            best_guess = guesses[0]
        return tuple(best_guess), best_score

    def estimate_parameters(self, x_data, y_data):
        """
        使用非线性最小二乘回归求解参数 C1, C2, C3

        参数:
            x_data : numpy array, 自变量 (例如等效应变)
            y_data : numpy array, 因变量 (例如真实应力)
            p0     : (C1_init, C2_init, C3_init) 初始猜测

        返回:
            popt: [C1, C2, C3] 最优参数
            pcov: 参数协方差矩阵
        """
        x_data = np.array(x_data, dtype=float)
        y_data = np.array(y_data, dtype=float)

        # 可选: 过滤无效数据 (如 x_data==C2, 但此时我们还不知道 C2 的真实值)
        # 这里仅示例: 若 y_data<0 或 x_data 有问题可以先剔除
        valid_idx = (y_data >= 0)
        x_valid = x_data[valid_idx]
        y_valid = y_data[valid_idx]
        p0, best_score = self.find_best_initial(x_valid, y_valid)
        try:
            popt, pcov = curve_fit(
                f=self.model_func,
                xdata=x_valid,
                ydata=y_valid,
                p0=p0,  # 初始猜测
                maxfev=10000  # 允许更多迭代次数
            )
            return popt, best_score
        except RuntimeError:
            # 若拟合不收敛或出错，则返回 (None, None)
            return (0, 0, 0), None

    def compute_y(self, x, C1, C2, C3):
        """
        给定回归得到的参数 (C1, C2, C3) 和新的 x，
        计算模型预测的 y。
        """
        x = np.array(x, dtype=float)
        return self.model_func(x, C1, C2, C3)

    def get_values(self, strain_values, stress_values):
        (C1_est, C2_est, C3_est), best_score = self.estimate_parameters(strain_values, stress_values)
        pred = self.compute_y(strain_values, C1_est, C2_est, C3_est)
        return pred

    def get_pred_values(self, ref_strain_values, ref_stress_values, obs_strain_values):
        (C1_est, C2_est, C3_est), best_score = self.estimate_parameters(ref_strain_values, ref_stress_values)
        pred = self.compute_y(obs_strain_values, C1_est, C2_est, C3_est)
        return pred

class compute_strain_hardening_2():
    def model_func(self,x, C1, C2, C3):
        return C1 * np.exp(1 / (C2 * x + C3))

    def generate_initial_guesses(self,len_variables=3, num_samples=100):
        """
        Generate initial guesses for parameters.
        :param bounds: List of (min, max) tuples for each parameter.
        :param num_samples: Number of initial guesses to generate.
        :return: List of initial guesses.
        """
        guesses = []
        for _ in range(num_samples):
            guess = []
            for _ in range(len_variables):
                guess.append(np.random.choice([-10, -2, -1, 1, 2, 10]))
            guesses.append(guess)
        return guesses

    def find_best_initial(self,x_data, y_data):
        guesses = self.generate_initial_guesses()

        best_guess = None
        best_score = float('inf')
        for guess in guesses:
            try:
                popt, pcov = curve_fit(
                    f=self.model_func,
                    xdata=x_data,
                    ydata=y_data,
                    p0=tuple(guess),  # 初始猜测
                    maxfev=10000  # 允许更多迭代次数
                )
                y_fit = self.model_func(x_data, *popt)

                # 计算残差
                residuals = y_data - y_fit

                # 计算均方根误差 (RMSE)
                rmse = np.sqrt(np.mean(residuals ** 2))
                if rmse < best_score:
                    best_score = rmse
                    best_guess = guess
            except RuntimeError:
                popt = (0, 0, 0)
                rmse = 1e8

        if best_guess == None:
            best_guess = guesses[0]
        return tuple(best_guess), best_score

    def estimate_parameters(self,x_data, y_data):
        """
        使用非线性最小二乘回归求解参数 C1, C2, C3

        参数:
            x_data : numpy array, 自变量 (例如等效应变)
            y_data : numpy array, 因变量 (例如真实应力)
            p0     : (C1_init, C2_init, C3_init) 初始猜测

        返回:
            popt: [C1, C2, C3] 最优参数
            pcov: 参数协方差矩阵
        """
        x_data = np.array(x_data, dtype=float)
        y_data = np.array(y_data, dtype=float)

        # 可选: 过滤无效数据 (如 x_data==C2, 但此时我们还不知道 C2 的真实值)
        # 这里仅示例: 若 y_data<0 或 x_data 有问题可以先剔除
        valid_idx = (y_data >= 0)
        x_valid = x_data[valid_idx]
        y_valid = y_data[valid_idx]
        p0, best_score = self.find_best_initial(x_valid, y_valid)
        try:
            popt, pcov = curve_fit(
                f=self.model_func,
                xdata=x_valid,
                ydata=y_valid,
                p0=p0,  # 初始猜测
                maxfev=10000  # 允许更多迭代次数
            )
            return popt, best_score
        except RuntimeError:
            # 若拟合不收敛或出错，则返回 (None, None)
            return (0, 0, 0), None

    def compute_y(self,x, C1, C2, C3):
        """
        给定回归得到的参数 (C1, C2, C3) 和新的 x，
        计算模型预测的 y。
        """
        x = np.array(x, dtype=float)
        return self.model_func(x, C1, C2, C3)

    def get_values(self,strain_values,stress_values):
        (C1_est, C2_est, C3_est), best_score = self.estimate_parameters(strain_values, stress_values)
        pred = self.compute_y(strain_values, C1_est, C2_est, C3_est)
        return pred
    def get_pred_values(self,ref_strain_values,ref_stress_values,obs_strain_values):
        (C1_est, C2_est, C3_est), best_score = self.estimate_parameters(ref_strain_values, ref_stress_values)
        pred = self.compute_y(obs_strain_values, C1_est, C2_est, C3_est)
        return pred

class compute_strain_rate_effect():
    def model_func(self,x, C1, C2):
        """
        模型: y = C3*(C1*x + 1) / (x - C2)
        """
        return (((C1 * x + 1) ** (-2)) ** C2) ** 0.5

    def generate_initial_guesses(self,len_variables=2, num_samples=20):
        """
        Generate initial guesses for parameters.
        :param bounds: List of (min, max) tuples for each parameter.
        :param num_samples: Number of initial guesses to generate.
        :return: List of initial guesses.
        """
        guesses = []
        for _ in range(num_samples):
            guess = []
            for _ in range(len_variables):
                guess.append(np.random.choice([-10, -2, -1, 1, 2, 10]))
            guesses.append(guess)
        return guesses

    def find_best_initial(self,x_data, y_data):
        guesses = self.generate_initial_guesses()

        best_guess = None
        best_score = float('inf')
        for guess in guesses:
            try:
                popt, pcov = curve_fit(
                    f=self.model_func,
                    xdata=x_data,
                    ydata=y_data,
                    p0=tuple(guess),  # 初始猜测
                    maxfev=10000  # 允许更多迭代次数
                )
                y_fit = self.model_func(x_data, *popt)

                # 计算残差
                residuals = y_data - y_fit

                # 计算均方根误差 (RMSE)
                rmse = np.sqrt(np.mean(residuals ** 2))
                if rmse < best_score:
                    best_score = rmse
                    best_guess = guess
            except RuntimeError:
                popt = (0, 0)
                rmse = 1e8

        if best_guess == None:
            best_guess = guesses[0]
        return tuple(best_guess), best_score

    def estimate_parameters(self,x_data, y_data):
        """
        使用非线性最小二乘回归求解参数 C1, C2, C3

        参数:
            x_data : numpy array, 自变量 (例如等效应变)
            y_data : numpy array, 因变量 (例如真实应力)
            p0     : (C1_init, C2_init, C3_init) 初始猜测

        返回:
            popt: [C1, C2, C3] 最优参数
            pcov: 参数协方差矩阵
        """
        x_data = np.array(x_data, dtype=float)
        y_data = np.array(y_data, dtype=float)

        # 可选: 过滤无效数据 (如 x_data==C2, 但此时我们还不知道 C2 的真实值)
        # 这里仅示例: 若 y_data<0 或 x_data 有问题可以先剔除
        valid_idx = (y_data >= 0)
        x_valid = x_data[valid_idx]
        y_valid = y_data[valid_idx]
        p0, best_score = self.find_best_initial(x_valid, y_valid)
        try:
            popt, pcov = curve_fit(
                f=self.model_func,
                xdata=x_valid,
                ydata=y_valid,
                p0=p0,  # 初始猜测
                maxfev=10000  # 允许更多迭代次数
            )
            return popt, best_score
        except RuntimeError:
            # 若拟合不收敛或出错，则返回 (None, None)
            return (0, 0), None

    def compute_y(self,x, C1, C2):
        """
        给定回归得到的参数 (C1, C2, C3) 和新的 x，
        计算模型预测的 y。
        """
        x = np.array(x, dtype=float)
        return self.model_func(x, C1, C2)

    def get_values(self, strain_rate_values, DIF_values):
        (C1_est, C2_est), best_score = self.estimate_parameters(strain_rate_values, DIF_values)
        pred = self.compute_y(strain_rate_values, C1_est, C2_est)
        return pred

    def get_pred_values(self, ref_strain_rate_values, ref_DIF_values, obs_strain_rate_values):
        (C1_est, C2_est), best_score = self.estimate_parameters(ref_strain_rate_values, ref_DIF_values)
        pred = self.compute_y(obs_strain_rate_values, C1_est, C2_est)
        return pred


class compute_J_C_strain_hardening():
    def ludwik_model(self,epsilon, sigma0, K, n):
        """
        Ludwik 模型:
        sigma = sigma0 + K * epsilon^n
        """
        return sigma0 + K * epsilon ** n

    def estimate_ludwik_parameters(self,epsilon, sigma, p0=(100.0, 500.0, 0.3)):
        """
        使用非线性最小二乘回归求解 Ludwik 模型参数:
        sigma0, K, n 使得
            sigma = sigma0 + K * epsilon^n

        参数:
            epsilon : numpy array, 等效应变 (>= 0)
            sigma   : numpy array, 真实应力 (>= 0)
            p0      : (sigma0_init, K_init, n_init), 初始猜测值

        返回:
            popt: [sigma0, K, n] 最优参数
            pcov: 参数协方差矩阵
        """
        # 转为 numpy 数组
        epsilon = np.array(epsilon, dtype=float)
        sigma = np.array(sigma, dtype=float)

        # 可选: 过滤无效数据 (例如应力<0)
        valid_idx = (sigma >= 0)
        eps_valid = epsilon[valid_idx]
        sigma_valid = sigma[valid_idx]

        # 若有效数据点不足则无法回归
        if len(eps_valid) < 3:
            return (0, 0, 0), None

        try:
            popt, pcov = curve_fit(
                f=self.ludwik_model,
                xdata=eps_valid,
                ydata=sigma_valid,
                p0=p0,  # 初始猜测
                maxfev=10000  # 允许更多迭代次数
            )
            return popt, pcov
        except RuntimeError:
            # 若拟合不收敛或出错，则返回 (0,0,0)
            return (0, 0, 0), None

    def compute_ludwik(self,epsilon, sigma0, K, n):
        """
        给定回归得到的参数 (sigma0, K, n) 和新的应变 epsilon，
        计算 Ludwik 模型预测应力。
        """
        epsilon = np.array(epsilon, dtype=float)
        return sigma0 + K * epsilon ** n
    def get_values(self, strain_values, stress_values):
        (sigma0_est, K_est, n_est), pcov = self.estimate_ludwik_parameters(strain_values, stress_values)
        pred = self.compute_ludwik(strain_values, sigma0_est, K_est, n_est)
        return pred
    def get_pred_values(self, strain_values, stress_values,obs_strain_values):
        (sigma0_est, K_est, n_est), pcov = self.estimate_ludwik_parameters(strain_values, stress_values)
        pred = self.compute_ludwik(obs_strain_values, sigma0_est, K_est, n_est)
        return pred
class compute_J_C_strain_rate_effect():
    def estimate_jc_parameter(self,strain_rate, DIF, reference_strain_rate=1e-4):
        """
        根据给定的应变速率和 DIF 数组，估计 Johnson-Cook 模型中的参数 C
        模型形式: DIF = 1 + C * ln(strain_rate / reference_strain_rate)

        参数：
            strain_rate : numpy array，应变速率 (必须 > 0)
            DIF         : numpy array，动态增益因子
            reference_strain_rate : 参考应变速率 (必须 > 0)

        返回：
            C: 拟合得到的 Johnson-Cook 参数
               若有效数据不足或出现异常，则返回 0
        """
        # 1. 过滤无效数据（例如应变速率 <= 0，无法取对数）
        valid_idx = (strain_rate > 0)
        strain_rate = strain_rate[valid_idx]
        DIF = DIF[valid_idx]

        # 如果过滤后数据太少，直接返回
        if len(strain_rate) < 2:
            return 0

        # 2. 计算无量纲应变速率并取对数
        epsilon_star = strain_rate / reference_strain_rate
        x = np.log(epsilon_star)  # ln(epsilon*)

        # 3. 计算 y = DIF - 1
        y = DIF - 1

        # 若全部 x 都为 0（极端情况：strain_rate 全部相同且等于 reference_strain_rate），无法拟合
        if np.allclose(x, 0):
            return 0

        # 4. 使用强制过原点的最小二乘法: y = C * x
        #    C = sum(x*y) / sum(x^2)
        numerator = np.sum(x * y)
        denominator = np.sum(x ** 2)

        if abs(denominator) < 1e-12:
            return 0

        C = numerator / denominator

        return C

    def compute_dif_jc(self,strain_rate, C, reference_strain_rate=1e-4):
        """
        给定应变速率数组和参数 C，计算 Johnson-Cook 模型的 DIF
        模型形式: DIF = 1 + C * ln(strain_rate / reference_strain_rate)
        """
        strain_rate = np.array(strain_rate, dtype=float)
        epsilon_star = strain_rate / reference_strain_rate
        return 1.0 + C * np.log(epsilon_star)
    def get_values(self, strain_rate_values, DIF_values,reference_strain_rate):
        C = self.estimate_jc_parameter(strain_rate_values, DIF_values,reference_strain_rate=reference_strain_rate)
        pred = self.compute_dif_jc(strain_rate_values, C,reference_strain_rate=reference_strain_rate)
        return pred
def extract_strain_rate(filename):
    """
    从类似 'Zeng_2020_0.0019.xlsx' 文件名中提取最后的数值字符串作为应变速率。
    返回字符串形式，可在图例中使用。
    """
    # 去掉扩展名
    name_no_ext = os.path.splitext(filename)[0]  # 'Zeng_2020_0.0019'
    # 按下划线分割
    parts = name_no_ext.split('_')  # ['Zeng','2020','0.0019']
    # 取最后一部分当作应变速率
    strain_rate_str = parts[-1]
    return strain_rate_str



def plot_strain_stress_curves(folder):
    """
    从 folder 中读取所有 Zeng_2020_*.xlsx 文件，
    绘制 (true plastic strain vs. true plastic stress) 曲线，
    并在同一张图中区分不同的应变速率。
    """
    # 设置图形尺寸、dpi
    plt.figure(figsize=(3, 2.5), dpi=300)

    # 字体设置
    font_settings = {'family': 'Arial', 'size': 7}
    dir_loc = 'data_hardening_strain_rate/' + folder
    for filename in os.listdir(dir_loc):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(dir_loc, filename)
            strain_rate_str = extract_strain_rate(filename)

            # 读取Excel数据 (假设文件中只有两列: 'true plastic strain' 和 'true plastic stress')
            df = pd.read_excel(filepath)

            # 取出两列
            strain = df['true plastic strain']
            stress = df['true plastic stress']

            # 绘制
            plt.plot(strain, stress, label=f"{strain_rate_str} s$^{{-1}}$")

    # 设置坐标轴标签及字体
    plt.xlabel('True plastic strain', fontdict=font_settings)
    plt.ylabel('True plastic stress', fontdict=font_settings)

    # 设置刻度字体
    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')

    # 设置图例
    plt.legend(prop={"family": "Arial", "size": 6}, ncol=2)
    plt.title(f'{folder}', fontdict={'family': 'Arial', 'fontsize': 6})
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格
    # 调整布局并显示或保存
    plt.tight_layout()
    plt.show()
    # 或者 plt.savefig("strain_stress_curves.png", dpi=300)


def match_strain_rates_and_dif(strain_rates, xlsx_file):
    """
    从 pkl_file 读取应变速率列表，
    在 xlsx_file 中查找对应应变速率，并获取相应 DIF。
    """
    # 2. 读取 xlsx 文件，并假设有两列: "Strain rate" 与 "DIF"
    df = pd.read_excel(xlsx_file)
    # 3. 将 xlsx 数据转换为字典，方便根据应变速率查找 DIF
    #    注意：如果浮点数有精度问题，可能需要做近似匹配
    sr_to_dif = dict(zip(df["strain rate"], df["DIF"]))

    # 4. 逐一匹配并获取 DIF
    results = []
    for sr in strain_rates:
        dif_val = sr_to_dif.get(sr, 1.0)  # 如果找不到精确匹配则返回 None
        results.append(dif_val)

    return results


def pack_data(folder):
    dir_loc = 'data_hardening_strain_rate/' + folder
    all_strain = []
    all_stress = []
    all_strain_rate = []
    for filename in os.listdir(dir_loc):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(dir_loc, filename)
            strain_rate_str = extract_strain_rate(filename)

            # 读取Excel数据 (假设文件中只有两列: 'true plastic strain' 和 'true plastic stress')
            df = pd.read_excel(filepath)

            # 取出两列
            strain = df['true plastic strain'].values
            stress = df['true plastic stress'].values
            all_strain.append(strain)
            all_stress.append(stress)
            strain_rate = float(strain_rate_str)
            all_strain_rate.append(strain_rate)
        DIF = match_strain_rates_and_dif(all_strain_rate, f'data_C-S/{folder}_y.xlsx')
    packed_data = {'strain': all_strain, 'stress': all_stress, 'strain_rate': all_strain_rate, 'DIF': DIF}
    pickle.dump(packed_data, open(f'saved_data_hardening_strain_rate/{folder}.pkl', 'wb'))


def process_data():
    data_dirs = os.listdir('data_hardening_strain_rate')
    for folder in data_dirs:
        print(folder)
        # plot_strain_stress_curves(folder)
        pack_data(folder)


def plot_all_strain_stress_curves():
    for folder in os.listdir('data_hardening_strain_rate'):
        plot_strain_stress_curves(folder)


def plot_simu_strain_stress_curves(strain, stress, strain_rate):
    plt.figure(figsize=(3, 2.5), dpi=300)
    # 字体设置
    font_settings = {'family': 'Arial', 'size': 7}
    for i in range(len(stress)):
        plt.plot(strain, stress[i], label=f"{strain_rate[i]} s$^{{-1}}$")

    # 设置坐标轴标签及字体
    plt.xlabel('True plastic strain', fontdict=font_settings)
    plt.ylabel('True plastic stress', fontdict=font_settings)

    # 设置刻度字体
    plt.xticks(fontsize=7, fontname='Arial')
    plt.yticks(fontsize=7, fontname='Arial')

    # 设置图例
    plt.legend(prop={"family": "Arial", "size": 6}, ncol=2)
    plt.grid(which="both", linestyle="--", linewidth=0.5)  # 细化网格
    # 调整布局并显示或保存
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    strain_hardening_eq = -C3 * (C1 * x + 1) / (C2 - x)
    strain_rate_eq =(((C4*x1 + 1)**(-2))**C5)**0.5
    strain_stress_eq=strain_hardening_eq*strain_rate_eq


    Simu_Hardening = compute_strain_hardening_2()
    Strain_rate_effect=compute_strain_rate_effect()
    all_x = []
    all_y = []
    mode = 'Valid'
    load_data=True
    data_name = os.listdir('saved_data_hardening_strain_rate')
    save_all_strain = []
    save_all_strain_rate = []
    save_all_stress = []
    save_all_DIF = []
    save_all_C1_ini = []
    save_all_C2_ini = []
    save_all_C3_ini = []
    if load_data==False:
        for name in data_name:
            print(name)
            data = pickle.load(open(f'saved_data_hardening_strain_rate/{name}', 'rb'))
            stress = data['stress']
            strain = data['strain']
            DIF = data['DIF']
            strain_rate = data['strain_rate']
            # Computed_DIF
            Computed_DIF = Strain_rate_effect.get_values(strain_rate, DIF)

            all_strain = []
            all_strain_rate = []
            all_stress=[]
            all_DIF=[]
            all_C1_ini=[]
            all_C2_ini = []
            all_C3_ini = []
            for i in range(len(strain)):

                all_strain.append(strain[i])
                all_strain_rate.append(np.array([float(strain_rate[i])] * len(strain[i])))

                #Add True DIF
                #all_DIF.append(np.array([DIF[i]] * len(strain[i])))

                #Add Computed DIF
                all_DIF.append(np.array([Computed_DIF[i]] * len(strain[i])))


                all_stress.append(stress[i])
                (C1_est_ini, C2_est_ini, C3_est_ini), best_score = Simu_Hardening.estimate_parameters(all_strain[0], all_stress[0])
                all_C1_ini.append(np.array([C1_est_ini] * len(strain[i])))
                all_C2_ini.append(np.array([C2_est_ini] * len(strain[i])))
                all_C3_ini.append(np.array([C3_est_ini] * len(strain[i])))
                # all_strain_rate.append(np.array([float(math.log(strain_rate[i]/strain_rate[0]))]*num_of_grid))
                (C1_est, C2_est, C3_est), best_score = Simu_Hardening.estimate_parameters(all_strain[i], all_stress[i])
                # print(f'strain rate :{strain_rate[i]}  params:   {C1_est}, {C2_est},{C3_est}\n'
                #       f'strain rate :{strain_rate[i]}  params:   {C1_est_ini*DIF[i]}, {C2_est},{C3_est}')

            save_all_DIF.append(all_DIF)
            save_all_strain.append(all_strain)
            save_all_stress.append(all_stress)
            save_all_strain_rate.append(all_strain_rate)
            save_all_C1_ini.append(all_C1_ini)
            save_all_C2_ini.append(all_C2_ini)
            save_all_C3_ini.append(all_C3_ini)

            all_strain = np.concatenate(all_strain)
            all_stress = np.concatenate(all_stress)
            all_strain_rate = np.concatenate(all_strain_rate)
            all_DIF = np.concatenate(all_DIF)
            all_C1_ini = np.concatenate(all_C1_ini)
            all_C2_ini = np.concatenate(all_C2_ini)
            all_C3_ini = np.concatenate(all_C3_ini)
            X = [all_strain, all_C1_ini, all_C2_ini, all_C3_ini, all_DIF]
            y = all_stress
            all_x.append(X)
            all_y.append(y)
        pickle.dump(save_all_strain, open(f'outcome_save/all_strain.pkl', 'wb'))
        pickle.dump(save_all_stress, open(f'outcome_save/all_stress.pkl', 'wb'))
        pickle.dump(save_all_strain_rate, open(f'outcome_save/all_strain_rate.pkl', 'wb'))
        pickle.dump(save_all_C1_ini, open(f'outcome_save/all_C1_ini.pkl', 'wb'))
        pickle.dump(save_all_C2_ini, open(f'outcome_save/all_C2_ini.pkl', 'wb'))
        pickle.dump(save_all_C3_ini, open(f'outcome_save/all_C3_ini.pkl', 'wb'))
        pickle.dump(save_all_DIF, open(f'outcome_save/all_DIF.pkl', 'wb'))

    if load_data == True:
        save_all_strain = pickle.load(open(f'outcome_save/all_strain.pkl', 'rb'))
        save_all_stress = pickle.load(open(f'outcome_save/all_stress.pkl', 'rb'))
        save_all_strain_rate = pickle.load(open(f'outcome_save/all_strain_rate.pkl', 'rb'))
        save_all_C1_ini = pickle.load(open(f'outcome_save/all_C1_ini.pkl', 'rb'))
        save_all_C2_ini = pickle.load(open(f'outcome_save/all_C2_ini.pkl', 'rb'))
        save_all_C3_ini = pickle.load(open(f'outcome_save/all_C3_ini.pkl', 'rb'))
        save_all_DIF = pickle.load(open(f'outcome_save/all_DIF.pkl', 'rb'))
        for i in range(len(save_all_strain)):
            all_strain = np.concatenate(save_all_strain[i])
            all_stress = np.concatenate(save_all_stress[i])
            all_strain_rate = np.concatenate(save_all_strain_rate[i])
            all_DIF = np.concatenate(save_all_DIF[i])
            all_C1_ini = np.concatenate(save_all_C1_ini[i])
            all_C2_ini = np.concatenate(save_all_C2_ini[i])
            all_C3_ini = np.concatenate(save_all_C3_ini[i])
            X = [all_strain, all_C1_ini, all_C2_ini, all_C3_ini, all_DIF]
            y = all_stress
            all_x.append(X)
            all_y.append(y)


    model = Random_graph_for_expr()
    model_sympy = Graph_to_sympy()
    GA = Genetic_algorithm(x_data=all_x, y_data=all_y)


    if mode == 'Train':
        GA.evolution('hardening_strain_rate')
    if mode == 'Valid':


        #J-C model
        #expr=(C1+C2*x**C3)*x4

        #Integrated model
        expr=x1*x4*exp(1 / (x2 * x + x3))

        #Modified model
        #expr=x1*x4 *exp(1 / ((x2-C1*(x4-1)) * x + (x3-C2*(x4-1))))



        fitness, optimal_params = GA.get_fitness_from_expr(expr, all_x, all_y)
        print(fitness, optimal_params)
        sim_expr = GA.get_regressed_function_from_expr(expr, optimal_params)
        print(sim_expr)
        pred_concate = []
        all_pred = []
        y_concate = []

        # expand_strain_rate = np.logspace(-5, 4, num=100)
        all_pred_concate = []
        all_y_concate = []

        all_strain = pickle.load(open(f'outcome_save/all_strain.pkl', 'rb'))
        all_stress = pickle.load(open(f'outcome_save/all_stress.pkl', 'rb'))
        all_strain_rate = pickle.load(open(f'outcome_save/all_strain_rate.pkl', 'rb'))
        all_C1_ini = pickle.load(open(f'outcome_save/all_C1_ini.pkl', 'rb'))
        all_C2_ini = pickle.load(open(f'outcome_save/all_C2_ini.pkl', 'rb'))
        all_C3_ini = pickle.load(open(f'outcome_save/all_C3_ini.pkl', 'rb'))
        all_DIF = pickle.load(open(f'outcome_save/all_DIF.pkl', 'rb'))

        for n in range(len(sim_expr)):
            print('data_name:', data_name[n], '  ', sim_expr[n], '\n')
            name = data_name[n]
            # print(name)

            all_pred = []
            all_y = []
            for i in range(len(all_DIF[n])):
                pred = GA.get_function_prediction(sim_expr[n], [all_strain[n][i], all_C1_ini[n][i],all_C2_ini[n][i],all_C3_ini[n][i],all_DIF[n][i]])
                # expand_pred = GA.get_function_prediction(sim_expr[n], expand_strain_rate)
                all_pred.append(pred)
                all_y.append(all_stress[n][i])

            # plot_pred_obs_hardening_strain_rate(all_strain[n], all_pred, all_y, all_strain_rate[n], data_name=data_name[n],
            #                                     plot=False,save=True)
            all_pred_concate.append(np.concatenate(all_pred))
            all_y_concate.append(np.concatenate(all_y))
            # plot_expand_predict_C_S(expand_strain_rate, expand_pred, all_x[n], all_y[n])

        print(calculate_relative_error(np.concatenate(all_pred_concate), np.concatenate(all_y_concate)))

    if mode=='J_C_model':
        JC_strain_harderning=compute_J_C_strain_hardening()
        JC_strain_rate_effect=compute_J_C_strain_rate_effect()
        all_pred_concate = []
        all_y_concate = []
        for name in data_name:
            print(name)
            data = pickle.load(open(f'saved_data_hardening_strain_rate/{name}', 'rb'))
            stress = data['stress']
            strain = data['strain']
            DIF = data['DIF']
            strain_rate = data['strain_rate']
            # Computed_DIF

            Computed_DIF = JC_strain_rate_effect.get_values(np.array(strain_rate), np.array(DIF),reference_strain_rate=strain_rate[0])

            Dyn_stress=[]
            for i in range(len(Computed_DIF)):
                Computed_stress = JC_strain_harderning.get_pred_values(strain[0], stress[0],strain[i])
                Dyn_stress.append(np.array(Computed_DIF[i])*np.array(Computed_stress))
                print(DIF[i],Computed_DIF[i])
                print(Dyn_stress[i])
                print(stress[i])
            plot_pred_obs_hardening_strain_rate_JC(strain, Dyn_stress, stress, strain_rate, data_name=name,
                                                            plot=False,save=True)
            all_pred_concate.append(np.concatenate(Dyn_stress))
            all_y_concate.append(np.concatenate(stress))
        print(calculate_relative_error(np.concatenate(all_pred_concate), np.concatenate(all_y_concate)))


