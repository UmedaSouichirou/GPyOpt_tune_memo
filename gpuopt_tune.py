
from .base import AcquisitionBase
from ..util.general import get_quantiles
import numpy as np
import cupy as cp  # CuPy を追加

class AcquisitionEI_GPU(AcquisitionBase):
    """
    Expected Improvement acquisition function (GPU Version)
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        self.optimizer = optimizer
        super(AcquisitionEI_GPU, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return AcquisitionEI_GPU(model, space, optimizer, cost_withGradients, jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes the Expected Improvement per unit of cost (GPU 版)
        """
        x = cp.asarray(x)  # NumPy から CuPy に変換
        m, s = self.model.predict(cp.asnumpy(x))  # GPy は CuPy 非対応なので NumPy に変換
        m, s = cp.asarray(m), cp.asarray(s)  # CuPy に戻す
        fmin = self.model.get_fmin()

        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        phi, Phi, u = cp.asarray(phi), cp.asarray(Phi), cp.asarray(u)  # CuPy に変換

        f_acqu = s * (u * Phi + phi)
        return f_acqu
    
    def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (GPU 版)
        """
        x = cp.asarray(x)  # NumPy から CuPy に変換
        fmin = self.model.get_fmin()

        # GPy は CuPy 非対応のため、NumPy に戻して計算
        m, s, dmdx, dsdx = self.model.predict_withGradients(cp.asnumpy(x))
        m, s, dmdx, dsdx = cp.asarray(m), cp.asarray(s), cp.asarray(dmdx), cp.asarray(dsdx)  # CuPy に変換

        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        phi, Phi, u = cp.asarray(phi), cp.asarray(Phi), cp.asarray(u)  # CuPy に変換

        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx

        return f_acqu, df_acqu




# GPyOpt.methods.BayesianOptimization.py内
if self.acquisition_type == 'EI':
    self.acquisition = AcquisitionEI(self.model, self.space, self.acquisition_optimizer)
elif self.acquisition_type == 'EI_GPU':
    from GPyOpt.acquisitions.Acquisition import AcquisitionEI_GPU
    self.acquisition = AcquisitionEI_GPU(self.model, self.space, self.acquisition_optimizer)


# 例
optimizer = BayesianOptimization(
    f=objective_function,
    domain=bounds,
    acquisition_type='EI_GPU'  # GPU 版 Expected Improvement を使用
)
optimizer.run_optimization(max_iter=10)

# ベンチマーク
import time
import numpy as np
import cupy as cp
from GPyOpt.methods import BayesianOptimization

# 目的関数
def objective_function(x):
    return (x - 2) ** 2

# 探索空間の定義
bounds = [{'name': 'x', 'type': 'continuous', 'domain': (-5, 5)}]

# ベイズ最適化の設定 (CPU 版)
optimizer_cpu = BayesianOptimization(
    f=objective_function,
    domain=bounds,
    acquisition_type='EI'  # CPU 版
)

# ベイズ最適化の設定 (GPU 版)
optimizer_gpu = BayesianOptimization(
    f=objective_function,
    domain=bounds,
    acquisition_type='EI_GPU'  # GPU 版
)

# 1D の評価用グリッドを作成 (10,000 点)
N = 10000
grid_x_np = np.linspace(-5, 5, N).reshape(-1, 1)
grid_x_cp = cp.linspace(-5, 5, N).reshape(-1, 1)

# CPU 版の Acquisition 関数評価
start_cpu = time.time()
acq_scores_cpu = optimizer_cpu.acquisition.evaluate(grid_x_np)
end_cpu = time.time()

# GPU 版の Acquisition 関数評価
start_gpu = time.time()
acq_scores_gpu = optimizer_gpu.acquisition.evaluate(grid_x_cp.get())  # CuPy → NumPy に変換
end_gpu = time.time()

# 最もスコアの高い点を取得 (CPU)
best_index_cpu = np.argmax(acq_scores_cpu)
best_point_cpu = grid_x_np[best_index_cpu]
best_score_cpu = acq_scores_cpu[best_index_cpu]

# 最もスコアの高い点を取得 (GPU)
best_index_gpu = np.argmax(acq_scores_gpu)
best_point_gpu = grid_x_cp[best_index_gpu].get()  # CuPy → NumPy に変換
best_score_gpu = acq_scores_gpu[best_index_gpu]

# 結果の表示
print(f"CPU 版の実行時間: {end_cpu - start_cpu:.4f} 秒")
print(f"GPU 版の実行時間: {end_gpu - start_gpu:.4f} 秒")
print(f"CPU 版の最適な点: {best_point_cpu}, スコア: {best_score_cpu}")
print(f"GPU 版の最適な点: {best_point_gpu}, スコア: {best_score_gpu}")
