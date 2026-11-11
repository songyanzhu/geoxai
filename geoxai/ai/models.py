import numpy as np
import pandas as pd
import copy
import itertools
from scipy import ndimage
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedGroupKFold

class SmartForest4D:
    """
    SmartForest4D is an ensemble learning model designed to handle complex 4D input data 
    (samples, time, spatial, features). It integrates ideas from gradient-boosted decision trees 
    (XGBoost) with LSTM-style forget gates and spatial max pooling.

    The model builds layers of regressors, each layer taking the previous output as part of its 
    input (deep forest style). A forget gate mechanism is applied along the time dimension to 
    emphasize recent temporal information. Spatial max pooling is used to reduce dimensionality 
    across spatial units before flattening and feeding into the regressors.

    Parameters:
    -----------
    n_estimators_per_layer : int
        Number of XGBoost regressors per layer.

    max_layers : int
        Maximum number of layers in the deep forest.

    early_stopping_rounds : int
        Number of rounds without improvement on the validation set before early stopping.

    param_grid : dict
        Dictionary of hyperparameter lists to search over for XGBoost.

    use_gpu : bool
        Whether to use GPU for training XGBoost models.

    gpu_id : int
        GPU device ID to use if use_gpu is True.

    kernel: np.ndarray
        Convolutional kernel for spatial processing.
        # ===============================
        # 0. Do nothing
        # ===============================

        identity_kernel = np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

        # ===============================
        # 1. Sobel Edge Detection Kernels
        # ===============================

        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])

        sobel_y = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ])

        # ===============================
        # 2. Gaussian Blur Kernel (3x3)
        # ===============================
        gaussian_kernel = (1/16) * np.array([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ])

        # ===============================
        # 3. Morphological Structuring Element (3x3 cross)
        # Used in binary dilation/erosion
        # ===============================
        morph_kernel = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ])

        # ===============================
        # 4. Sharpening Kernel
        # Enhances edges and contrast
        # ===============================
        sharpen_kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])

        # ===============================
        # 5. Embossing Kernel
        # Creates a 3D-like shadowed effect
        # ===============================
        emboss_kernel = np.array([
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ])

    spatial_h : int
        The height of the 2D grid for the flattened spatial dimension.

    spatial_w : int
        The width of the 2D grid for the flattened spatial dimension.
    
    forget_factor : float
        Exponential decay rate applied along the time axis. Higher values mean stronger forgetting.

    verbose : int
        Verbosity level for training output.
    eval_metric : str
        Statistical metric for evaluating model performance.

    Attributes:
    -----------
    layers : list
        List of trained layers, each containing a list of regressors.

    best_model : list
        The set of layers corresponding to the best validation RMSE seen during training.

    best_score : float
        The best metric e.g., lowest RMSE achieved on the validation set.

    Methods:
    --------
    fit(X, y, X_val=None, y_val=None):
        Train the SmartForest4D model on the given 4D input data.

    predict(X):
        Predict targets for new 4D input data using the trained model.

    get_best_model():
        Return the best set of layers and corresponding RMSE.

    Notes:
    ------
    - The product of spatial_h and spatial_w must equal spatial_size (spatial_h * spatial_w = spatial_size).

    Example:
    --------
    >>> model = SmartForest4D(n_estimators_per_layer=5, max_layers=10, early_stopping_rounds=3, forget_factor=0.3, verbose=1)
    >>> model.fit(X_train, y_train, X_val, y_val)
    >>> y_pred = model.predict(X_val)
    >>> best_model, best_rmse = model.get_best_model()
    """
    def __init__(self, n_estimators_per_layer=5, max_layers=10, early_stopping_rounds=3, param_grid=None,
                 use_gpu=False, gpu_id=0, kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), spatial_h=None, spatial_w=None,
                 forget_factor=0.0, verbose=1, eval_metric='rmse'):
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.param_grid = param_grid or {
            "objective": ["reg:squarederror"],
            "random_state": [42],
            'n_estimators': [100],
            'max_depth': [6],
            'min_child_weight': [4],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [0],
            'reg_alpha': [0],
            'reg_lambda': [1],
            'learning_rate': [0.05],
        }
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.kernel = kernel
        self.spatial_h = spatial_h
        self.spatial_w = spatial_w
        self.forget_factor = forget_factor
        self.layers = []
        self.best_model = None
        self.verbose = verbose
        self.eval_metric = eval_metric.lower()
        self.best_score = float("inf") if self.eval_metric != 'r2' else float("-inf")
        if (self.spatial_h is None) or (self.spatial_w is None): 
            raise ValueError("Please specify spatial_h and spatial_w")

    def _evaluate(self, y_true, y_pred):
        if self.eval_metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.eval_metric == 'nrmse':
            return np.sqrt(mean_squared_error(y_true, y_pred)) / np.mean(np.abs(y_true))
        elif self.eval_metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif self.eval_metric == 'mape':
            return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100
        elif self.eval_metric == 'r2':
            return r2_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown evaluation metric: {self.eval_metric}")

    def _get_param_combinations(self):
        keys, values = zip(*self.param_grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    def _prepare_input(self, X, y=None, apply_forget=False, layer_index=0):
        if X.ndim == 2:
            X = X[:, np.newaxis, np.newaxis, :]
        elif X.ndim == 3:
            X = X[:, :, np.newaxis, :]
        elif X.ndim == 4:
            pass
        else:
            raise ValueError("Input must be 2D, 3D, or 4D.")

        n_samples, n_time, n_spatial, n_features = X.shape

        if apply_forget and self.forget_factor > 0:
            decay = np.exp(-self.forget_factor * np.arange(n_time))[::-1]
            decay = decay / decay.sum()
            decay = decay.reshape(1, n_time, 1, 1)
            X = X * decay

        if n_spatial != 1:
            if self.spatial_h * self.spatial_w != n_spatial: raise ValueError("spatial_h * spatial_w != n_spatial")
            X_out = np.zeros_like(X)
            for sample in range(X.shape[0]):
                for t in range(X.shape[1]):
                    for f in range(X.shape[3]):
                        spatial_2d = X[sample, t, :, f].reshape(self.spatial_h, self.spatial_w)
                        filtered = ndimage.convolve(spatial_2d, self.kernel, mode='constant', cval=0.0)
                        X_out[sample, t, :, f] = filtered.reshape(n_spatial)
            X = X_out; del(X_out)
        X_pooled = X.max(axis=2)
        X_flattened = X_pooled.reshape(n_samples, -1)
        return X_flattened

    def _fit_layer(self, X, y, X_val=None, y_val=None, layer_index=0):
        layer = []
        layer_outputs = []
        param_combos = self._get_param_combinations()

        for i in range(self.n_estimators_per_layer):
            best_metric = float('-inf') if self.eval_metric == 'r2' else float('inf')
            best_model = None

            for params in param_combos:
                if self.use_gpu:
                    params['tree_method'] = 'hist'
                    params['device'] = 'cuda'

                params = params.copy()
                params['random_state'] = i

                model = XGBRegressor(**params)
                model.fit(X, y)

                if X_val is not None:
                    preds_val = model.predict(X_val)
                    metric = self._evaluate(y_val, preds_val)
                    better = (metric > best_metric) if self.eval_metric == 'r2' else (metric < best_metric)
                    if better:
                        best_metric = metric
                        best_model = model
                else:
                    best_model = model

            preds = best_model.predict(X).reshape(-1, 1)
            layer.append(best_model)
            layer_outputs.append(preds)

        output = np.hstack(layer_outputs)
        return layer, output

    def fit(self, X, y, X_val=None, y_val=None):
        y = y.ravel()
        X_current = self._prepare_input(X, apply_forget=True)
        X_val_current = self._prepare_input(X_val, apply_forget=True) if X_val is not None else None

        no_improve_rounds = 0

        for layer_index in range(self.max_layers):
            if self.verbose:
                print(f"Training Layer {layer_index + 1}")

            layer, output = self._fit_layer(X_current, y, X_val_current, y_val, layer_index)
            self.layers.append(layer)
            X_current = np.hstack([X_current, output])

            if X_val is not None:
                val_outputs = []
                for reg in layer:
                    n_features = reg.n_features_in_
                    preds = reg.predict(X_val_current[:, :n_features]).reshape(-1, 1)
                    val_outputs.append(preds)
                val_output = np.hstack(val_outputs)
                X_val_current = np.hstack([X_val_current, val_output])

                y_pred = self.predict(X_val)
                score = self._evaluate(y_val, y_pred)
                if self.verbose:
                    print(f"Validation {self.eval_metric.upper()}: {score:.4f}")

                improvement = (score < self.best_score) if self.eval_metric != 'r2' else (score > self.best_score)
                if improvement:
                    self.best_score = score
                    self.best_model = copy.deepcopy(self.layers)
                    no_improve_rounds = 0
                    if self.verbose:
                        print(f"\u2705 New best {self.eval_metric.upper()}: {self.best_score:.4f}")
                else:
                    no_improve_rounds += 1
                    if no_improve_rounds >= self.early_stopping_rounds:
                        if self.verbose:
                            print("Early stopping triggered.")
                        break

        # 训练结束：predict() 改用验证集上最优的层，而不是最后训练到的层
        if self.best_model is not None:
            self.layers = self.best_model

    def predict(self, X):
        X_current = self._prepare_input(X, apply_forget=True)

        for layer in self.layers:
            layer_outputs = []
            for reg in layer:
                n_features = reg.n_features_in_
                preds = reg.predict(X_current[:, :n_features]).reshape(-1, 1)
                layer_outputs.append(preds)
            output = np.hstack(layer_outputs)
            X_current = np.hstack([X_current, output])

        final_outputs = []
        for reg in self.layers[-1]:
            n_features = reg.n_features_in_
            final_outputs.append(reg.predict(X_current[:, :n_features]).reshape(-1, 1))
        return np.mean(np.hstack(final_outputs), axis=1)

    def get_best_model(self):
        return self.best_model, self.best_score

"""
# ============================== Test Example ==============================
import numpy as np
import copy
import itertools
from scipy import ndimage
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Generate synthetic 4D data: (samples, time, spatial, features)
# time order is like [t (today), t - 1 (yesterday), t -2, ...]
n_samples = 200
n_time = 5
n_spatial = 4
n_features = 5

np.random.seed(42)
X = np.random.rand(n_samples, n_time, n_spatial, n_features)
y = X[:, :3, :2, :4].mean(axis=(1, 2, 3)) + 0.1 * np.random.randn(n_samples)
y = y.ravel()

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = SmartForest4D(
    n_estimators_per_layer=5,
    max_layers=20,
    early_stopping_rounds=5,
    spatial_h = 2, 
    spatial_w = 2,
    forget_factor=0.1,
    verbose=1
)
model.fit(X_train, y_train, X_val, y_val)

# Predict
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print("\n✅ Final RMSE on validation set:", rmse)


# Output best model and RMSE
best_model, best_rmse = model.get_best_model()
print("\nBest validation RMSE:", best_rmse)
"""

# ============================================================================================================================================================

# =====================================================================
# SmartForestTransfer（逐层堆叠的 XGBoost 深度森林）
# =====================================================================


class SmartForestTransfer:
    """
    SmartForestTransfer（SFT）：面向空间外推的逐层堆叠 XGBoost 深度森林（表格输入）。

    SFT 的目标是空间可迁移性（spatial transferability）：让模型从已有通量站学到的规律，
    能用到训练时从未见过的地点，比如新的通量站，或者全球没有观测的像元。为此，SFT 在训练
    过程中始终按站点（推荐用空间簇）分折，用"预测没见过的站点"的能力来决定树的数量、层数
    和超参数，而不是在已知站点内的拟合精度。

    与 SmartForest4D（SF）是两个独立的分支，互不影响：
    - SF  擅长已知站点内的插补（随机划分下精度高），但对新站点的泛化较弱；
    - SFT 擅长对新站点和全球像元的预测，适合上推；在随机划分下的分数会低于 SF，这是设计使然。

    每一层
    ------
    - n_estimators_per_layer 个位置 × n_folds 折；每个折模型在自己的留出折上早停（加权）。
    - 下一层新增特征 = 本层 OOF 预测（无信息泄漏）及其均值、标准差（add_output_stats）。
    - 下一层输入 = 按第 1 层 gain 重要性筛选后的原始特征（feature_keep）+ 新增特征。
    - 折模型训练时每个站点最多用 max_per_site 条记录（OOF 仍覆盖全部样本）。

    分折：groups（推荐空间簇）分组，stratify（推荐 IGBP）分层 → StratifiedGroupKFold。
    选层与调参：按（加权）OOF 分数，或给定的验证集；tune_trials > 0 时第 1 层随机搜索参数。
    训练结束只保留最优层。

    预测：refit=True 时每个位置用全部（抽样后）训练数据重训一个模型；
          predict(X, return_std=True) 返回最后一层全部折模型预测的标准差（模型方差，相对指标）。
    设备：use_gpu / gpu_id；训练后可用 set_device('cpu' | 'cuda' | 'cuda:1') 切换。
    """

    DEFAULT_XGB = {
        "objective": "reg:squarederror",
        "n_estimators": 2000,          # 上限；实际树数由早停决定
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_alpha": 0,
        "reg_lambda": 1,
    }

    # 动态调参的搜索范围：(最小值, 最大值, 类型)；"log" 表示在对数尺度上取值
    TUNE_SPACE = {
        "max_depth": (3, 10, "int"),
        "min_child_weight": (1, 20, "log"),
        "learning_rate": (0.02, 0.2, "log"),
        "subsample": (0.5, 1.0, "float"),
        "colsample_bytree": (0.5, 1.0, "float"),
        "reg_lambda": (0.1, 20, "log"),
        "reg_alpha": (1e-3, 5, "log"),
        "gamma": (0, 2, "float"),
    }

    def __init__(self, n_estimators_per_layer=2, max_layers=6, early_stopping_rounds=2,
                 n_folds=5, tree_early_stopping=50, xgb_params=None,
                 refit=True, use_gpu=False, gpu_id=0, eval_metric="r2",
                 feature_keep=0.8, add_output_stats=True, max_per_site=None,
                 tune_trials=0, tune_space=None, retune_each_layer=False,
                 keep_fold_models=True, random_state=42, verbose=1):
        self.n_estimators_per_layer = n_estimators_per_layer
        self.max_layers = max_layers
        self.early_stopping_rounds = early_stopping_rounds
        self.n_folds = n_folds
        self.tree_early_stopping = tree_early_stopping
        self.refit = refit
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.eval_metric = eval_metric.lower()
        self.feature_keep = feature_keep
        self.add_output_stats = add_output_stats
        self.max_per_site = max_per_site
        self.tune_trials = tune_trials
        self.tune_space = tune_space or self.TUNE_SPACE
        self.retune_each_layer = retune_each_layer
        self.keep_fold_models = keep_fold_models
        self.random_state = random_state
        self.verbose = verbose

        params = dict(self.DEFAULT_XGB)
        if xgb_params:
            params.update(xgb_params)
        params.pop("random_state", None)
        params.pop("seed", None)
        self.xgb_params = params

        self.layers = []              # 预测用模型：每层一个列表（refit 模型或折模型列表）
        self.fold_layers = []         # 每层每个位置的折模型（用于不确定性）
        self.keep_idx = None          # 第 2 层起保留的原始特征列号
        self.feature_importance = None
        self.folds = None
        self.best_score = None
        self.history = None
        self.tuning = []

    # ------------------------------------------------------------------
    @property
    def device(self):
        return f"cuda:{self.gpu_id}" if self.use_gpu else "cpu"

    def _better(self, a, b):
        return a > b if self.eval_metric == "r2" else a < b

    def _score(self, y, p, w=None):
        if self.eval_metric == "r2":
            return r2_score(y, p, sample_weight=w)
        if self.eval_metric == "rmse":
            return np.sqrt(mean_squared_error(y, p, sample_weight=w))
        if self.eval_metric == "mae":
            return mean_absolute_error(y, p, sample_weight=w)
        raise ValueError(f"未知的 eval_metric：{self.eval_metric}")

    def _new_reg(self, seed, params, n_estimators=None, early_stop=True):
        p = dict(params)
        p.update(random_state=seed, tree_method="hist", device=self.device)
        if n_estimators is not None:
            p["n_estimators"] = n_estimators
        if early_stop:
            p["early_stopping_rounds"] = self.tree_early_stopping
        return XGBRegressor(**p)

    @staticmethod
    def _fit_reg(reg, X, y, w, X_eval=None, y_eval=None, w_eval=None):
        kw = {"sample_weight": w, "verbose": False}
        if X_eval is not None:
            kw["eval_set"] = [(X_eval, y_eval)]
            kw["sample_weight_eval_set"] = [w_eval]
        return reg.fit(X, y, **kw)

    def _cap_mask(self, site_ids, seed):
        """每个站点最多保留 max_per_site 条记录（随机），返回布尔掩膜。"""
        n = len(site_ids) if site_ids is not None else 0
        if not self.max_per_site or site_ids is None:
            return None
        rng = np.random.default_rng(seed)
        order = rng.permutation(n)
        s = pd.Series(np.asarray(site_ids)[order])
        rank = s.groupby(s).cumcount().to_numpy()
        mask = np.zeros(n, bool)
        mask[order[rank < self.max_per_site]] = True
        return mask

    def _make_folds(self, X, y, groups, stratify):
        if groups is not None and stratify is not None:
            spl = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            return list(spl.split(X, np.asarray(stratify).astype(str), groups))
        if groups is not None:
            return list(GroupKFold(n_splits=self.n_folds).split(X, y, groups))
        return list(KFold(self.n_folds, shuffle=True, random_state=self.random_state).split(X))

    def _sample_params(self, rng):
        p = dict(self.xgb_params)
        for k, (lo, hi, kind) in self.tune_space.items():
            if kind == "int":
                p[k] = int(rng.integers(lo, hi + 1))
            elif kind == "log":
                p[k] = float(np.exp(rng.uniform(np.log(max(lo, 1e-12)), np.log(hi))))
            else:
                p[k] = float(rng.uniform(lo, hi))
        return p

    def _stats(self, out):
        if not self.add_output_stats:
            return out
        return np.hstack([out, out.mean(axis=1, keepdims=True), out.std(axis=1, keepdims=True)])

    def _next_input(self, X, out):
        base = X if self.keep_idx is None else X[:, self.keep_idx]
        return np.hstack([base, self._stats(out)]).astype(np.float32)

    @staticmethod
    def _predict_unit(unit, X):
        if isinstance(unit, list):
            return np.mean([m.predict(X) for m in unit], axis=0)
        return unit.predict(X)

    def _layer_outputs(self, layer, X):
        return np.column_stack([self._predict_unit(u, X) for u in layer]).astype(np.float32)

    # ------------------------------------------------------------------
    def _tune(self, X_cur, y, w, folds, cap, layer_index):
        rng = np.random.default_rng(self.random_state + layer_index)
        candidates = [dict(self.xgb_params)] + [self._sample_params(rng) for _ in range(self.tune_trials)]
        best_p, best_s, records = None, None, []
        for i, p in enumerate(candidates):
            oof = np.zeros(len(y), dtype=np.float32)
            for k, (tr, te) in enumerate(folds):
                tr_used = tr if cap is None else tr[cap[tr]]
                reg = self._new_reg(self.random_state + k, p)
                self._fit_reg(reg, X_cur[tr_used], y[tr_used], w[tr_used], X_cur[te], y[te], w[te])
                oof[te] = reg.predict(X_cur[te])
            sc = self._score(y, oof, w)
            records.append({**{k: p[k] for k in self.tune_space}, "score": sc})
            if best_s is None or self._better(sc, best_s):
                best_p, best_s = p, sc
            if self.verbose:
                print(f"  调参 {i + 1}/{len(candidates)}：OOF {self.eval_metric.upper()} = {sc:.4f}"
                      + ("（初始参数）" if i == 0 else ""), flush=True)
        self.tuning.append({"layer": layer_index + 1, "best_score": best_s,
                            "best_params": best_p, "trials": pd.DataFrame(records)})
        return best_p

    # ------------------------------------------------------------------
    def fit(self, X, y, groups=None, stratify=None, sample_weight=None, site_ids=None,
            X_val=None, y_val=None, w_val=None):
        """
        X, y           训练特征和目标（NaN 可直接保留）
        groups         分折分组（推荐空间簇编号）
        stratify       分折分层标签（推荐 IGBP）
        sample_weight  逐行权重
        site_ids       逐行站点 ID（用于 max_per_site）；不给则用 groups
        X_val, y_val   可选：给了按验证集选层，否则按 OOF 分数
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()
        n, n_orig = X.shape
        w = np.ones(n, np.float32) if sample_weight is None else np.asarray(sample_weight, np.float32)
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32).ravel()
            w_val = np.ones(len(y_val), np.float32) if w_val is None else np.asarray(w_val, np.float32)
        site_ids = site_ids if site_ids is not None else groups

        folds = self._make_folds(X, y, groups, stratify)
        self.folds = folds
        self.keep_idx = None
        self.tuning = []

        best_score, best_n, bad = None, 0, 0
        layers, fold_layers, history = [], [], []
        X_cur, Xv_cur = X, X_val
        params = dict(self.xgb_params)

        for L in range(self.max_layers):
            cap = self._cap_mask(site_ids, self.random_state + 7919 * L)
            if self.verbose and cap is not None and L == 0:
                print(f"每站最多 {self.max_per_site} 条：训练使用 {cap.sum():,} / {n:,} 条", flush=True)

            if self.tune_trials and (L == 0 or self.retune_each_layer):
                if self.verbose:
                    print(f"Layer {L + 1}：动态调参（初始参数 + {self.tune_trials} 组随机参数）", flush=True)
                params = self._tune(X_cur, y, w, folds, cap, L)

            oof = np.zeros((n, self.n_estimators_per_layer), dtype=np.float32)
            layer, fold_layer = [], []
            importances = np.zeros(X_cur.shape[1])
            for e in range(self.n_estimators_per_layer):
                fold_models, best_iters = [], []
                for k, (tr, te) in enumerate(folds):
                    tr_used = tr if cap is None else tr[cap[tr]]
                    reg = self._new_reg(self.random_state + 1000 * L + 100 * e + k, params)
                    self._fit_reg(reg, X_cur[tr_used], y[tr_used], w[tr_used], X_cur[te], y[te], w[te])
                    oof[te, e] = reg.predict(X_cur[te])
                    fold_models.append(reg)
                    best_iters.append(getattr(reg, "best_iteration", None))
                    imp = getattr(reg, "feature_importances_", None)
                    if imp is not None:
                        importances += np.nan_to_num(np.asarray(imp, float))

                if self.refit:
                    iters = [b for b in best_iters if b is not None]
                    n_trees = int(np.mean(iters)) + 1 if iters else None
                    full = self._new_reg(self.random_state + 1000 * L + 100 * e + 99, params,
                                         n_estimators=n_trees, early_stop=False)
                    used = np.arange(n) if cap is None else np.flatnonzero(cap)
                    self._fit_reg(full, X_cur[used], y[used], w[used])
                    layer.append(full)
                else:
                    layer.append(fold_models)
                fold_layer.append(fold_models if self.keep_fold_models else None)
            layers.append(layer)
            fold_layers.append(fold_layer)

            # ---------- 第 1 层后：按 gain 重要性筛选原始特征 ----------
            if L == 0:
                imp = importances[:n_orig]
                self.feature_importance = imp / imp.sum() if imp.sum() > 0 else np.full(n_orig, 1 / n_orig)
                if self.feature_keep and self.feature_keep < 1:
                    k_keep = max(1, int(np.ceil(self.feature_keep * n_orig)))
                    self.keep_idx = np.sort(np.argsort(-self.feature_importance)[:k_keep])

            # ---------- 本层分数 ----------
            oof_score = self._score(y, oof.mean(axis=1), w)
            if has_val:
                v_out = self._layer_outputs(layer, Xv_cur)
                score = self._score(y_val, v_out.mean(axis=1), w_val)
            else:
                score = oof_score
            history.append({"layer": L + 1, "oof_score": oof_score,
                            "val_score": score if has_val else np.nan, "n_features": X_cur.shape[1]})
            if self.verbose:
                metric = self.eval_metric.upper()
                print(f"Training Layer {L + 1}", flush=True)
                if has_val:
                    print(f"Validation {metric}: {score:.4f}（OOF {metric}: {oof_score:.4f}）", flush=True)
                else:
                    print(f"Validation {metric}: {score:.4f}（按站点交叉验证，OOF）", flush=True)

            if best_score is None or self._better(score, best_score):
                best_score, best_n, bad = score, L + 1, 0
                if self.verbose:
                    print(f"✅ New best {self.eval_metric.upper()}: {best_score:.4f}", flush=True)
            else:
                bad += 1
                if bad >= self.early_stopping_rounds:
                    if self.verbose:
                        print("Early stopping triggered.", flush=True)
                    break

            X_cur = self._next_input(X, oof)
            if has_val:
                Xv_cur = self._next_input(X_val, v_out)

        self.layers = layers[:best_n]
        self.fold_layers = fold_layers[:best_n]
        self.best_score = best_score
        self.history = pd.DataFrame(history)
        if self.verbose:
            print(f"保留 {best_n} 层，最优 {self.eval_metric.upper()} = {best_score:.4f}", flush=True)
        return self

    # ------------------------------------------------------------------
    def predict(self, X, return_std=False):
        X = np.asarray(X, dtype=np.float32)
        X_cur, out, X_last = X, None, X
        for layer in self.layers:
            X_last = X_cur
            out = self._layer_outputs(layer, X_cur)
            X_cur = self._next_input(X, out)
        mean = out.mean(axis=1)
        if not return_std:
            return mean
        last = self.fold_layers[-1] if self.fold_layers else None
        if not last or any(f is None for f in last):
            raise ValueError("没有保存折模型，无法计算不确定性：训练时请设 keep_fold_models=True")
        preds = np.column_stack([m.predict(X_last) for models in last for m in models])
        return mean, preds.std(axis=1)

    def best_params(self):
        return [t["best_params"] for t in self.tuning]

    def set_device(self, device):
        """训练后切换预测设备：'cpu'、'cuda'、'cuda:1'。"""
        self.use_gpu = device.startswith("cuda")
        if ":" in device:
            self.gpu_id = int(device.split(":")[1])
        for layer in list(self.layers) + list(self.fold_layers):
            for unit in layer:
                if unit is None:
                    continue
                for m in (unit if isinstance(unit, list) else [unit]):
                    m.set_params(device=device)
        return self

    def n_models(self):
        return sum(len(u) if isinstance(u, list) else 1 for layer in self.layers for u in layer)

# ============================================================================================================================================================
# Function mode

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

def srcnn(learning_rate=0.001):
    """
    Builds and compiles a Super-Resolution Convolutional Neural Network (SRCNN) model 
    that fuses features from both low-resolution and high-resolution images.

    This model uses two parallel input streams:
    - A low-resolution input which undergoes upscaling through convolutional layers.
    - A high-resolution input from which texture features are extracted and fused with the low-resolution stream.

    Args:
        save_path (str, optional): Path to save the compiled model. If None, the model is not saved.
        learning_rate (float): Learning rate for the Adam optimizer.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """
    # Input layers
    lowres_input = layers.Input(shape=(None, None, 1))  # Low-resolution input
    highres_input = layers.Input(shape=(None, None, 1))  # High-resolution image

    # Feature extraction from high-resolution image
    highres_features = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(highres_input)
    highres_features = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(highres_features)

    # Processing low-resoltuion input
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(lowres_input)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

    # Fusion of high-resolution image textures
    fusion = layers.Concatenate()([x, highres_features])
    fusion = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(fusion)
    fusion = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(fusion)

    # Output
    output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(fusion)

    model = keras.Model(inputs=[lowres_input, highres_input], outputs=output)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")

    return model

def print_model(model):
    return model.summary()

def train(lowres_data, highres_data, epochs=100, batch_size=1, verbose=1, save_path=None):
    model = srcnn()
    # Train SRCNN
    model.fit([lowres_data, highres_data], highres_data, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # Save the complete model
    # Recommended in newer versions of Keras (TensorFlow 2.11+): e.g., 'texture_fusion_model.keras'
    if save_path: model.save(save_path)

def apply(model, lowres_data_app, highres_data):
    super_resolved = model.predict([lowres_data_app, highres_data]).squeeze() 
    super_resolved = xr.DataArray(
        super_resolved, 
        dims = ("latitude", "longitude"), 
        coords={"latitude": highres_data.latitude, "longitude": highres_data.longitude}, 
        name="super_res"
    )
    return super_resolved

def load_model(save_path):
    model = load_model('texture_fusion_model.keras') 

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# Class mode

import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

class TextureFusionSRCNN:
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        # Input layers
        lowres_input = layers.Input(shape=(None, None, 1))  # Low-resolution input
        highres_input = layers.Input(shape=(None, None, 1))  # High-resolution image

        # Feature extraction from high-resolution image
        highres_features = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(highres_input)
        highres_features = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(highres_features)

        # Processing low-resolution input
        x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(lowres_input)
        x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)

        # Fusion of high-resolution image textures
        fusion = layers.Concatenate()([x, highres_features])
        fusion = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(fusion)
        fusion = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(fusion)

        # Output
        output = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(fusion)

        model = keras.Model(inputs=[lowres_input, highres_input], outputs=output)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate), loss="mse")

        return model

    def summary(self):
        return self.model.summary()

    def train(self, lowres_data, highres_data, epochs=100, batch_size=1, verbose=1, save_path=None):
        early_stop = EarlyStopping(
            monitor='loss',       # You can change to 'val_loss' if you add validation
            patience=10,          # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True
        )

        self.model.fit(
            [lowres_data, highres_data], highres_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[early_stop]
        )

        if save_path:
            self.model.save(save_path)

    def apply(self, lowres_data_app, highres_data):
        super_resolved = self.model.predict([lowres_data_app, highres_data]).squeeze()
        return super_resolved

    @staticmethod
    def load(save_path):
        model = keras.models.load_model(save_path)
        instance = TextureFusionSRCNN()
        instance.model = model
        return instance

