import numpy as np
import time
import os
import argparse
import configparser
import logging
import sys
import re
from datetime import datetime
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
import concurrent.futures
from scipy.sparse.linalg import splu

def load_data_safe(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    
    basename = os.path.basename(path)
    if os.path.exists(basename):
        logging.warning(f"File not found at {path}, using {basename} instead.")
        return np.load(basename, allow_pickle=True)

    data_path = os.path.join("data", basename)
    if os.path.exists(data_path):
        logging.warning(f"File not found at {path}, using {data_path} instead.")
        return np.load(data_path, allow_pickle=True)
        
    raise FileNotFoundError(f"Could not find file: {path} (or {basename})")

def compute_metrics_torch(var, var_hat, mask=None):
    if mask is None:
        mask = torch.ones_like(var, dtype=torch.bool)
    else:
        mask = mask & (~torch.isnan(var))
    
    if torch.sum(mask) == 0:
        return 0.0, 0.0, 0.0
        
    var = var[mask]
    var_hat = var_hat[mask]
    
    mae = torch.mean(torch.abs(var - var_hat)).item()
    rmse = torch.sqrt(torch.mean((var - var_hat) ** 2)).item()
    
    non_zero_mask = var != 0
    if torch.sum(non_zero_mask) > 0:
        mape = torch.mean(torch.abs(var[non_zero_mask] - var_hat[non_zero_mask]) / torch.abs(var[non_zero_mask])).item() * 100
    else:
        mape = 0.0
    return mae, rmse, mape

def compute_metrics_numpy(var, var_hat, mask=None):
    if mask is None:
        mask = ~np.isnan(var)
    else:
        mask = mask & (~np.isnan(var))
        
    var = var[mask]
    var_hat = var_hat[mask]
    
    if var.size == 0:
        return 0.0, 0.0, 0.0
    
    mae = np.mean(np.abs(var - var_hat))
    rmse = np.sqrt(np.mean((var - var_hat) ** 2))
    
    non_zero_mask = var != 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs(var[non_zero_mask] - var_hat[non_zero_mask]) / np.abs(var[non_zero_mask])) * 100
    else:
        mape = 0.0
    return mae, rmse, mape

class LATC_Inductive:
    def __init__(self, time_lags, theta=10, c=1.0, ranks=None, device='cpu'):
        self.time_lags = time_lags
        self.theta = theta
        self.c = c
        self.ranks = ranks
        self.device = device
        
        self.final_tau = None
        self.A_global = None
        self.U_bases = {}
        self.dim = None

    def save_model(self, path):
        if self.A_global is None:
            logging.warning("Model not fitted, nothing to save.")
            return
        
        save_dict = {
            'A_global': self.A_global,
            'time_lags': self.time_lags,
            'dim': self.dim,
            'final_tau': self.final_tau if self.final_tau is not None else 0.0
        }
        for mode, U in self.U_bases.items():
            save_dict[f'U_{mode}'] = U
            
        np.savez_compressed(path, **save_dict)
        logging.info(f"[Model] Parameters saved to {path}")

    def load_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        data = np.load(path)
        self.A_global = data['A_global']
        if 'dim' in data:
            self.dim = data['dim']
        if 'final_tau' in data:
            self.final_tau = float(data['final_tau'])
        
        self.U_bases = {}
        for key in data.files:
            if key.startswith('U_'):
                mode = int(key.split('_')[1])
                self.U_bases[mode] = data[key]
        logging.info(f"[Model] Parameters loaded from {path}")

    def _ten2mat(self, tensor, mode):
        if isinstance(tensor, np.ndarray):
            return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order='C')
        return torch.reshape(torch.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

    def _mat2ten(self, mat, tensor_size, mode):
        index = list()
        index.append(mode)
        for i in range(len(tensor_size)):
            if i != mode:
                index.append(i)
        
        target_shape = [tensor_size[mode]]
        for i in range(len(tensor_size)):
            if i != mode:
                target_shape.append(tensor_size[i])
        
        if isinstance(mat, np.ndarray):
             return np.moveaxis(np.reshape(mat, target_shape, order='C'), 0, mode)
                
        mat = torch.reshape(mat, target_shape)
        return torch.moveaxis(mat, 0, mode)

    def _svt_tnn(self, mat, tau, theta):
        [m, n] = mat.shape
        if 2 * m < n:
            w = torch.matmul(mat, mat.T)
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            s = torch.sqrt(s)
            idx = torch.sum(s > tau)
            mid = torch.zeros(idx, device=mat.device, dtype=mat.dtype)
            if idx > 0:
                if theta < idx:
                    mid[:theta] = 1.0
                    s_part = s[theta:idx]
                    mid[theta:idx] = (s_part - tau) / s_part
                else:
                    mid[:] = 1.0
            u_k = u[:, :idx]
            tmp = u_k * mid.unsqueeze(0) 
            return torch.matmul(torch.matmul(tmp, u_k.T), mat)
        elif m > 2 * n:
            return self._svt_tnn(mat.T, tau, theta).T
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        idx = torch.sum(s > tau)
        vec = s[:idx].clone()
        if idx > theta:
            vec[theta:idx] = s[theta:idx] - tau
        u_k = u[:, :idx]
        vh_k = vh[:idx, :]
        return torch.matmul(u_k * vec.unsqueeze(0), vh_k)

    def _generate_Psi(self, dim_time, time_lags):
        Psis = []
        max_lag = np.max(time_lags)
        for i in range(len(time_lags) + 1):
            row = np.arange(0, dim_time - max_lag)
            if i == 0:
                col = np.arange(0, dim_time - max_lag) + max_lag
            else:
                col = np.arange(0, dim_time - max_lag) + max_lag - time_lags[i - 1]
            data = np.ones(dim_time - max_lag)
            Psi = sparse.coo_matrix((data, (row, col)), shape=(dim_time - max_lag, dim_time))
            Psis.append(Psi)
        return Psis

    def fit(self, dense_tensor, sparse_tensor, maxiter=50, epsilon=1e-4, K=3, rho0=1e-5):
        logging.info(f"[Fit] Starting training on data {sparse_tensor.shape} (Device: {self.device})...")
        start_time = time.time()

        dim = np.array(sparse_tensor.shape)
        self.dim = dim
        dim_time = np.int32(np.prod(dim) / dim[0])
        d = len(self.time_lags)
        max_lag = np.max(self.time_lags)

        sparse_tensor_torch = torch.tensor(sparse_tensor, dtype=torch.float32, device=self.device)
        dense_tensor_torch = torch.tensor(dense_tensor, dtype=torch.float32, device=self.device)

        mask_test = (dense_tensor_torch != 0) & (sparse_tensor_torch == 0)
        dense_test = dense_tensor_torch[mask_test]

        sparse_mat = self._ten2mat(sparse_tensor_torch, 0)
        pos_missing = (sparse_mat == 0)

        T_tensor = torch.zeros(tuple(dim), device=self.device)
        Z_tensor = sparse_tensor_torch.clone()
        Z = sparse_mat.clone()
        A = 0.001 * torch.rand((dim[0], d), device=self.device)

        Psis = self._generate_Psi(dim_time, self.time_lags)
        iden = sparse.coo_matrix((np.ones(dim_time), (np.arange(0, dim_time), np.arange(0, dim_time))),
                                 shape=(dim_time, dim_time))

        ind = np.zeros((d, dim_time - max_lag), dtype=np.int64)
        for i in range(d):
            ind[i, :] = np.arange(max_lag - self.time_lags[i], dim_time - self.time_lags[i])

        last_mat = sparse_mat.clone()
        snorm = torch.norm(sparse_mat, p='fro')
        rho = rho0
        lambda0 = self.c * rho

        for it in range(maxiter):
            temp_list = []
            A_cpu = A.cpu().numpy()
            for m in range(dim[0]):
                Psis0 = [P.copy() for P in Psis]
                for i in range(d):
                    Psis0[i + 1] = A_cpu[m, i] * Psis[i + 1]
                B = Psis0[0] - sum(Psis0[1:])
                temp_list.append(B.T @ B)

            for k in range(K):
                rho = min(rho * 1.05, 1e5)

                tensor_hat = torch.zeros(tuple(dim), device=self.device)
                for p in range(len(dim)):
                    tensor_hat += (1 / 3) * self._mat2ten(
                        self._svt_tnn(self._ten2mat(Z_tensor - T_tensor / rho, p),
                                      (1 / 3) / rho, self.theta), dim, p)

                temp0 = rho / lambda0 * self._ten2mat(tensor_hat + T_tensor / rho, 0)
                temp0_cpu = temp0.cpu().numpy()
                mat_z_cpu = np.zeros((dim[0], dim_time))

                rho_lambda_iden = rho * iden / lambda0
                for m in range(dim[0]):
                    mat_z_cpu[m, :] = spsolve(temp_list[m] + rho_lambda_iden, temp0_cpu[m, :])

                mat_z = torch.tensor(mat_z_cpu, dtype=torch.float32, device=self.device)
                Z[pos_missing] = mat_z[pos_missing]
                Z_tensor = self._mat2ten(Z, dim, 0)
                T_tensor = T_tensor + rho * (tensor_hat - Z_tensor)

            for m in range(dim[0]):
                zm = Z[m]
                ind_flat = torch.tensor(ind.flatten(), device=self.device, dtype=torch.long)
                zm_ind_flat = zm[ind_flat]
                zm_ind = zm_ind_flat.reshape(d, -1)
                
                target = zm[max_lag:]
                X = zm_ind.T
                y = target

                sol = torch.linalg.lstsq(X, y).solution
                A[m, :] = sol

            mat_hat = self._ten2mat(tensor_hat, 0)
            tol = torch.norm((mat_hat - last_mat), p='fro') / snorm
            last_mat = mat_hat.clone()

            if (it + 1) % 10 == 0:
                if dense_test.numel() > 0:
                    pred_test = tensor_hat[mask_test]
                    mae, rmse, mape = compute_metrics_torch(dense_test, pred_test)
                    logging.info(f"  [Fit Iter {it + 1}] Tol: {tol.item():.6f} | MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
                else:
                    logging.info(f"  [Fit Iter {it + 1}] Tol: {tol.item():.6f}")

            if tol < epsilon:
                break

        logging.info("[Fit] Training finished. Extracting bases...")
        self.A_global = A.cpu().numpy()

        self.final_tau = 1.0 / (3.0 * rho)
        logging.info(f"[Fit] Final Tau: {self.final_tau:.4f}")

        for mode in range(len(dim)):
            mat_mode = self._ten2mat(Z_tensor, mode)
            U, S, _ = torch.linalg.svd(mat_mode, full_matrices=False)

            r = len(S)
            logging.info(f"  [Fit] Mode {mode} Saving Full Rank: {r}")

            self.U_bases[mode] = U[:, :r].cpu().numpy()

        logging.info(f"[Fit] Done. Time: {time.time() - start_time:.2f}s")
        return self

    def predict(self, sparse_tensor, dense_tensor=None, maxiter=100, epsilon=1e-4):
        if self.A_global is None:
            raise ValueError("Model not fitted yet!")

        logging.info(f"[Predict] Starting Fast Inference (Device: {self.device}, MaxIter: {maxiter}, Epsilon: {epsilon})...")
        start_time = time.time()

        sparse_tensor_torch = torch.tensor(sparse_tensor, dtype=torch.float32, device=self.device)

        mask_eval = None
        dense_eval = None
        if dense_tensor is not None:
            dense_tensor_torch = torch.tensor(dense_tensor, dtype=torch.float32, device=self.device)
            mask_eval = (dense_tensor_torch != 0) & (sparse_tensor_torch == 0)
            dense_eval = dense_tensor_torch[mask_eval]

        dim = np.array(sparse_tensor_torch.shape)
        dim_time = np.int32(np.prod(dim) / dim[0])
        d = len(self.time_lags)

        Projections = {}
        
        for mode, U_np in self.U_bases.items():
            U_full = torch.tensor(U_np, dtype=torch.float32, device=self.device)
            
            if self.ranks is not None and mode < len(self.ranks):
                r_use = self.ranks[mode]
            else:
                r_use = max(self.theta, int(U_full.shape[0] * 0.2))
                logging.info(f"  [Predict] Mode {mode} Truncating Rank to {r_use}")
            
            U_trunc = U_full[:, :r_use]
            Projections[mode] = torch.matmul(U_trunc, U_trunc.T)

        A_global_cpu = self.A_global

        Psis = self._generate_Psi(dim_time, self.time_lags)
        T_tensor = torch.zeros(tuple(dim), device=self.device)
        sparse_mat = self._ten2mat(sparse_tensor_torch, 0)
        pos_missing = (sparse_mat == 0)
        Z = sparse_mat.clone()
        Z_tensor = sparse_tensor_torch.clone()

        rho = 1e-5
        lambda0 = self.c * rho
        iden = sparse.coo_matrix((np.ones(dim_time), (np.arange(0, dim_time), np.arange(0, dim_time))),
                                 shape=(dim_time, dim_time))

        engine = 'gpu' if (dim_time <= 2048 and str(self.device) != 'cpu') else 'cpu'
        logging.info(f"  [Predict] T={dim_time}. Smart Router Selected Engine: {engine.upper()}")

        cpu_solvers = []
        gpu_L_factor = None
        rho_lambda_val = 1.0 / self.c
        rho_lambda_iden = rho_lambda_val * iden

        t_pre_start = time.time()
        if engine == 'cpu':
            for m in range(dim[0]):
                Psis0 = [P.copy() for P in Psis]
                for i in range(d):
                    Psis0[i + 1] = A_global_cpu[m, i] * Psis[i + 1]
                B = Psis0[0] - sum(Psis0[1:])
                lhs = B.T @ B + rho_lambda_iden
                cpu_solvers.append(splu(lhs.tocsc()))
        else:
            LHS_list = []
            for m in range(dim[0]):
                Psis0 = [P.copy() for P in Psis]
                for i in range(d):
                    Psis0[i + 1] = A_global_cpu[m, i] * Psis[i + 1]
                B = Psis0[0] - sum(Psis0[1:])
                lhs = B.T @ B + rho_lambda_iden
                LHS_list.append(lhs.toarray())

            LHS_tensor = torch.tensor(np.stack(LHS_list), dtype=torch.float32, device=self.device)
            gpu_L_factor = torch.linalg.cholesky(LHS_tensor)

        logging.info( f"  [Predict] Precompute {engine.upper()} factorizations done in: {time.time() - t_pre_start:.4f}s")
        last_tensor_hat = torch.zeros(tuple(dim), device=self.device)
        snorm = torch.norm(sparse_tensor_torch, p='fro')
        if snorm == 0: snorm = 1.0

        for it in range(maxiter):
            rho = min(rho * 1.05, 1e5)
            lambda0 = self.c * rho

            tensor_hat = torch.zeros(tuple(dim), device=self.device)
            for mode in range(len(dim)):
                mat_input = self._ten2mat(Z_tensor - T_tensor / rho, mode)
                rec_mode = torch.matmul(Projections[mode], mat_input)
                tensor_hat += (1 / 3) * self._mat2ten(rec_mode, dim, mode)

            rhs_base = rho / lambda0 * self._ten2mat(tensor_hat + T_tensor / rho, 0)
            if engine == 'gpu':
                rhs_expanded = rhs_base.unsqueeze(-1)
                mat_z_tensor = torch.cholesky_solve(rhs_expanded, gpu_L_factor)
                mat_z = mat_z_tensor.squeeze(-1)
            else:
                rhs_base_cpu = rhs_base.cpu().numpy()
                mat_z_cpu = np.zeros((dim[0], dim_time))

                def solve_node(m):
                    return cpu_solvers[m].solve(rhs_base_cpu[m, :])

                num_workers = min(32, (os.cpu_count() or 1) + 4)
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = {executor.submit(solve_node, m): m for m in range(dim[0])}
                    for future in concurrent.futures.as_completed(futures):
                        m = futures[future]
                        mat_z_cpu[m, :] = future.result()

                mat_z = torch.tensor(mat_z_cpu, dtype=torch.float32, device=self.device)
            Z[pos_missing] = mat_z[pos_missing]

            Z_tensor = self._mat2ten(Z, dim, 0)
            T_tensor = T_tensor + rho * (tensor_hat - Z_tensor)

            tol = torch.norm((tensor_hat - last_tensor_hat), p='fro') / snorm
            last_tensor_hat = tensor_hat.clone()

            if dense_eval is not None and dense_eval.numel() > 0:
                pred_eval = tensor_hat[mask_eval]
                mae, rmse, mape = compute_metrics_torch(dense_eval, pred_eval)
                logging.info(f"  [Predict Iter {it + 1}] Tol: {tol.item():.6f} | MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")
            else:
                logging.info(f"  [Predict Iter {it + 1}] Tol: {tol.item():.6f}")

            if tol < epsilon:
                logging.info(f"  [Predict] Converged at Iter {it + 1} (Tol < {epsilon}). Stopping early.")
                break

        logging.info(f"[Predict] Inference done. Time: {time.time() - start_time:.2f}s")

        logging.info("[Predict] Calculating Uncertainty...")
        max_lag = np.max(self.time_lags)
        A_global_torch = torch.tensor(self.A_global, dtype=torch.float32, device=self.device)

        sigma_local = torch.zeros_like(Z)
        for m in range(dim[0]):
            pred_m = torch.zeros(dim_time, device=self.device)
            for k in range(len(self.time_lags)):
                lag = self.time_lags[k]
                coeff = A_global_torch[m, k]
                pred_m[max_lag:] += coeff * Z[m, (max_lag - lag): (dim_time - lag)]
            sigma_local[m, :] = torch.abs(Z[m, :] - pred_m)

        mean_sigma = torch.mean(sigma_local[:, max_lag:], dim=1, keepdim=True)
        sigma_local[:, :max_lag] = mean_sigma

        sigma_global_acc = torch.zeros(tuple(dim), device=self.device)
        for mode in range(len(dim)):
            mat_mode = self._ten2mat(self._mat2ten(Z, dim, 0), mode)
            rec_mode = torch.matmul(Projections[mode], mat_mode)
            diff_mode = torch.abs(mat_mode - rec_mode)
            sigma_global_acc += self._mat2ten(diff_mode, dim, mode)
        sigma_global = self._ten2mat(sigma_global_acc / len(dim), 0)

        sigma_total = torch.sqrt(sigma_local ** 2 + sigma_global ** 2)

        tensor_hat_final = self._mat2ten(Z, dim, 0).cpu().numpy()
        sigma_tensor_final = self._mat2ten(sigma_total, dim, 0).cpu().numpy()

        return tensor_hat_final, sigma_tensor_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='configurations/PEMS04.conf', type=str, help="configuration file path")
    parser.add_argument("--mode", default='inference', choices=['train', 'inference'], type=str, help="Run mode: train (fit & save) or inference (load & predict)")
    parser.add_argument("--model_path", default='./latc_model_params.npz', type=str, help="Path to save/load model parameters")
    parser.add_argument("--maxiter", default=50, type=int, help="max iterations for training")
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    
    dataset_name = os.path.basename(args.config).replace('.conf', '')

    miss_rate_str = "unknown" # holdplace
    miss_pattern = "unknown" # holdplace
    try:
        miss_file_path = config['Data']['miss_graph_signal_matrix_filename']
        miss_basename = os.path.basename(miss_file_path)
        match_rate = re.search(r'(\d+\.\d+)', miss_basename)
        if match_rate: miss_rate_str = match_rate.group(1)

        known_patterns = ["SC-TC", "SR-TC"]
        for p in known_patterns:
            if p in miss_basename:
                miss_pattern = p
                break
    except Exception:
        pass

    script_dir = os.path.dirname(os.path.abspath(__file__))

    log_dir = os.path.join(script_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_online_{args.mode}_{dataset_name}_{miss_pattern}_{miss_rate_str}_{current_time_str}.txt")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_filename}")
    logger.info(f"Read configuration file: {args.config}")
    logger.info(f"Mode: {args.mode}")

    data_config = config['Data']
    training_config = config['Training']
    
    points_per_day = int(data_config['points_per_day'])
    c = float(training_config['c'])
    theta = int(training_config['theta'])

    device = training_config.get('device', 'cpu')

    if torch.cuda.is_available() and device != 'cpu':
        if device == 'cuda': device = 'cuda:0'
        pass
    else:
        device = 'cpu'
        
    logger.info(f"Device configuration: {device}")
    
    logger.info("Loading Data...")
    try:
        gt_data = load_data_safe(data_config['graph_signal_matrix_filename'])['data']
        sparse_data = load_data_safe(data_config['miss_graph_signal_matrix_filename'])['data']
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    if gt_data.ndim == 3: gt_data = gt_data[:, :, 0]
    if sparse_data.ndim == 3: sparse_data = sparse_data[:, :, 0]
    
    T_original, N = gt_data.shape
    days = (T_original + points_per_day - 1) // points_per_day
    padded_len = days * points_per_day

    if padded_len > T_original:
        padding = np.zeros((padded_len - T_original, N), dtype=gt_data.dtype)
        gt_tensor_np = np.concatenate([gt_data, padding], axis=0)
        sparse_tensor_np = np.concatenate([sparse_data, padding], axis=0)
    else:
        gt_tensor_np = gt_data
        sparse_tensor_np = sparse_data

    gt_tensor = gt_tensor_np.reshape(days, points_per_day, N).transpose(2, 1, 0)
    sparse_tensor = sparse_tensor_np.reshape(days, points_per_day, N).transpose(2, 1, 0)
    
    logger.info(f"Tensor Shape: {sparse_tensor.shape}")

    model = LATC_Inductive(time_lags=np.arange(1, 7), theta=theta, c=c, ranks=None, device=device)

    params_dir = os.path.join(script_dir, "params")
    if not os.path.exists(params_dir):
        os.makedirs(params_dir)

    if args.model_path == './latc_model_params.npz':
        model_filename = f"latc_model_{dataset_name}_{miss_pattern}_{miss_rate_str}.npz"
        final_model_path = os.path.join(params_dir, model_filename)
    else:
        final_model_path = args.model_path

    if args.mode == 'train':
        logger.info("=== Training Mode ===")
        model.fit(gt_tensor, sparse_tensor, maxiter=args.maxiter)

        model.save_model(final_model_path)
        logger.info(f"Training finished. Model saved to {final_model_path}")
        return

    elif args.mode == 'inference':
        logger.info("=== Inference Mode ===")

        try:
            model.load_model(final_model_path)
        except FileNotFoundError:
            logger.error(f"Model file not found at {final_model_path}. Please run with --mode train first.")
            return

        logger.info("--- Step 1: Forward Imputation ---")
        rec_tensor_step1, sigma_tensor_step1 = model.predict(sparse_tensor, dense_tensor=gt_tensor, maxiter=100)

        if model.dim is None:
            model.dim = np.array(sparse_tensor.shape)

        logger.info("--- Step 2: Reverse Imputation ---")
        rec_mat_step1 = model._ten2mat(rec_tensor_step1, 0)
        sparse_mat_orig = model._ten2mat(sparse_tensor, 0)
        
        mask_observed = (sparse_mat_orig != 0)
        
        reverse_sparse_mat = rec_mat_step1.copy()
        reverse_sparse_mat[mask_observed] = 0 # Hide originally observed
        
        reverse_sparse_tensor = model._mat2ten(reverse_sparse_mat, model.dim, 0)
        
        rec_tensor_step2, sigma_tensor_step2 = model.predict(reverse_sparse_tensor, dense_tensor=gt_tensor, maxiter=100)
        
        logger.info("--- Step 3: Merging Results ---")
        rec_mat_step2 = model._ten2mat(rec_tensor_step2, 0)
        sigma_mat_step1 = model._ten2mat(sigma_tensor_step1, 0)
        sigma_mat_step2 = model._ten2mat(sigma_tensor_step2, 0)
        
        final_mat = np.zeros_like(rec_mat_step1)
        final_sigma_mat = np.zeros_like(sigma_mat_step1)
        
        final_mat[~mask_observed] = rec_mat_step1[~mask_observed]
        final_sigma_mat[~mask_observed] = sigma_mat_step1[~mask_observed]
        
        final_mat[mask_observed] = rec_mat_step2[mask_observed]
        final_sigma_mat[mask_observed] = sigma_mat_step2[mask_observed]
        
        final_mat = np.maximum(final_mat, 0)

        final_output = final_mat.reshape(N, points_per_day, days).transpose(2, 1, 0).reshape(-1, N)
        final_sigma_output = final_sigma_mat.reshape(N, points_per_day, days).transpose(2, 1, 0).reshape(-1, N)
        
        if final_output.shape[0] > T_original:
            final_output = final_output[:T_original, :]
            final_sigma_output = final_sigma_output[:T_original, :]

        pre_impute_dir = os.path.join(script_dir, "../data/pre_impute")
        pre_impute_dir = os.path.normpath(pre_impute_dir)
        
        if not os.path.exists(pre_impute_dir):
            os.makedirs(pre_impute_dir)
            
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"imputed_{dataset_name}_{miss_pattern}_{miss_rate_str}_{current_time_str}.npz"
        final_output_path = os.path.join(pre_impute_dir, output_filename)
        
        np.savez_compressed(final_output_path, 
                            imputed_data=final_output, 
                            sigma=final_sigma_output,
                            A_global=model.A_global)
        
        logger.info(f"Results saved to {final_output_path}")
        
        mask_all = (gt_data != 0)
        mae, rmse, mape = compute_metrics_numpy(gt_data, final_output, mask=mask_all)
        logger.info(f"Final Global Evaluation - MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%")


if __name__ == "__main__":
    main()
