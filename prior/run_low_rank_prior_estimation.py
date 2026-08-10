import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .models import LowRankPriorEstimator
except ImportError:
    from models import LowRankPriorEstimator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRUE_PATH = os.path.join(
    PROJECT_ROOT, "data", "miss_data", "PEMS04", "true_data_SC-TC_0.8_v2.npz"
)
DEFAULT_MISS_PATH = os.path.join(
    PROJECT_ROOT, "data", "miss_data", "PEMS04", "miss_data_SC-TC_0.8_v2.npz"
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(args):
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        missing_type = args.missing_type.replace("-", "")
        missing_rate = args.missing_rate.replace(".", "p")
        run_name = (
            f"{args.dataset_name}_miss{missing_type}_rate{missing_rate}_"
            f"layers{args.num_layers}_{timestamp}"
        )
    args.run_name = run_name

    log_file = args.log_file
    if log_file is None:
        log_dir = args.log_dir
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_{run_name}.log")
    else:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    log_handle = open(log_file, "a", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_handle)
    sys.stderr = Tee(original_stderr, log_handle)
    args.log_file = log_file
    print(f"Logging to: {log_file}")
    return log_handle, original_stdout, original_stderr


def sync_if_cuda(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_since(start_time, device=None):
    if device is not None:
        sync_if_cuda(device)
    return time.perf_counter() - start_time


class LowRankPriorWindowDataset(Dataset):
    def __init__(self, input_data, target_data, observed_mask, eval_mask, start, end, window):
        self.input_data = input_data.astype(np.float32)
        self.target_data = target_data.astype(np.float32)
        self.observed_mask = observed_mask.astype(bool)
        self.eval_mask = eval_mask.astype(bool)
        self.start = start
        self.end = end
        self.window = window
        self.indices = np.arange(start, end - window + 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        slc = slice(t, t + self.window)
        x = self.input_data[slc]
        y = self.target_data[slc]
        obs = self.observed_mask[slc]
        ev = self.eval_mask[slc]
        u = time_encoding(np.arange(t, t + self.window), self.input_data.shape[1])
        return {
            "x": torch.from_numpy(x[..., None]),
            "y": torch.from_numpy(y[..., None]),
            "mask": torch.from_numpy(obs[..., None]),
            "eval_mask": torch.from_numpy(ev[..., None]),
            "u": torch.from_numpy(u),
            "start": torch.tensor(t, dtype=torch.long),
        }


def time_encoding(indices, n_nodes, points_per_day=288):
    tod = (indices % points_per_day).astype(np.float32) / float(points_per_day)
    enc = np.stack(
        [
            np.sin(2.0 * np.pi * tod),
            np.cos(2.0 * np.pi * tod),
        ],
        axis=-1,
    )
    enc = np.repeat(enc[:, None, :], n_nodes, axis=1)
    return enc.astype(np.float32)


def load_sparse_traffic_npz(true_path, miss_path, channel):
    true_npz = np.load(true_path)
    miss_npz = np.load(miss_path)
    true_data = true_npz["data"][:, :, channel].astype(np.float32)
    miss_data = miss_npz["data"][:, :, channel].astype(np.float32)
    if "mask" in true_npz.files:
        target_mask = true_npz["mask"][:, :, channel].astype(bool)
    else:
        target_mask = true_data != 0
    if "mask" in miss_npz.files:
        observed_mask = miss_npz["mask"][:, :, channel].astype(bool)
    else:
        observed_mask = miss_data != 0
    eval_missing_mask = ~observed_mask

    return true_data, miss_data, target_mask, observed_mask, eval_missing_mask


def split_lengths(t, val_ratio, test_ratio):
    val_len = int(t * val_ratio)
    test_len = int(t * test_ratio)
    train_len = t - val_len - test_len
    return train_len, val_len, test_len


def make_prior_eval_mask(base_eval_missing, split_name, train_len, val_len, test_len):
    t = base_eval_missing.shape[0]
    out = np.zeros_like(base_eval_missing, dtype=bool)
    if split_name == "train":
        out[:train_len] = base_eval_missing[:train_len]
    elif split_name == "val":
        out[train_len : train_len + val_len] = base_eval_missing[
            train_len : train_len + val_len
        ]
    elif split_name == "test":
        out[t - test_len :] = base_eval_missing[t - test_len :]
    elif split_name == "all":
        out = base_eval_missing.astype(bool)
    else:
        raise ValueError(f"Unknown split: {split_name}")
    return out


def fit_standardizer(data, mask, train_len):
    values = data[:train_len][mask[:train_len]]
    if values.size == 0:
        raise ValueError("No observed training values found for normalization.")
    mean = float(values.mean())
    std = float(values.std())
    if std == 0:
        std = 1.0
    return mean, std


def masked_mae(pred, true, mask):
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask])))


def masked_rmse(pred, true, mask):
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred[mask] - true[mask]) ** 2)))


def masked_mape(pred, true, mask):
    mask = mask & (true != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(pred[mask] - true[mask]) / np.abs(true[mask])) * 100.0)


def masked_mean(values, mask):
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(values[mask]))


def interval_coverage(pred, prior_uncertainty, true, mask):
    if mask.sum() == 0:
        return float("nan")
    covered = (true[mask] >= pred[mask] - prior_uncertainty[mask]) & (true[mask] <= pred[mask] + prior_uncertainty[mask])
    return float(np.mean(covered))


def make_loader(input_data, target_data, observed_mask, eval_mask, start, end, window, batch_size, shuffle):
    dataset = LowRankPriorWindowDataset(input_data, target_data, observed_mask, eval_mask, start, end, window)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_prior_estimator(args, n_nodes, device):
    return LowRankPriorEstimator(
        num_nodes=n_nodes,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=args.input_embedding_dim,
        learnable_embedding_dim=args.learnable_embedding_dim,
        feed_forward_dim=args.feed_forward_dim,
        num_temporal_heads=args.num_temporal_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        windows=args.window,
        dim_proj=args.dim_proj,
        sigma_min=args.sigma_min,
    ).to(device)


def prior_interval_loss(y_hat, prior_uncertainty, y, mask, alpha):
    if mask.sum() == 0:
        return torch.zeros((), device=y_hat.device, dtype=y_hat.dtype)
    lower = y_hat - prior_uncertainty
    upper = y_hat + prior_uncertainty
    width = upper - lower
    below = torch.relu(lower - y)
    above = torch.relu(y - upper)
    score = width + (2.0 / alpha) * below + (2.0 / alpha) * above
    return score[mask].mean()


def train_prior_epoch(model, loader, optimizer, device, whiten_prob,
                    prior_interval_loss_weight, interval_alpha):
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device).bool()
        u = batch["u"].to(device)

        keep = torch.rand(mask.shape, device=device) > whiten_prob
        train_mask = mask & keep
        injected_missing = mask & ~train_mask
        if injected_missing.sum() == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        y_hat, prior_uncertainty = model(x, u, train_mask.float())
        rec_loss = torch.abs(y_hat - y)[injected_missing].mean()
        interval_loss = prior_interval_loss(y_hat, prior_uncertainty, y, injected_missing, interval_alpha)
        loss = rec_loss + prior_interval_loss_weight * interval_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate_prior_estimator(model, loader, device, mean, std):
    model.eval()
    total_abs = 0.0
    total_sq = 0.0
    total_ape = 0.0
    total_count = 0
    total_mape_count = 0
    total_prior_uncertainty = 0.0
    total_covered = 0
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device).float()
        eval_mask = batch["eval_mask"].to(device).bool()
        u = batch["u"].to(device)
        if eval_mask.sum() == 0:
            continue
        pred, prior_uncertainty = model(x, u, mask)
        pred = pred * std + mean
        prior_uncertainty = prior_uncertainty * std
        y = y * std + mean
        diff = pred[eval_mask] - y[eval_mask]
        y_eval = y[eval_mask]
        prior_uncertainty_eval = prior_uncertainty[eval_mask]
        nonzero = y_eval != 0
        covered = (
            (y_eval >= pred[eval_mask] - prior_uncertainty_eval)
            & (y_eval <= pred[eval_mask] + prior_uncertainty_eval)
        )
        total_abs += float(diff.abs().sum().cpu())
        total_sq += float((diff ** 2).sum().cpu())
        total_prior_uncertainty += float(prior_uncertainty_eval.sum().cpu())
        total_covered += int(covered.sum().cpu())
        total_count += int(eval_mask.sum().cpu())
        if nonzero.sum() > 0:
            total_ape += float((diff[nonzero].abs() / y_eval[nonzero].abs()).sum().cpu())
            total_mape_count += int(nonzero.sum().cpu())
    if total_count == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    mape = float("nan")
    if total_mape_count > 0:
        mape = 100.0 * total_ape / total_mape_count
    mean_prior_uncertainty = total_prior_uncertainty / total_count
    coverage = total_covered / total_count
    return total_abs / total_count, (total_sq / total_count) ** 0.5, mape, mean_prior_uncertainty, coverage


@torch.no_grad()
def estimate_prior_for_full_series(model, data_norm, observed_mask, device, window, batch_size):
    model.eval()
    t, n = data_norm.shape
    sums = np.zeros((t, n), dtype=np.float64)
    prior_uncertainty_sums = np.zeros((t, n), dtype=np.float64)
    counts = np.zeros((t, n), dtype=np.float64)

    dataset = LowRankPriorWindowDataset(
        data_norm,
        data_norm,
        observed_mask,
        np.ones_like(observed_mask, dtype=bool),
        0,
        t,
        window,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        x = batch["x"].to(device)
        mask = batch["mask"].to(device).float()
        u = batch["u"].to(device)
        starts = batch["start"].numpy()
        pred, prior_uncertainty = model(x, u, mask)
        pred = pred.squeeze(-1).cpu().numpy()
        prior_uncertainty = prior_uncertainty.squeeze(-1).cpu().numpy()
        for b, start in enumerate(starts):
            end = start + window
            sums[start:end] += pred[b]
            prior_uncertainty_sums[start:end] += prior_uncertainty[b]
            counts[start:end] += 1.0
    counts[counts == 0] = 1.0
    return (sums / counts).astype(np.float32), (prior_uncertainty_sums / counts).astype(np.float32)


def save_prior_checkpoint(path, model, args, mean, std):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "mean": mean,
            "std": std,
        },
        path,
    )


def load_prior_checkpoint(path, args, n_nodes, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    ckpt_args = argparse.Namespace(**ckpt["args"])
    model = build_prior_estimator(ckpt_args, n_nodes, device)
    model.load_state_dict(ckpt["state_dict"])
    return model, float(ckpt["mean"]), float(ckpt["std"])


def run(args):
    set_seed(args.seed)
    log_handle, original_stdout, original_stderr = setup_logging(args)
    print("=" * 80)
    print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Arguments: {vars(args)}")
    print(f"seed: {args.seed}")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"True data path: {args.true_path}")
    print(f"Miss data path: {args.miss_path}")
    true_data, miss_data, target_mask, observed_mask, eval_missing_mask = load_sparse_traffic_npz(
        args.true_path, args.miss_path, args.channel
    )
    t, n = true_data.shape
    train_len, val_len, test_len = split_lengths(t, args.val_ratio, args.test_ratio)
    print(f"Data shape: T={t}, N={n}")
    print(f"Split: train={train_len}, val={val_len}, test={test_len}")
    print(f"Observed points: {observed_mask.sum()}, artificial missing points: {eval_missing_mask.sum()}")
    print(
        "Artificial missing by split: "
        f"train={eval_missing_mask[:train_len].sum()}, "
        f"val={eval_missing_mask[train_len:train_len + val_len].sum()}, "
        f"test={eval_missing_mask[t - test_len:].sum()}"
    )

    mean, std = fit_standardizer(true_data, target_mask, train_len)
    data_norm = (true_data - mean) / std
    x_norm = np.where(observed_mask, data_norm, 0.0).astype(np.float32)

    val_eval = make_prior_eval_mask(eval_missing_mask, "val", train_len, val_len, args.test_ratio and test_len)
    test_eval = make_prior_eval_mask(eval_missing_mask, "test", train_len, val_len, test_len)

    train_loader = make_loader(
        x_norm, data_norm, observed_mask, observed_mask, 0, train_len, args.window, args.batch_size, True
    )
    val_loader = make_loader(
        x_norm,
        data_norm,
        observed_mask,
        val_eval,
        train_len,
        train_len + val_len,
        args.window,
        args.eval_batch_size,
        False,
    )
    test_loader = make_loader(
        x_norm,
        data_norm,
        observed_mask,
        test_eval,
        t - test_len,
        t,
        args.window,
        args.eval_batch_size,
        False,
    )

    run_name = args.run_name
    out_dir = os.path.join(args.output_dir, run_name)
    ckpt_path = args.checkpoint or os.path.join(out_dir, "best_low_rank_prior.pt")
    print(f"Output dir: {out_dir}")
    print(f"Checkpoint path: {ckpt_path}")
    print(f"Pre-impute output dir: {args.pre_impute_dir}")
    print(f"Model layers: {args.num_layers}")
    print(f"Temporal attention: linear")
    print(f"Spatial attention: linear")

    if args.mode in ("train", "both"):
        train_start = time.perf_counter()
        model = build_prior_estimator(args, n, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_val = float("inf")
        bad_epochs = 0
        completed_epochs = 0
        for epoch in range(1, args.epochs + 1):
            completed_epochs = epoch
            train_loss = train_prior_epoch(
                model, train_loader, optimizer, device, args.whiten_prob,
                args.prior_interval_loss_weight, args.interval_alpha
            )
            val_mae, val_rmse, val_mape, val_prior_uncertainty, val_coverage = evaluate_prior_estimator(
                model, val_loader, device, mean, std
            )
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} "
                f"| val_mae={val_mae:.4f} | val_rmse={val_rmse:.4f} "
                f"| val_mape={val_mape:.4f}% | val_prior_uncertainty={val_prior_uncertainty:.4f} "
                f"| val_coverage={val_coverage:.4f}"
            )
            if val_mae < best_val:
                best_val = val_mae
                bad_epochs = 0
                save_prior_checkpoint(ckpt_path, model, args, mean, std)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch}. Best val_mae={best_val:.4f}")
                    break
        train_seconds = elapsed_since(train_start, device)
        print(
            f"Training finished in {train_seconds:.2f}s "
            f"({train_seconds / 60.0:.2f} min) over {completed_epochs} epoch(s)."
        )
        print(f"Batch sizes used: train_batch_size={args.batch_size}, eval_test_batch_size={args.eval_batch_size}")
        print(f"Prior interval loss weight: {args.prior_interval_loss_weight}")
        print("Temporal attention: linear")
        print("Spatial attention: linear")

    if args.mode in ("impute", "both"):
        impute_total_start = time.perf_counter()
        model, mean, std = load_prior_checkpoint(ckpt_path, args, n, device)
        test_eval_start = time.perf_counter()
        test_mae, test_rmse, test_mape, test_prior_uncertainty, test_coverage = evaluate_prior_estimator(
            model, test_loader, device, mean, std
        )
        test_eval_seconds = elapsed_since(test_eval_start, device)
        print(f"Loaded checkpoint: {ckpt_path}")
        print(
            f"Test missing MAE={test_mae:.4f}, RMSE={test_rmse:.4f}, "
            f"MAPE={test_mape:.4f}%, mean_prior_uncertainty={test_prior_uncertainty:.4f}, "
            f"coverage={test_coverage:.4f}"
        )
        print(f"Test evaluation time: {test_eval_seconds:.2f}s")

        print("Step 1: forward imputation for originally missing entries.")
        step1_start = time.perf_counter()
        pred_step1_norm, prior_uncertainty_step1_norm = estimate_prior_for_full_series(
            model, x_norm, observed_mask, device, args.window, args.eval_batch_size
        )
        step1_seconds = elapsed_since(step1_start, device)
        pred_step1 = pred_step1_norm * std + mean
        prior_uncertainty_step1 = prior_uncertainty_step1_norm * std
        filled_step1 = np.where(observed_mask, true_data, pred_step1)

        test_mask = np.zeros_like(eval_missing_mask, dtype=bool)
        test_mask[t - test_len :] = eval_missing_mask[t - test_len :]
        print(
            "Step 1 test-missing metrics: "
            f"MAE={masked_mae(pred_step1, true_data, test_mask):.4f}, "
            f"RMSE={masked_rmse(pred_step1, true_data, test_mask):.4f}, "
            f"MAPE={masked_mape(pred_step1, true_data, test_mask):.4f}%, "
            f"mean_prior_uncertainty={masked_mean(prior_uncertainty_step1, test_mask):.4f}, "
            f"coverage={interval_coverage(pred_step1, prior_uncertainty_step1, true_data, test_mask):.4f}"
        )
        print(f"Step 1 forward imputation time: {step1_seconds:.2f}s")

        print("Step 2: reverse imputation for originally observed entries.")
        step2_start = time.perf_counter()
        reverse_observed_mask = eval_missing_mask
        reverse_x_norm = np.where(reverse_observed_mask, (filled_step1 - mean) / std, 0.0).astype(
            np.float32
        )
        pred_step2_norm, prior_uncertainty_step2_norm = estimate_prior_for_full_series(
            model, reverse_x_norm, reverse_observed_mask, device, args.window, args.eval_batch_size
        )
        step2_seconds = elapsed_since(step2_start, device)
        pred_step2 = pred_step2_norm * std + mean
        prior_uncertainty_step2 = prior_uncertainty_step2_norm * std
        print(
            "Step 2 observed-entry metrics: "
            f"MAE={masked_mae(pred_step2, true_data, observed_mask):.4f}, "
            f"RMSE={masked_rmse(pred_step2, true_data, observed_mask):.4f}, "
            f"MAPE={masked_mape(pred_step2, true_data, observed_mask):.4f}%, "
            f"mean_prior_uncertainty={masked_mean(prior_uncertainty_step2, observed_mask):.4f}, "
            f"coverage={interval_coverage(pred_step2, prior_uncertainty_step2, true_data, observed_mask):.4f}"
        )
        print(f"Step 2 reverse imputation time: {step2_seconds:.2f}s")

        final_prior_mean = np.where(observed_mask, pred_step2, pred_step1).astype(np.float32)
        if args.clip_nonnegative:
            final_prior_mean = np.maximum(final_prior_mean, 0.0)
        prior_uncertainty = np.where(observed_mask, prior_uncertainty_step2, prior_uncertainty_step1).astype(np.float32)

        all_mask = np.ones_like(observed_mask, dtype=bool)
        print(
            "Final global metrics: "
            f"MAE={masked_mae(final_prior_mean, true_data, all_mask):.4f}, "
            f"RMSE={masked_rmse(final_prior_mean, true_data, all_mask):.4f}, "
            f"MAPE={masked_mape(final_prior_mean, true_data, all_mask):.4f}%"
        )
        print(
            "Final test-missing metrics: "
            f"MAE={masked_mae(final_prior_mean, true_data, test_mask):.4f}, "
            f"RMSE={masked_rmse(final_prior_mean, true_data, test_mask):.4f}, "
            f"MAPE={masked_mape(final_prior_mean, true_data, test_mask):.4f}%, "
            f"mean_prior_uncertainty={masked_mean(prior_uncertainty, test_mask):.4f}, "
            f"coverage={interval_coverage(final_prior_mean, prior_uncertainty, true_data, test_mask):.4f}"
        )

        pre_impute_dir = args.pre_impute_dir
        os.makedirs(pre_impute_dir, exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            pre_impute_dir,
            f"prior_{args.dataset_name}_{args.missing_type}_{args.missing_rate}_{current_time}.npz",
        )
        np.savez_compressed(
            output_path,
            prior_mean=final_prior_mean,
            prior_uncertainty=prior_uncertainty,
            imputed_data=final_prior_mean,
            sigma=prior_uncertainty,
        )
        print(f"Saved low-rank prior file: {output_path}")
        impute_total_seconds = elapsed_since(impute_total_start, device)
        print(
            f"Total checkpoint evaluation + two-step imputation time: "
            f"{impute_total_seconds:.2f}s ({impute_total_seconds / 60.0:.2f} min)"
        )
    print(f"Run finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_handle.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true-path", default=DEFAULT_TRUE_PATH)
    parser.add_argument("--miss-path", default=DEFAULT_MISS_PATH)
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs", "low_rank_prior"))
    parser.add_argument("--log-dir", default=os.path.join(PROJECT_ROOT, "logs", "low_rank_prior"))
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--pre-impute-dir", default=os.path.join(PROJECT_ROOT, "data", "pre_impute"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-name", default="PEMS04")
    parser.add_argument("--missing-type", default="SC-TC")
    parser.add_argument("--missing-rate", default="0.8")
    parser.add_argument("--mode", choices=["train", "impute", "both"], default="both")
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--whiten-prob", type=float, default=0.2)
    parser.add_argument("--f1-loss-weight", type=float, default=0.0, help="Deprecated; Fourier regularization is disabled.")
    parser.add_argument("--prior-interval-loss-weight", dest="prior_interval_loss_weight", type=float, default=0.01)
    parser.add_argument("--mis-loss-weight", dest="prior_interval_loss_weight", type=float, default=None,
                        help="Deprecated alias for --prior-interval-loss-weight.")
    parser.add_argument("--interval-alpha", type=float, default=0.1)
    parser.add_argument("--sigma-min", type=float, default=1e-3)
    parser.add_argument("--input-embedding-dim", type=int, default=32)
    parser.add_argument("--learnable-embedding-dim", type=int, default=96)
    parser.add_argument("--feed-forward-dim", type=int, default=256)
    parser.add_argument("--num-temporal-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dim-proj", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-nonnegative", action="store_true", default=True)
    args = parser.parse_args()
    if args.prior_interval_loss_weight is None:
        args.prior_interval_loss_weight = 0.01
    return args


if __name__ == "__main__":
    run(parse_args())
