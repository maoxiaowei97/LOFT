import numpy as np
import torch
from torch.optim import Adam
import time
from tqdm import tqdm
import logging
import datetime
import csv
import os
import random
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_time = datetime.datetime.now()
_PLOT_IMPORT_WARNING_EMITTED = False


def get_seaborn():
    global _PLOT_IMPORT_WARNING_EMITTED
    try:
        import seaborn as sns
        return sns
    except ImportError:
        if not _PLOT_IMPORT_WARNING_EMITTED:
            logging.warning("seaborn is not installed; diagnostic boxplots will be skipped.")
            _PLOT_IMPORT_WARNING_EMITTED = True
        return None


def get_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        logging.warning("pandas is not installed; diagnostic boxplots will be skipped.")
        return None


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def quick_evaluate(model, test_loader, _std, _mean, nsample=10, limit_batches=None, epoch=None, savename="", output_root="."):
    model.eval()
    mse_total, mae_total, mape_total, evalpoints_total = 0, 0, 0, 0
    device = next(model.parameters()).device
    scaler = torch.tensor(_std, device=device, dtype=torch.float32) if not isinstance(_std, torch.Tensor) else _std.to(device)
    mean_scaler = torch.tensor(_mean, device=device, dtype=torch.float32) if not isinstance(_mean,   torch.Tensor) else _mean.to(device)

    all_vmr = []
    all_epe = []
    all_tgt_cos = []
    all_sigma_raw = []
    all_sigma_sample = []

    with torch.no_grad():
        for batch_no, test_batch in enumerate(test_loader, start=1):
            if limit_batches is not None and batch_no > limit_batches: break

            output = model.evaluate(test_batch, nsample)

            if len(output) == 6:
                samples, c_target, eval_points, _, _, inf_metrics = output
                all_vmr.append(inf_metrics['vmr'])
                all_epe.append(inf_metrics['epe'])
                all_tgt_cos.append(inf_metrics['tgt_cos'])
                all_sigma_raw.append(inf_metrics['sigma_raw'])
                all_sigma_sample.append(inf_metrics['sigma_sample'])
            else:
                samples, c_target, eval_points, _, _ = output

            samples = samples.permute(0, 1, 3, 2)
            c_target = c_target.permute(0, 2, 1)
            eval_points = eval_points.permute(0, 2, 1).long()
            samples_median = samples.median(dim=1).values
            mse_current = (((samples_median - c_target) * eval_points) ** 2) * (scaler ** 2)
            mae_current = (torch.abs((samples_median - c_target) * eval_points)) * scaler
            mape_current = torch.divide(
                torch.abs((samples_median - c_target) * scaler),
                (c_target * scaler + mean_scaler) * ((c_target * scaler + mean_scaler) > 1e-4)
            ).nan_to_num(posinf=0, neginf=0, nan=0) * eval_points
            mse_total += mse_current.sum().item()
            mae_total += mae_current.sum().item()
            mape_total += mape_current.sum().item()
            evalpoints_total += eval_points.sum().item()

    rmse = np.sqrt(mse_total / evalpoints_total) if evalpoints_total > 0 else 0
    mae = mae_total / evalpoints_total if evalpoints_total > 0 else 0
    mape = mape_total / evalpoints_total if evalpoints_total > 0 else 0

    model.train()
    if hasattr(model, 'velocity_net'): model.velocity_net.train()

    if all_vmr:
        final_vmr = np.concatenate(all_vmr)
        final_epe = np.concatenate(all_epe)
        final_tgt_cos = np.concatenate(all_tgt_cos)
        final_sigma_raw = np.concatenate(all_sigma_raw)
        final_sigma_sample = np.concatenate(all_sigma_sample)


        if epoch is not None and savename:
            save_dir = os.path.join(output_root, "results", "quick_eval_metrics")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{savename}_ep{epoch}.npz")
            np.savez(save_path, vmr=final_vmr, epe=final_epe, tgt_cos=final_tgt_cos,
                     sigma_raw=final_sigma_raw, sigma_sample=final_sigma_sample)
            logging.info(f"Saved Baseline complete inference metrics and raw sigma to {save_path}")

        return rmse, mae, mape, final_vmr, final_epe, final_tgt_cos, final_sigma_sample

    return rmse, mae, mape


def draw_conflict_boxplot(cos_array, sigma_array, epoch, savename, output_root="."):
    if len(cos_array) == 0 or len(sigma_array) == 0: return
    sns = get_seaborn()
    pd = get_pandas()
    if sns is None or pd is None:
        return

    q75 = np.percentile(sigma_array, 75)
    high_mask = sigma_array > q75
    labels = np.where(high_mask, r"$\bar{\sigma}_{soft} > Q_{0.75}$", r"$\bar{\sigma}_{soft} \leq Q_{0.75}$")

    df = pd.DataFrame({"TargetCos": cos_array, "Group": labels})
    order = [r"$\bar{\sigma}_{soft} > Q_{0.75}$", r"$\bar{\sigma}_{soft} \leq Q_{0.75}$"]
    palette = {r"$\bar{\sigma}_{soft} > Q_{0.75}$": "#c44e52", r"$\bar{\sigma}_{soft} \leq Q_{0.75}$": "#4c72b0"}

    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "axes.edgecolor": "lightgray"})

    ax = sns.boxplot(x="Group", y="TargetCos", hue="Group", data=df, order=order, palette=palette,
                     width=0.4, showfliers=False,
                     boxprops=dict(edgecolor="#333333", linewidth=1.5, alpha=0.9),
                     whiskerprops=dict(color="#333333", linewidth=1.5),
                     capprops=dict(color="#333333", linewidth=1.5), medianprops=dict(color="#333333", linewidth=2.0))

    df_sample = df.groupby("Group", group_keys=False).apply(lambda x: x.sample(min(len(x), 1000))) if len(
        df) > 2000 else df
    sns.stripplot(x="Group", y="TargetCos", data=df_sample, order=order, color="black", alpha=0.5, jitter=0.25,
                  size=3.5, ax=ax)

    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.axhline(1.0, color="green", linestyle=":", linewidth=2, alpha=0.7)
    plt.ylabel(r"Intrinsic Alignment $\cos(\mathbf{v}_{target}, \mathbf{v}_{teacher})$", fontsize=15, fontweight="bold")
    plt.xlabel("")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    plt.ylim(-1.05, 1.05)
    plt.title(f"Baseline Intrinsic Task Alignment (Epoch {epoch})", fontsize=14, fontweight="bold", pad=15)

    if ax.legend_ is not None:
        ax.legend_.remove()

    plt.tight_layout()
    save_dir = os.path.join(output_root, "results", "conflict_boxplots_sample")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{savename}_epoch_{epoch:03d}.png"), dpi=300)
    plt.close()


def draw_inference_boxplots(vmr, epe, tgt_cos, sigma_array, epoch, savename, output_root="."):
    if len(vmr) == 0 or len(sigma_array) == 0: return
    sns = get_seaborn()
    pd = get_pandas()
    if sns is None or pd is None:
        return

    q75 = np.percentile(sigma_array, 75)
    high_mask = sigma_array > q75
    labels = np.where(high_mask, r"$\bar{\sigma}_{soft} > Q_{0.75}$", r"$\bar{\sigma}_{soft} \leq Q_{0.75}$")
    order = [r"$\bar{\sigma}_{soft} > Q_{0.75}$", r"$\bar{\sigma}_{soft} \leq Q_{0.75}$"]
    palette = {r"$\bar{\sigma}_{soft} > Q_{0.75}$": "#c44e52", r"$\bar{\sigma}_{soft} \leq Q_{0.75}$": "#4c72b0"}


    df_vmr = pd.DataFrame({"VMR": vmr, "Group": labels})
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "axes.edgecolor": "lightgray"})

    ax1 = sns.boxplot(x="Group", y="VMR", hue="Group", data=df_vmr, order=order, palette=palette,
                      width=0.4, showfliers=False,
                      boxprops=dict(edgecolor="#333333", linewidth=1.5, alpha=0.9),
                      whiskerprops=dict(color="#333333", linewidth=1.5),
                      capprops=dict(color="#333333", linewidth=1.5), medianprops=dict(color="#333333", linewidth=2.0))
    df_sample_vmr = df_vmr.groupby("Group", group_keys=False).apply(lambda x: x.sample(min(len(x), 1000))) if len(
        df_vmr) > 2000 else df_vmr
    sns.stripplot(x="Group", y="VMR", data=df_sample_vmr, order=order, color="black", alpha=0.5, jitter=0.25,
                  size=3.5, ax=ax1)

    plt.axhline(1.0, color="green", linestyle="--", linewidth=2, label="Ideal Confidence")
    plt.ylabel(r"Velocity Magnitude Ratio $\|\mathbf{v}_{pred}\| / \|\mathbf{v}_{ideal}\|$", fontsize=15,
               fontweight="bold")
    plt.xlabel("")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    plt.ylim(0, max(2.0, np.percentile(vmr, 95) * 1.1))
    plt.title(f"Baseline Inference Vector Field Confidence (Epoch {epoch})", fontsize=14, fontweight="bold", pad=15)

    handles, labels_leg = ax1.get_legend_handles_labels()
    idx = [i for i, label in enumerate(labels_leg) if label == "Ideal Confidence"]
    if idx:
        plt.legend([handles[i] for i in idx], [labels_leg[i] for i in idx], loc="upper right")
    else:
        if ax1.legend_ is not None:
            ax1.legend_.remove()

    plt.tight_layout()
    save_dir = os.path.join(output_root, "results", "inference_boxplots_vmr")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{savename}_vmr_epoch_{epoch:03d}.png"), dpi=300)
    plt.close()


    df_epe = pd.DataFrame({"EPE": epe, "Group": labels})
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "axes.edgecolor": "lightgray"})

    ax2 = sns.boxplot(x="Group", y="EPE", hue="Group", data=df_epe, order=order, palette=palette,
                      width=0.4, showfliers=False,
                      boxprops=dict(edgecolor="#333333", linewidth=1.5, alpha=0.9),
                      whiskerprops=dict(color="#333333", linewidth=1.5),
                      capprops=dict(color="#333333", linewidth=1.5), medianprops=dict(color="#333333", linewidth=2.0))
    df_sample_epe = df_epe.groupby("Group", group_keys=False).apply(lambda x: x.sample(min(len(x), 1000))) if len(
        df_epe) > 2000 else df_epe
    sns.stripplot(x="Group", y="EPE", data=df_sample_epe, order=order, color="black", alpha=0.5, jitter=0.25,
                  size=3.5, ax=ax2)

    plt.ylabel(r"Endpoint Projection Error (MAE)", fontsize=15, fontweight="bold")
    plt.xlabel("")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    plt.ylim(bottom=0)
    plt.title(f"Baseline Inference Implicit Endpoint Deviation (Epoch {epoch})", fontsize=14, fontweight="bold", pad=15)

    if ax2.legend_ is not None:
        ax2.legend_.remove()

    plt.tight_layout()
    save_dir = os.path.join(output_root, "results", "inference_boxplots_epe")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{savename}_epe_epoch_{epoch:03d}.png"), dpi=300)
    plt.close()


    df_tgt = pd.DataFrame({"TgtCos": tgt_cos, "Group": labels})
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "axes.edgecolor": "lightgray"})

    ax3 = sns.boxplot(x="Group", y="TgtCos", hue="Group", data=df_tgt, order=order, palette=palette,
                      width=0.4, showfliers=False,
                      boxprops=dict(edgecolor="#333333", linewidth=1.5, alpha=0.9),
                      whiskerprops=dict(color="#333333", linewidth=1.5),
                      capprops=dict(color="#333333", linewidth=1.5), medianprops=dict(color="#333333", linewidth=2.0))
    df_sample_tgt = df_tgt.groupby("Group", group_keys=False).apply(lambda x: x.sample(min(len(x), 1000))) if len(
        df_tgt) > 2000 else df_tgt
    sns.stripplot(x="Group", y="TgtCos", data=df_sample_tgt, order=order, color="black", alpha=0.5, jitter=0.25,
                  size=3.5, ax=ax3)

    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.axhline(1.0, color="green", linestyle=":", linewidth=2, alpha=0.7)
    plt.ylabel(r"Inference Target Alignment $\cos(\mathbf{v}^{(n)}, \mathbf{v}_{target})$", fontsize=15,
               fontweight="bold")
    plt.xlabel("")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    plt.ylim(-1.05, 1.05)
    plt.title(f"Baseline Inference Target Pointing Accuracy (Epoch {epoch})", fontsize=14, fontweight="bold", pad=15)

    if ax3.legend_ is not None:
        ax3.legend_.remove()

    plt.tight_layout()
    save_dir = os.path.join(output_root, "results", "inference_boxplots_tgt")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{savename}_tgt_epoch_{epoch:03d}.png"), dpi=300)
    plt.close()


def train(
        model, config, train_loader, valid_loader=None, test_loader=None,
        _std=None, _mean=None, valid_epoch_interval=2, savename="", output_root="."
):
    if savename == "":
        current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        savename = f"model_t_{current_time_str}"

    cond_model_save_path = os.path.join(output_root, "params", f"{savename}_cond.pth")

    dynamics_dir = os.path.join(output_root, "results", "dynamics_log")
    os.makedirs(dynamics_dir, exist_ok=True)
    batch_metrics_csv = os.path.join(dynamics_dir, f"{savename}_batch_metrics.csv")

    with open(batch_metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Batch", "Global_Step", "Loss_Tot", "Grad_Cos", "LR"])

    initial_lr = float(config.get("initial_lr", config.get("lr", 1e-3)))
    final_lr = float(config.get("final_lr", 2e-4))
    rectification_lr = float(config.get("rectification_lr", 5e-5))
    high_lr_epochs = int(config.get("high_lr_epochs", 37))
    warmup_epochs = int(config.get("warmup_epochs", 0))
    rectification_epochs = int(config["epochs"])
    epoch_offset = int(config.get("epoch_offset", 0))
    epochs = rectification_epochs if epoch_offset > 0 else warmup_epochs + rectification_epochs
    schedule_total_epochs = warmup_epochs + rectification_epochs
    config["total_epochs"] = str(epochs)
    if hasattr(model, "warmup_epochs"):
        model.warmup_epochs = warmup_epochs

    optimizer_2 = Adam(model.velocity_net.parameters(), lr=initial_lr, weight_decay=1e-5)
    gamma = final_lr / initial_lr if initial_lr > 0 else 1.0
    lr_scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, milestones=[high_lr_epochs], gamma=gamma)
    logging.info(
        f"Optimizer LR schedule: initial_lr={initial_lr:g}, final_lr={final_lr:g}, "
        f"high_lr_epochs={high_lr_epochs}, gamma={gamma:g}, rectification_lr={rectification_lr:g}"
    )

    inf_steps = model.config["flow_matching"].get("inference_steps", 10)
    global_step = 0
    log_interval = 10

    logging.info(
        f"--- Starting LOFT Training (Warm-up: {warmup_epochs}, Rectification: {rectification_epochs}, "
        f"Run Epochs: {epochs}, Epoch Offset: {epoch_offset}, Schedule Total: {schedule_total_epochs}) ---"
    )
    for epoch in range(epochs):
        schedule_epoch = epoch_offset + epoch
        display_epoch = schedule_epoch + 1
        if schedule_epoch >= warmup_epochs:
            for group in optimizer_2.param_groups:
                group["lr"] = rectification_lr
        avg_loss = 0
        epoch_metrics = {"loss_fm": 0.0, "loss_cons": 0.0, "cos_tgt": 0.0, "cos_tch": 0.0, "grad_cos": 0.0}

        epoch_batch_records = []
        epoch_target_cos_sample = []
        epoch_sigma_sample = []

        model.train()
        if hasattr(model, 'velocity_net'): model.velocity_net.train()

        for batch_no, train_batch in enumerate(train_loader):
            optimizer_2.zero_grad()
            loss, metrics = model(train_batch, current_epoch=schedule_epoch, total_epochs=schedule_total_epochs)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.velocity_net.parameters(), max_norm=1.0)
            optimizer_2.step()

            global_step += 1
            cur_lr = optimizer_2.param_groups[0]['lr']
            cur_grad_cos = metrics.get('grad_cos', 0.0)

            if "target_cos_sample" in metrics and "sigma_sample" in metrics:
                valid_mask = metrics["sigma_sample"] > 0
                if valid_mask.sum() > 0:
                    epoch_target_cos_sample.append(metrics["target_cos_sample"][valid_mask])
                    epoch_sigma_sample.append(metrics["sigma_sample"][valid_mask])

            epoch_batch_records.append([display_epoch, batch_no + 1, global_step, loss.item(), cur_grad_cos, cur_lr])

            detailed_batch_logging = schedule_epoch >= warmup_epochs
            if detailed_batch_logging and (batch_no + 1) % log_interval == 0:
                logging.info(
                    f"  ↳ [Ep:{display_epoch:03d} | B:{batch_no + 1:03d}] Tot:{loss.item():.4f} | GradCos:{cur_grad_cos: 6.4f}")

            avg_loss += loss.item()
            for k in epoch_metrics: epoch_metrics[k] += metrics.get(k, 0.0)

        if epoch_batch_records:
            with open(batch_metrics_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(epoch_batch_records)

        if epoch_target_cos_sample and epoch_sigma_sample:
            all_target_cos_sample = np.concatenate(epoch_target_cos_sample)
            all_sigma_sample = np.concatenate(epoch_sigma_sample)
            if len(all_target_cos_sample) > 0:
                draw_conflict_boxplot(all_target_cos_sample, all_sigma_sample, display_epoch, savename, output_root=output_root)

        num_batches = batch_no + 1
        avg_loss /= num_batches
        for k in epoch_metrics: epoch_metrics[k] /= num_batches

        logging.info(
            f"Ep:{display_epoch:03d}/{schedule_total_epochs} | Tot:{avg_loss:.4f} | FM:{epoch_metrics['loss_fm']:6.4f} | Cons:{epoch_metrics['loss_cons']:6.4f} | GradCos:{epoch_metrics['grad_cos']: 6.4f} | LR:{optimizer_2.param_groups[0]['lr']:.1e}")

        if display_epoch <= warmup_epochs:
            should_quick_eval = test_loader is not None and display_epoch == warmup_epochs
        else:
            should_quick_eval = test_loader is not None and display_epoch % 3 == 0
        if should_quick_eval:
            quick_out = quick_evaluate(model, test_loader, _std, _mean, nsample=10, limit_batches=None, epoch=display_epoch,
                                       savename=savename, output_root=output_root)

            if len(quick_out) == 7:
                q_rmse, q_mae, q_mape, q_vmr, q_epe, q_tgt, q_sig = quick_out
                draw_inference_boxplots(q_vmr, q_epe, q_tgt, q_sig, display_epoch, savename, output_root=output_root)
                logging.info(
                    f"  └── 🚀 [Quick Eval @ Epoch {display_epoch}] -> RMSE: {q_rmse:.4f} |  MAPE: {q_mape:.4f} |  MAE: {q_mae:.4f} | VMR (幅度保持): {q_vmr.mean():.3f} | EPE (终点误差): {q_epe.mean():.3f} | TgtCos: {q_tgt.mean():.3f}")
            else:
                q_rmse, q_mae, q_mape = quick_out
                logging.info(
                    f"  └── 🚀 [Quick Eval @ Epoch {display_epoch}] -> RMSE: {q_rmse:.4f} | MAE: {q_mae:.4f} | MAPE: {q_mape:.4f}")

            save_dir = os.path.dirname(cond_model_save_path)
            if not os.path.exists(save_dir): os.makedirs(save_dir)

        if schedule_epoch + 1 < warmup_epochs:
            lr_scheduler_2.step()

    logging.info("Training finished.")
    save_dir = os.path.dirname(cond_model_save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.velocity_net.state_dict(), cond_model_save_path)
    logging.info(f"Final LOFT weight saved to {cond_model_save_path}")


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q)))


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for q in quantiles:
        q_pred = torch.quantile(forecast, q, dim=1)
        q_loss = quantile_loss(target, q_pred, q, eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(
        model, test_loader, _std, _mean, use_nni, nsample=10, results_file=None,
        tensor_save_path=None, trace_file=None, trace_batches=1, trace_samples=1
):

    with torch.no_grad():
        model.eval()
        mse_total, mae_total, mape_total = 0, 0, 0
        prior_mse_total, prior_mae_total, prior_mape_total = 0, 0, 0
        evalpoints_total = 0
        all_generated_samples, all_target, all_evalpoint = [], [], []
        all_observed_point, all_observed_time = [], []
        all_traces = []

        device = next(model.parameters()).device
        scaler = torch.tensor(_std, device=device, dtype=torch.float32) if not isinstance(_std,
                                                                                          torch.Tensor) else _std.to(
            device)
        mean_scaler = torch.tensor(_mean, device=device, dtype=torch.float32) if not isinstance(_mean,
                                                                                                torch.Tensor) else _mean.to(
            device)

        logging.info("START TEST...")
        test_start = time.time()
        total_batches = len(test_loader) if hasattr(test_loader, "__len__") else None
        total_batches_text = str(total_batches) if total_batches is not None else "?"
        eval_batch_size = getattr(test_loader, "batch_size", None)
        logging.info(f"Evaluation batches: {total_batches_text}")
        logging.info(f"Evaluation batch_size: {eval_batch_size}, nsample: {nsample}")
        if trace_file is not None:
            trace_scope = "all batches" if trace_batches <= 0 else f"{trace_batches} batch(es)"
            logging.info(f"Trace saving enabled: {trace_scope}, trace_samples={trace_samples}, path={trace_file}")
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0, file=sys.stdout) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                collect_trace = trace_file is not None and (trace_batches <= 0 or batch_no <= trace_batches)
                output = model.evaluate(
                    test_batch, nsample,
                    collect_trace=collect_trace,
                    trace_samples=trace_samples,
                )

                if len(output) == 6:
                    samples, c_target, eval_points, observed_points, observed_time, _ = output
                    inf_metrics = output[-1]
                else:
                    samples, c_target, eval_points, observed_points, observed_time = output
                    inf_metrics = None

                if inf_metrics is not None and "trace" in inf_metrics:
                    all_traces.append(inf_metrics["trace"])

                samples = samples.permute(0, 1, 3, 2)
                c_target = c_target.permute(0, 2, 1)
                eval_points = eval_points.permute(0, 2, 1).long()
                observed_points = observed_points.permute(0, 2, 1)


                samples_median = samples.median(dim=1).values
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (((samples_median - c_target) * eval_points) ** 2) * (scaler ** 2)
                mae_current = (torch.abs((samples_median - c_target) * eval_points)) * scaler
                mape_current = torch.divide(torch.abs((samples_median - c_target) * scaler)
                                            , (c_target * scaler + mean_scaler) * (
                                                    (c_target * scaler + mean_scaler) > (1e-4))) \
                                   .nan_to_num(posinf=0, neginf=0, nan=0) * eval_points

                prior_data = test_batch["prior_mean"].to(device).float()
                prior_mse_current = (((prior_data - c_target) * eval_points) ** 2) * (scaler ** 2)
                prior_mae_current = (torch.abs((prior_data - c_target) * eval_points)) * scaler
                prior_mape_current = torch.divide(torch.abs((prior_data - c_target) * scaler)
                                                  , (c_target * scaler + mean_scaler) * (
                                                          (c_target * scaler + mean_scaler) > (1e-4))) \
                                         .nan_to_num(posinf=0, neginf=0, nan=0) * eval_points

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mape_total += mape_current.sum().item()
                prior_mse_total += prior_mse_current.sum().item()
                prior_mae_total += prior_mae_current.sum().item()
                prior_mape_total += prior_mape_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                current_crps = calc_quantile_CRPS(torch.cat(all_target, dim=0), torch.cat(all_generated_samples, dim=0),
                                                  torch.cat(all_evalpoint, dim=0), mean_scaler, scaler)

                running_rmse = np.sqrt(mse_total / evalpoints_total)
                running_mae = mae_total / evalpoints_total
                running_mape = mape_total / evalpoints_total
                running_prior_rmse = np.sqrt(prior_mse_total / evalpoints_total)
                running_prior_mae = prior_mae_total / evalpoints_total
                elapsed = time.time() - test_start

                it.set_postfix(
                    ordered_dict={"rmse": running_rmse, "mae": running_mae},
                    refresh=True)
                logging.info(
                    f"[Eval Batch {batch_no:03d}/{total_batches_text}] "
                    f"LOFT RMSE:{running_rmse:.4f} | MAE:{running_mae:.4f} | MAPE:{running_mape:.4f} || "
                    f"Prior RMSE:{running_prior_rmse:.4f} | MAE:{running_prior_mae:.4f} | "
                    f"elapsed:{elapsed:.1f}s"
                )
        test_end_time = time.time()
        logging.info(f"Testing time: {test_end_time - test_start}")
        final_rmse = np.sqrt(mse_total / evalpoints_total)
        final_mae = mae_total / evalpoints_total
        final_mape = mape_total / evalpoints_total
        prior_rmse = np.sqrt(prior_mse_total / evalpoints_total)
        prior_mae = prior_mae_total / evalpoints_total
        prior_mape = prior_mape_total / evalpoints_total
        final_target = torch.cat(all_target, dim=0)
        final_samples = torch.cat(all_generated_samples, dim=0)
        final_evalpoint = torch.cat(all_evalpoint, dim=0)
        final_crps = calc_quantile_CRPS(final_target, final_samples, final_evalpoint, mean_scaler, scaler)

        final_observed_point = torch.cat(all_observed_point, dim=0)
        final_observed_time = torch.cat(all_observed_time, dim=0)

        logging.info(f"Prior RMSE: {prior_rmse:.4f} | LOFT RMSE: {final_rmse:.4f}")
        logging.info(f"Prior MAE: {prior_mae:.4f} | LOFT MAE: {final_mae:.4f}")
        logging.info(f"Prior MAPE: {prior_mape:.4f} | LOFT MAPE: {final_mape:.4f}")
        logging.info(f"LOFT CRPS: {final_crps:.4f}")

        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
        RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
        os.makedirs(RESULTS_DIR, exist_ok=True)

        if tensor_save_path:
            os.makedirs(os.path.dirname(tensor_save_path), exist_ok=True)
            torch.save({
                'samples': final_samples, 'target': final_target,
                'eval_points': final_evalpoint, 'observed_points': final_observed_point,
                'observed_time': final_observed_time
            }, tensor_save_path)
            logging.info(f"Tensors saved to {tensor_save_path}")

        if trace_file and all_traces:
            os.makedirs(os.path.dirname(trace_file), exist_ok=True)
            trace_out = {"timesteps": all_traces[0]["timesteps"]}
            for key in [
                "trajectory", "velocity", "cond_mask", "target_mask",
                "observed_data", "target_data", "prior_mean",
            ]:
                trace_out[key] = torch.cat([trace[key] for trace in all_traces], dim=0)
            torch.save(trace_out, trace_file)
            logging.info(f"Trace tensors saved to {trace_file}")

        if results_file:
            miss_rate = model.config['train']['miss_rate']
            file_exists = os.path.isfile(results_file)
            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow(
                        ['miss_rate', 'prior_rmse', 'prior_mae', 'prior_mape', 'rmse', 'mae', 'mape', 'crps'])
                writer.writerow([miss_rate, f"{prior_rmse:.4f}", f"{prior_mae:.4f}", f"{prior_mape:.4f}",
                                 f"{final_rmse:.4f}", f"{final_mae:.4f}", f"{final_mape:.4f}", f"{final_crps:.4f}"])
            logging.info(f"results saved to: {results_file}")
