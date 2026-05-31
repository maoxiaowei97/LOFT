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

current_time = datetime.datetime.now()


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
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
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
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


def train(
        model,
        config,
        train_loader,
        valid_loader=None,
        test_loader=None,
        _std=None,
        _mean=None,
        valid_epoch_interval=2,
        savename=""
):
    if savename == "":
        current_time_str = time.strftime("%Y_%m_%d_%H_%M_%S")
        savename = f"model_t_{current_time_str}"

    cond_model_save_path = f"./params/{savename}_cond.pth"

    initial_lr = float(config.get("lr", 1e-3))
    final_lr = float(config.get("final_lr", 2e-4))
    high_lr_epochs = int(config.get("high_lr_epochs", 37))

    epochs = int(config["epochs"])
    optimizer_2 = Adam(
        model.velocity_net.parameters(),
        lr=initial_lr,
        weight_decay=1e-5
    )

    if initial_lr > 0:
        gamma = final_lr / initial_lr
    else:
        gamma = 1.0

    milestones = [high_lr_epochs]

    lr_scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_2,
        milestones=milestones,
        gamma=gamma
    )

    logging.info(f"--- Starting Training (Max Epochs: {epochs}) ---")
    for epoch in range(epochs):
        avg_loss = 0
        model.train()
        model.velocity_net.train()

        for batch_no, train_batch in enumerate(train_loader):
            optimizer_2.zero_grad()
            loss = model(train_batch, current_epoch=epoch, total_epochs=epochs)

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(
                    f"Warning: Found NaN/Inf loss at Epoch {epoch}, Batch {batch_no}. Skipping update.")
                continue


            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(model.velocity_net.parameters(), max_norm=1.0)

            if torch.isnan(total_norm) or torch.isinf(total_norm):
                logging.warning(
                    f"Warning: Gradient NaN/Inf (Norm: {total_norm}) at Epoch {epoch}. Skipping step.")
                continue

            torch.nn.utils.clip_grad_norm_(model.velocity_net.parameters(), max_norm=1.0)
            avg_loss += loss.item()
            optimizer_2.step()

        avg_loss /= (batch_no + 1)

        logging.info(
            f"Epoch: {epoch + 1}/{epochs}, Avg Train Loss: {avg_loss:.6f}, LR: {optimizer_2.param_groups[0]['lr']:.6f}")

        lr_scheduler_2.step()

        save_dir = os.path.dirname(cond_model_save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(model.velocity_net.state_dict(), cond_model_save_path)

    logging.info("Training finished.")
    logging.info(f"Loading best conditional model from: {cond_model_save_path}")
    model.velocity_net.load_state_dict(torch.load(cond_model_save_path, map_location=model.device))


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


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


def evaluate(model, test_loader, _std, _mean, use_nni, nsample=10, results_file=None, tensor_save_path=None):
    test_start = time.time()
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        mape_total = 0
        evalpoints_total = 0

        all_generated_samples = []
        all_target = []
        all_evalpoint = []
        all_observed_point = []
        all_observed_time = []
        device = next(model.parameters()).device
        if not isinstance(_std, torch.Tensor):
            scaler = torch.tensor(_std, device=device, dtype=torch.float32)
        else:
            scaler = _std.to(device)

        if not isinstance(_mean, torch.Tensor):
            mean_scaler = torch.tensor(_mean, device=device, dtype=torch.float32)
        else:
            mean_scaler = _mean.to(device)
        logging.info("START TEST...")

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:  # 根据batch-size计算
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1).long()  # (B,L,K)
                observed_points = observed_points.permute(0, 2, 1)
                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (((samples_median.values - c_target) * eval_points) ** 2) * (scaler ** 2)
                mae_current = (torch.abs((samples_median.values - c_target) * eval_points)) * scaler
                mape_current = torch.divide(torch.abs((samples_median.values - c_target) * scaler)
                                            , (c_target * scaler + mean_scaler) * (
                                                        (c_target * scaler + mean_scaler) > (1e-4))) \
                                   .nan_to_num(posinf=0, neginf=0, nan=0) * eval_points

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mape_total += mape_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "mape_total": mape_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
                logging.info("rmse_total={}".format(np.sqrt(mse_total / evalpoints_total)))
                logging.info("mae_total={}".format(mae_total / evalpoints_total))
                logging.info("mape_total={}".format(mape_total / evalpoints_total))
                logging.info("batch_no={}".format(batch_no))

        final_rmse = np.sqrt(mse_total / evalpoints_total)
        final_mae = mae_total / evalpoints_total
        final_mape = mape_total / evalpoints_total
        final_target = torch.cat(all_target, dim=0)
        final_samples = torch.cat(all_generated_samples, dim=0)
        final_evalpoint = torch.cat(all_evalpoint, dim=0)
        final_crps = calc_quantile_CRPS(
            final_target, final_samples, final_evalpoint, mean_scaler, scaler)

        final_target = torch.cat(all_target, dim=0)
        final_samples = torch.cat(all_generated_samples, dim=0)
        final_evalpoint = torch.cat(all_evalpoint, dim=0)
        final_observed_point = torch.cat(all_observed_point, dim=0)
        final_observed_time = torch.cat(all_observed_time, dim=0)

        logging.info(f"RMSE: {final_rmse}")
        logging.info(f"MAE: {final_mae}")
        logging.info(f"MAPE: {final_mape}")
        logging.info(f"CRPS: {final_crps:.4f}")
        PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

        RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR, exist_ok=True)

        if tensor_save_path:
            output_tensors_file = tensor_save_path
        else:
            output_tensors_file = os.path.join(RESULTS_DIR, 'evaluation_tensors.pth')
            
        if output_tensors_file:
            logging.info(f"Saving output tensors to {output_tensors_file}...")
            torch.save({
                'samples': final_samples,
                'target': final_target,
                'eval_points': final_evalpoint,
                'observed_points': final_observed_point,
                'observed_time': final_observed_time
            }, output_tensors_file)
            logging.info("Tensors saved successfully.")

        if results_file:

            miss_rate = model.config['train']['miss_rate']

            file_exists = os.path.isfile(results_file)

            with open(results_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow([
                        'miss_rate',  'rmse', 'mae', 'mape', 'crps'
                    ])

                writer.writerow([
                    miss_rate,
                    f"{final_rmse:.4f}", f"{final_mae:.4f}", f"{final_mape:.4f}", f"{final_crps:.4f}"
                ])
            logging.info(f"results saved to: {results_file}")
    test_end_time = time.time()
    logging.info(f"Testing time: {test_end_time - test_start}")
