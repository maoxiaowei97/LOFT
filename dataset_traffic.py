import numpy as np
from torch.utils.data import DataLoader, Dataset


class TrafficDataset(Dataset):
    def __init__(
        self,
        observed_data,
        true_data,
        observed_masks,
        gt_mask,
        time_of_day_encoding,
        prior_uncertainty,
        isTest=False,
        eval_length=12,
    ):
        self.eval_length = eval_length
        self.observed_masks = observed_masks
        self.gt_masks = gt_mask
        self.observed_data = observed_data
        self.time_of_day_encoding = time_of_day_encoding
        self.prior_mean = observed_data
        self.true_data = true_data
        self.prior_uncertainty = prior_uncertainty

        if isTest:
            self.observed_data = true_data
            self.observed_masks = np.ones_like(self.gt_masks)

    def __getitem__(self, index):
        return {
            "observed_data": self.observed_data[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            "time_of_day": self.time_of_day_encoding[index],
            "prior_mean": self.prior_mean[index],
            "true_data": self.true_data[index],
            "prior_uncertainty": self.prior_uncertainty[index],
        }

    def __len__(self):
        return len(self.observed_data)

def construct_time_of_day_encoding(c_data, timesteps_per_day=288):

    N_TIMESTEPS, N_NODES = c_data.shape

    time_indices = np.arange(N_TIMESTEPS)

    tod_indices = time_indices % timesteps_per_day

    normalized_tod = tod_indices / timesteps_per_day

    normalized_tod_col = normalized_tod.reshape(-1, 1)

    tod_encoding = np.tile(normalized_tod_col, (1, N_NODES))

    return tod_encoding


def make_overlapped_sliding_windows(prior_mean, true_values, mask, gt_mask, time_of_day, prior_uncertainty, sample_len):
    prior_window, true_window, mask_window, gt_mask_window, tod_window, uncertainty_window = [], [], [], [], [], []
    for i in range(prior_mean.shape[0] - sample_len + 1):
        prior_window.append(prior_mean[i:i + sample_len])
        true_window.append(true_values[i:i + sample_len])
        mask_window.append(mask[i:i + sample_len])
        gt_mask_window.append(gt_mask[i:i + sample_len])
        tod_window.append(time_of_day[i:i + sample_len])
        uncertainty_window.append(prior_uncertainty[i:i + sample_len])

    prior_window = np.array(prior_window)
    true_window = np.array(true_window)
    mask_window = np.array(mask_window)
    gt_mask_window = np.array(gt_mask_window)
    tod_window = np.array(tod_window)
    uncertainty_window = np.array(uncertainty_window)

    return prior_window, true_window, mask_window, gt_mask_window, tod_window, uncertainty_window


def get_dataloader(
    true_datapath,
    miss_datapath,
    low_rank_prior_path,
    val_ratio,
    test_ratio,
    batch_size=16,
    eval_length=12,
    node_start=0,
    node_end=30,
):
    miss = np.load(miss_datapath)
    true = np.load(true_datapath)


    observed_masks = np.nan_to_num(true['mask'][:, :, 0])[:, :]
    gt_masks = np.nan_to_num(miss['mask'][:, :, 0])[:, :]
    true_data = np.nan_to_num(true['data'][:, :, 0].astype(np.float32))[:, :]

    T, N = true_data.shape
    print(f"Using node range [{node_start}, {node_end}) -> N={N}")

    val_len = int(T * val_ratio)
    test_len = int(T * test_ratio)
    train_len = T - val_len - test_len

    train_observed_values = true_data[:train_len][observed_masks[:train_len] == 1]
    mean = np.mean(train_observed_values)
    std = np.std(train_observed_values)
    true_data = (true_data - mean) / std

    print(f"Loading Low-Rank Prior from: {low_rank_prior_path}")
    prior_npz = np.load(low_rank_prior_path)

    if 'prior_mean' in prior_npz:
        prior_mean = prior_npz['prior_mean']
    elif 'data_impute' in prior_npz:
        prior_mean = prior_npz['data_impute']
    elif 'imputed_data' in prior_npz:
        prior_mean = prior_npz['imputed_data']
    else:
        print(f"Warning: prior mean not found in {low_rank_prior_path}. Using the first key: {prior_npz.files[0]}")
        prior_mean = prior_npz[prior_npz.files[0]]

    if 'prior_uncertainty' in prior_npz:
        prior_uncertainty = prior_npz['prior_uncertainty']
        print("Successfully loaded 'prior_uncertainty' from low-rank prior file.")
    elif 'prior_std' in prior_npz:
        prior_uncertainty = prior_npz['prior_std']
        print("Successfully loaded 'prior_std' from low-rank prior file.")
    elif 'sigma' in prior_npz:
        prior_uncertainty = prior_npz['sigma']
        print("Successfully loaded legacy 'sigma' as prior uncertainty.")
    else:
        print("Warning: prior uncertainty not found in low-rank prior file. Using zeros.")
        prior_uncertainty = np.zeros_like(prior_mean)

    prior_mean = prior_mean[:, :]
    prior_uncertainty = prior_uncertainty[:, :]

    c_data = (prior_mean - mean) / std

    prior_uncertainty_norm = prior_uncertainty / std
    print(f"Prior Uncertainty (Norm) -> Mean: {prior_uncertainty_norm.mean():.4f}, Min: {prior_uncertainty_norm.min():.4f}, Max: {prior_uncertainty_norm.max():.4f}")

    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    norm_quantiles = np.quantile(prior_uncertainty_norm, qs)

    orig_quantiles = np.quantile(prior_uncertainty, qs)

    print(f"{'Quantile':>10} | {'Norm Value':>12} | {'Original Value':>15}")
    print("-" * 45)

    for q, n_val, o_val in zip(qs, norm_quantiles, orig_quantiles):
        print(f"{q:>10.1%} | {n_val:>12.4f} | {o_val:>15.4f}")

    time_of_day_encoding = construct_time_of_day_encoding(c_data)

    train_X, val_X, test_X = c_data[:-(val_len + test_len)], \
        c_data[-(val_len + test_len):-(test_len)], \
        c_data[-test_len:]

    train_Y, val_Y, test_Y = true_data[:-(val_len + test_len)], \
        true_data[-(val_len + test_len):-(test_len)], \
        true_data[-test_len:]

    train_mask, val_mask, test_mask = observed_masks[:-(val_len + test_len)], \
        observed_masks[-(val_len + test_len):-(test_len)], \
        observed_masks[-test_len:]

    train_gtmask, val_gtmask, test_gtmask = gt_masks[:-(val_len + test_len)], \
        gt_masks[-(val_len + test_len):-(test_len)], \
        gt_masks[-test_len:]

    train_time_of_day, val_time_of_day, test_time_of_day = time_of_day_encoding[:-(val_len + test_len)], \
        time_of_day_encoding[-(val_len + test_len):-(test_len)], \
        time_of_day_encoding[-test_len:]


    train_uncertainty, val_uncertainty, test_uncertainty = prior_uncertainty_norm[:-(val_len + test_len)], \
        prior_uncertainty_norm[-(val_len + test_len):-(test_len)], \
        prior_uncertainty_norm[-test_len:]

    train_X, train_Y, train_mask, train_gtmask, train_time_of_day_imp, train_uncertainty_win = make_overlapped_sliding_windows(train_X, train_Y,
                                                                                                       train_mask,
                                                                                                       train_gtmask,
                                                                                                       train_time_of_day,
                                                                                                       train_uncertainty,
                                                                                                       eval_length)
    val_X, val_Y, val_mask, val_gtmask, val_time_of_day_imp, val_uncertainty_win = make_overlapped_sliding_windows(val_X, val_Y, val_mask,
                                                                                             val_gtmask, val_time_of_day,
                                                                                             val_uncertainty,
                                                                                             eval_length)
    test_X, test_Y, test_mask, test_gtmask, test_time_of_day_imp, test_uncertainty_win = make_overlapped_sliding_windows(test_X, test_Y,
                                                                                                  test_mask,
                                                                                                  test_gtmask, test_time_of_day,
                                                                                                  test_uncertainty,
                                                                                                  eval_length)

    dataset = TrafficDataset(
        train_X, train_Y, train_mask, train_gtmask, train_time_of_day_imp, train_uncertainty_win, isTest=False, eval_length=12
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = TrafficDataset(
        val_X, val_Y, val_mask, val_gtmask, val_time_of_day_imp, val_uncertainty_win, isTest=False, eval_length=12
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = TrafficDataset(
        test_X, test_Y, test_mask, test_gtmask, test_time_of_day_imp, test_uncertainty_win, isTest=True, eval_length=12
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader, N, std, mean
