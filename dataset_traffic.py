import numpy as np
from torch.utils.data import DataLoader, Dataset


class Traffic_Dataset(Dataset):
    def __init__(self, c_data, true_data, observed_masks, gt_mask, avg_imp, sigma_data, isTest=False, eval_length=12):
        self.eval_length = eval_length
        self.observed_masks = observed_masks
        self.gt_masks = gt_mask
        self.observed_data = c_data
        self.avg_imp = avg_imp
        self.imputed_data = c_data
        self.true_data = true_data
        self.sigma_data = sigma_data

        if isTest:
            self.observed_data = true_data
            self.observed_masks = np.ones_like(self.gt_masks)

    def __getitem__(self, index):
        s = {
            "observed_data": self.observed_data[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
            "avg_imp": self.avg_imp[index],
            "imputed_data": self.imputed_data[index],
            "true_data": self.true_data[index],
            "sigma_data": self.sigma_data[index],
        }
        return s

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


def get_sample_by_overlaped_Sliding_window(X, Y, mask, gt_mask, avg, sigma, sample_len):
    X_window, Y_window, mask_window, gt_mask_window, avg_window, sigma_window = [], [], [], [], [], []
    for i in range(X.shape[0] - sample_len + 1):
        X_window.append(X[i:i + sample_len])
        Y_window.append(Y[i:i + sample_len])
        mask_window.append(mask[i:i + sample_len])
        gt_mask_window.append(gt_mask[i:i + sample_len])
        avg_window.append(avg[i:i + sample_len])
        sigma_window.append(sigma[i:i + sample_len])

    X_window = np.array(X_window)
    Y_window = np.array(Y_window)
    mask_window = np.array(mask_window)
    gt_mask_window = np.array(gt_mask_window)
    avg_window = np.array(avg_window)
    sigma_window = np.array(sigma_window)

    return X_window, Y_window, mask_window, gt_mask_window, avg_window, sigma_window


def get_dataloader(true_datapath, miss_datapath, imputed_datapath, val_ratio, test_ratio, batch_size=16, eval_length=12):
    miss = np.load(miss_datapath)
    true = np.load(true_datapath)

    observed_masks = np.nan_to_num(true['mask'][:, :, 0])
    gt_masks = np.nan_to_num(miss['mask'][:, :, 0])
    true_data = np.nan_to_num(np.load(true_datapath)['data'][:, :, 0].astype(np.float32))

    mean = np.mean(true_data[observed_masks == 1])
    std = np.std(true_data[observed_masks == 1])
    true_data = (true_data - mean) / std

    print(f"Loading Imputed Data from: {imputed_datapath}")
    imputed_npz = np.load(imputed_datapath)

    if 'data_impute' in imputed_npz:
        imputed_data = imputed_npz['data_impute']
    elif 'imputed_data' in imputed_npz:
        imputed_data = imputed_npz['imputed_data']
    else:

        print(f"Warning: 'data_impute' or 'imputed_data' not found in {imputed_datapath}. Using the first key: {imputed_npz.files[0]}")
        imputed_data = imputed_npz[imputed_npz.files[0]]

    if 'sigma' in imputed_npz:
        sigma_data = imputed_npz['sigma']
        print("Successfully loaded 'sigma' from imputed file.")
    else:
        print("Warning: 'sigma' not found in imputed file. Using zeros.")
        sigma_data = np.zeros_like(imputed_data)

    imputed_data = imputed_data[:, :]
    sigma_data = sigma_data[:, :]

    c_data = (imputed_data - mean) / std

    sigma_data_norm = sigma_data / std
    print(f"Prior Std (Norm) -> Mean: {sigma_data_norm.mean():.4f}, Min: {sigma_data_norm.min():.4f}, Max: {sigma_data_norm.max():.4f}")

    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    norm_quantiles = np.quantile(sigma_data_norm, qs)

    orig_quantiles = np.quantile(sigma_data, qs)

    print(f"{'Quantile':>10} | {'Norm Value':>12} | {'Original Value':>15}")
    print("-" * 45)

    for q, n_val, o_val in zip(qs, norm_quantiles, orig_quantiles):
        print(f"{q:>10.1%} | {n_val:>12.4f} | {o_val:>15.4f}")

    avg_imputed_data = construct_time_of_day_encoding(c_data)

    T, N = true_data.shape

    val_len = int(T * val_ratio)
    test_len = int(T * test_ratio)

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

    train_avg, val_avg, test_avg = avg_imputed_data[:-(val_len + test_len)], \
        avg_imputed_data[-(val_len + test_len):-(test_len)], \
        avg_imputed_data[-test_len:]
        

    train_sigma, val_sigma, test_sigma = sigma_data_norm[:-(val_len + test_len)], \
        sigma_data_norm[-(val_len + test_len):-(test_len)], \
        sigma_data_norm[-test_len:]

    train_X, train_Y, train_mask, train_gtmask, train_avg_imp, train_sigma_win = get_sample_by_overlaped_Sliding_window(train_X, train_Y,
                                                                                                       train_mask,
                                                                                                       train_gtmask,
                                                                                                       train_avg,
                                                                                                       train_sigma,
                                                                                                       eval_length)
    val_X, val_Y, val_mask, val_gtmask, val_avg_imp, val_sigma_win = get_sample_by_overlaped_Sliding_window(val_X, val_Y, val_mask,
                                                                                             val_gtmask, val_avg,
                                                                                             val_sigma,
                                                                                             eval_length)
    test_X, test_Y, test_mask, test_gtmask, test_avg_imp, test_sigma_win = get_sample_by_overlaped_Sliding_window(test_X, test_Y,
                                                                                                  test_mask,
                                                                                                  test_gtmask, test_avg,
                                                                                                  test_sigma,
                                                                                                  eval_length)

    dataset = Traffic_Dataset(
        train_X, train_Y, train_mask, train_gtmask, train_avg_imp, train_sigma_win, isTest=False, eval_length=12
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = Traffic_Dataset(
        val_X, val_Y, val_mask, val_gtmask, val_avg_imp, val_sigma_win, isTest=False, eval_length=12
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Traffic_Dataset(
        test_X, test_Y, test_mask, test_gtmask, test_avg_imp, test_sigma_win, isTest=True, eval_length=12
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    return train_loader, valid_loader, test_loader, N, std, mean
