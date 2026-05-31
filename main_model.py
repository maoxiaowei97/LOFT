import numpy as np
import torch
import torch.nn as nn
from models import LOFTNet
import threading
import math
import logging
class FlowBase(nn.Module):

    def __init__(self, target_dim, config, device):
        super().__init__()
        self.target_dim = target_dim
        self.device = device
        self.config = config
        self.emb_time_dim = int(config["model"]["timeemb"])
        self.emb_feature_dim = int(config["model"]["featureemb"])
        self.target_strategy = config["model"]["target_strategy"]


        self.device_cond = torch.device(config["model"]["device"])
        self.alpha_warmup_ratio =  float(config["train"]["alpha_warmup_ratio"])

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_fm = config["flow_matching"]
        config_fm["side_dim"] = str(self.emb_total_dim)
        self.min_alpha = config_fm["min_alpha"]
        self.exp_for_easy = config_fm["exp_for_easy"]
        self.exp_for_hard = config_fm["exp_for_hard"]

        input_dim = 2
        self.velocity_net = LOFTNet(config_fm, input_dim)
        self.num_steps = int(config_fm["num_steps"])
        self.results = {}

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(
            self.device)  # [32,12,128]
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask, avg_imp):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        side_mask = cond_mask.unsqueeze(1)
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
            self, observed_data, cond_mask, observed_mask, avg_imp, imputed_data, side_info, is_train, true_data=None, sigma_data=None
    ):
        loss_sum = 0
        for t in range(self.num_steps):
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, avg_imp, imputed_data, side_info, is_train, set_t=t, true_data=true_data, sigma_data=sigma_data
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps


    def get_alpha(self, current_epoch, total_epochs, sigma_data):

        ks = int(total_epochs * self.alpha_warmup_ratio)
        ke = total_epochs

        if current_epoch < ks:
            if isinstance(sigma_data, torch.Tensor):
                return torch.ones(sigma_data.size(0), 1, 1, device=sigma_data.device)
            return 1.0

        if current_epoch >= ke:
            cosine_factor = 0.0
        else:
            progress = (current_epoch - ks) / (float(ke - ks) * 1.1 + 1e-6)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        sigma_scale = 5.0

        if sigma_data is not None and isinstance(sigma_data, torch.Tensor):
            if sigma_data.dim() > 1:
                sigma_metric = sigma_data.reshape(sigma_data.size(0), -1).mean(dim=1)
            else:
                sigma_metric = sigma_data

            sigma_norm = torch.tanh(sigma_metric * sigma_scale)
        else:
            val = sigma_data if (sigma_data is not None and not isinstance(sigma_data, torch.Tensor)) else 0.0
            device = self.device if hasattr(self, 'device') else 'cpu'
            sigma_norm = torch.tensor(math.tanh(val * sigma_scale), device=device)

        target_floor = self.min_alpha
        operating_range = 1.0 - self.min_alpha

        exp_for_hard = self.exp_for_hard
        exp_for_easy = self.exp_for_easy

        exponent = exp_for_easy - (exp_for_easy - exp_for_hard) * sigma_norm

        adjusted_decay = cosine_factor ** exponent

        alpha_final = target_floor + operating_range * adjusted_decay

        if isinstance(alpha_final, torch.Tensor) and alpha_final.dim() == 1:

            alpha_final = alpha_final.reshape(-1, 1, 1)

        return alpha_final

    def calc_loss(self, observed_data, cond_mask, observed_mask, avg_imp, imputed_data, side_info, is_train,
                  set_t=-1, true_data=None, sigma_data=None, current_epoch=0, total_epochs=1):
        B, K, L = observed_data.shape

        imputed_data = torch.nan_to_num(imputed_data, nan=0.0)
        observed_data = torch.nan_to_num(observed_data, nan=0.0)
        if true_data is not None:
            true_data = torch.nan_to_num(true_data, nan=0.0)

        if not is_train:
            t = (torch.ones(B) * set_t).long().to(self.device)
            t_float = t.float() / (self.num_steps - 1)
        else:
            t_float = torch.rand(1, device=self.device).expand(B)
            t_indices = (t_float * (self.num_steps - 1)).long()
            t = t_indices

        target_cond = true_data if true_data is not None else observed_data
        target_data = observed_mask * true_data + (1 - observed_mask) * observed_data

        epsilon = torch.randn_like(observed_data)
        z0 = imputed_data + epsilon
        t_expand = t_float.view(B, 1, 1)

        v_target = target_data - z0
        target_v_final = v_target

        warmup_epochs = int(total_epochs * self.alpha_warmup_ratio)
        current_t_idx = t[0].item()

        with torch.no_grad():

            t_dense = torch.linspace(0, 0.8, 9)
            t_sparse = torch.tensor([1.0])

            schedule_float = torch.cat([t_dense, t_sparse])

            schedule_indices = torch.round(schedule_float * (self.num_steps - 1)).long().to(self.device)
            schedule_indices = torch.unique(schedule_indices, sorted=True)

            dense_cutoff_float = 0.8
            dense_cutoff_idx = int(round(dense_cutoff_float * (self.num_steps - 1)))

        is_in_schedule = (current_t_idx == schedule_indices[:-1]).any().item()
        is_sparse_step = current_t_idx >= dense_cutoff_idx

        if (current_epoch >= warmup_epochs and is_in_schedule and is_sparse_step):
            alpha_val = self.get_alpha(current_epoch, total_epochs, sigma_data)

            loc = (schedule_indices == current_t_idx).nonzero(as_tuple=True)[0].item()
            s_idx_scalar = schedule_indices[loc + 1].item()

            s_indices = torch.full((B,), s_idx_scalar, dtype=torch.long, device=self.device)
            s_float = s_indices.float() / (self.num_steps - 1)
            s_expand = s_float.view(B, 1, 1)

            z_s = (1 - s_expand) * z0 + s_expand * target_data

            with torch.no_grad():
                input_cond_teacher = self.set_input_to_diffmodel(z_s, target_cond, cond_mask)
                v_teacher, _ = self.velocity_net(input_cond_teacher, side_info, s_indices)
                v_teacher = torch.nan_to_num(v_teacher, nan=0.0)
                v_teacher = torch.clamp(v_teacher, min=-10.0, max=10.0)

            target_v_final = alpha_val * v_target + (1 - alpha_val) * v_teacher

        noisy_data_t = (1 - t_expand) * z0 + t_expand * target_data
        input_cond_student = self.set_input_to_diffmodel(noisy_data_t, target_cond, cond_mask)
        predicted, _ = self.velocity_net(input_cond_student, side_info, t)

        target_mask = observed_mask - cond_mask
        residual = (target_v_final - predicted) * target_mask

        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)

        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):

        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)

        total_input = torch.cat([cond_obs, noisy_target], dim=1)
        return total_input


    def run_conditional(self, diff_input_cpu, side_info_tensor, time_step):
        diff_input_cond = diff_input_cpu.to(self.device_cond, non_blocking=True)
        side_info_cond = side_info_tensor.to(self.device_cond, non_blocking=True)
        time_tensor_cond = torch.tensor([time_step]).to(self.device_cond)
        
        self.velocity_net.to(self.device_cond)

        predicted_cond, attn_cond = self.velocity_net(diff_input_cond, side_info_cond, time_tensor_cond)

        self.results['predicted_cond'] = predicted_cond
        self.results['attn_cond'] = attn_cond

    def impute(self, observed_data, cond_mask, avg_imp, imputed_data, side_info, n_samples, sigma_data=None):

        n_inference_steps = int(self.config["flow_matching"].get("inference_steps", 2))

        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        timesteps = torch.linspace(0, 1.0, n_inference_steps + 1).to(self.device)

        # 实际循环次数
        steps = len(timesteps) - 1

        def get_time_idx(t_float):

            idx = int(round(t_float * (self.num_steps - 1)))
            return min(idx, self.num_steps - 1)

        for i in range(n_samples):

            epsilon = torch.randn_like(observed_data)

            current_sample = imputed_data + epsilon

            z0_start = current_sample.clone()

            for s in range(steps):
                t_curr = timesteps[s].item()
                t_next = timesteps[s + 1].item()

                dt = t_next - t_curr

                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                diff_input_cond_cpu = torch.cat([cond_obs, noisy_target], dim=1).cpu()

                t_idx_curr = get_time_idx(t_curr)

                thread_cond = threading.Thread(target=self.run_conditional,
                                               args=(diff_input_cond_cpu, side_info, t_idx_curr))
                thread_cond.start()
                thread_cond.join()

                v1 = self.results['predicted_cond'].to(self.device)

                next_sample = current_sample + dt * v1

                ideal_known = (1 - t_next) * z0_start + t_next * observed_data

                current_sample = cond_mask * ideal_known + (1 - cond_mask) * next_sample

            imputed_samples[:, i] = current_sample.detach()

        return imputed_samples

    def forward(self, batch, is_train=1, current_epoch=0, total_epochs=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            avg_imp,
            imputed_data,
            for_pattern_mask,
            _,
            true_data,
            sigma_data,
        ) = self.process_data(batch, is_train)

        if self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask, avg_imp)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, avg_imp, imputed_data, side_info, is_train, true_data=true_data, sigma_data=sigma_data, current_epoch=current_epoch, total_epochs=total_epochs) # 新增参数

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            avg_imp,
            imputed_data,
            for_pattern_mask,
            cut_length,
            _,
            sigma_data,
        ) = self.process_data(batch, 0)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask, avg_imp)

            samples = self.impute(observed_data, cond_mask, avg_imp, imputed_data, side_info, n_samples, sigma_data=sigma_data) # 新增参数

            for i in range(len(cut_length)):
                target_mask[i, ..., 0: cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class LOFT_Traffic(FlowBase):
    def __init__(self, config, target_dim, device):
        super(LOFT_Traffic, self).__init__(target_dim, config, device)
        self.device = device

    def process_data(self, batch, is_train=1):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        avg_imp = batch["avg_imp"].to(self.device).float()
        imputed_data = batch["imputed_data"].to(self.device).float()
        true_data = batch["true_data"].to(self.device).float()
        sigma_data = batch["sigma_data"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        avg_imp = avg_imp.permute(0, 2, 1)
        imputed_data = imputed_data.permute(0, 2, 1)
        true_data = true_data.permute(0, 2, 1)
        sigma_data = sigma_data.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            avg_imp,
            imputed_data,
            for_pattern_mask,
            cut_length,
            true_data,
            sigma_data,
        )