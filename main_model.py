import numpy as np
import torch
import torch.nn as nn
import threading
import math
import logging
import torch.nn.functional as F


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class IntgTimeEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        projection_dim = projection_dim or embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim // 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, intg_time_step):
        x = self.embedding[intg_time_step]
        x = F.silu(self.projection1(x))
        return F.silu(self.projection2(x))

    @staticmethod
    def _build_embedding(num_steps, dim):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class FlashAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).reshape(batch, length, channels)
        return self.out_proj(output), None


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, intg_time_embedding_dim, nheads):
        super().__init__()
        self.intg_time_projection = nn.Linear(intg_time_embedding_dim, channels)
        self.cond_projection = conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = conv1d_with_init(channels, 2 * channels, 1)
        self.feature_attn = FlashAttention(channels, nheads)
        self.feature_ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )
        self.feature_norm1 = nn.LayerNorm(channels)
        self.feature_norm2 = nn.LayerNorm(channels)
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, values, base_shape):
        batch, channels, nodes, steps = base_shape
        if steps == 1:
            return values
        values = values.reshape(batch, channels, nodes, steps)
        values = values.permute(0, 2, 1, 3).reshape(batch * nodes, channels, steps)
        values = self.time_layer(values.permute(2, 0, 1)).permute(1, 2, 0)
        return values.reshape(batch, nodes, channels, steps).permute(0, 2, 1, 3).reshape(
            batch, channels, nodes * steps
        )

    def forward_feature(self, values, base_shape):
        batch, channels, nodes, steps = base_shape
        if nodes == 1:
            return values, None
        values = values.reshape(batch, channels, nodes, steps)
        values = values.permute(0, 3, 2, 1).reshape(batch * steps, nodes, channels)
        values = values + self.feature_attn(self.feature_norm1(values))[0]
        values = values + self.feature_ffn(self.feature_norm2(values))
        values = values.reshape(batch, steps, nodes, channels)
        return values.permute(0, 3, 2, 1).reshape(batch, channels, nodes * steps), None

    def forward(self, x, cond_info, intg_time_emb):
        batch, channels, nodes, steps = x.shape
        base_shape = x.shape
        values = x.reshape(batch, channels, nodes * steps)
        values = values + self.intg_time_projection(intg_time_emb).unsqueeze(-1)
        values = self.forward_time(values, base_shape)
        values, attn_weights = self.forward_feature(values, base_shape)
        values = self.mid_projection(values)
        cond_info = cond_info.reshape(batch, cond_info.shape[1], nodes * steps)
        values = values + self.cond_projection(cond_info)
        gate, filter_values = torch.chunk(values, 2, dim=1)
        values = self.output_projection(torch.sigmoid(gate) * torch.tanh(filter_values))
        residual, skip = torch.chunk(values, 2, dim=1)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip, attn_weights


class LOFTNet(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = int(config["channels"])
        self.intg_time_embedding = IntgTimeEmbedding(
            num_steps=int(config["num_steps"]),
            embedding_dim=int(config["intg_time_embedding_dim"]),
        )
        self.input_projection = conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=int(config["side_dim"]),
                    channels=self.channels,
                    intg_time_embedding_dim=int(config["intg_time_embedding_dim"]),
                    nheads=int(config["nheads"]),
                )
                for _ in range(int(config["layers"]))
            ]
        )

    def forward(self, x, cond_info, intg_time_step):
        batch, inputdim, nodes, steps = x.shape
        x = self.input_projection(x.reshape(batch, inputdim, nodes * steps))
        x = F.relu(x).reshape(batch, self.channels, nodes, steps)
        intg_time_emb = self.intg_time_embedding(intg_time_step)
        skips = []
        attn_weights = None
        for layer in self.residual_layers:
            x, skip, attn_weights = layer(x, cond_info, intg_time_emb)
            skips.append(skip)
        x = torch.sum(torch.stack(skips), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(batch, self.channels, nodes * steps)
        x = F.relu(self.output_projection1(x))
        return self.output_projection2(x).reshape(batch, nodes, steps), attn_weights


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
        self.warmup_epochs = int(config["train"].get("warmup_epochs", 0))

        self.use_consistency = int(config["train"].get("use_consistency", 1)) == 1
        self.diagnose_consistency_when_disabled = 1

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim + 1
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_fm = config["flow_matching"]
        config_fm["side_dim"] = str(self.emb_total_dim)
        self.min_alpha = float(config_fm["min_alpha"])
        self.exp_for_easy = float(config_fm["exp_for_easy"])
        self.exp_for_hard = float(config_fm["exp_for_hard"])

        input_dim = 2
        self.velocity_net = LOFTNet(config_fm, input_dim)
        self.num_steps = int(config_fm["num_steps"])
        self.results = {}
        self.last_loss_components = {}

    def get_warmup_epochs(self, total_epochs):
        return min(self.warmup_epochs, total_epochs)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model)
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

    def get_side_info(self, observed_tp, cond_mask, time_of_day):
        B, K, L = cond_mask.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(torch.arange(self.target_dim).to(self.device))
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)
        side_info = side_info.permute(0, 3, 2, 1)

        side_mask = cond_mask.unsqueeze(1)
        side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(self, observed_data, cond_mask, observed_mask, time_of_day, prior_mean, side_info, is_train,
                        true_data=None, prior_uncertainty=None, current_epoch=0, total_epochs=1):
        loss_sum = 0
        metrics_sum = {"loss_fm": 0.0, "loss_cons": 0.0, "cos_tgt": 0.0, "cos_tch": 0.0, "vel_mag_ratio": 0.0,
                       "grad_cos": 0.0}
        for t in range(self.num_steps):
            loss, metrics = self.calc_loss(
                observed_data, cond_mask, observed_mask, time_of_day, prior_mean, side_info, is_train, set_t=t,
                true_data=true_data, prior_uncertainty=prior_uncertainty, current_epoch=current_epoch,
                total_epochs=total_epochs
            )
            loss_sum += loss.detach()
            for k in metrics_sum:
                metrics_sum[k] += metrics.get(k, 0.0)

        for k in metrics_sum:
            metrics_sum[k] /= self.num_steps

        return loss_sum / self.num_steps, metrics_sum


    def get_rectification_schedule(self, t_float, prior_uncertainty, target_mask, current_epoch, total_epochs):
        B = t_float.size(0)
        device = t_float.device
        sigma_scale = 5.0
        tau = 0.1

        if prior_uncertainty is not None and isinstance(prior_uncertainty, torch.Tensor) and target_mask is not None:
            uncertainty_metric = torch.zeros(B, device=device)
            flat_mask = target_mask.reshape(B, -1)
            flat_uncertainty = prior_uncertainty.reshape(B, -1)

            for b in range(B):
                valid_idx = torch.nonzero(flat_mask[b] > 0).squeeze(-1)
                if valid_idx.numel() > 0:
                    valid_uncertainty = flat_uncertainty[b, valid_idx]
                    max_uncertainty = torch.max(valid_uncertainty)
                    scaled_uncertainty = (valid_uncertainty - max_uncertainty) / tau
                    weights = F.softmax(scaled_uncertainty, dim=0)
                    uncertainty_metric[b] = torch.sum(weights * valid_uncertainty)
                else:
                    uncertainty_metric[b] = 0.0

            uncertainty_norm = torch.tanh(uncertainty_metric * sigma_scale)
            sample_floor = self.min_alpha + (1.0 - self.min_alpha) * uncertainty_norm
            sample_floor = sample_floor.view(-1, 1, 1)
        else:
            uncertainty_metric = torch.zeros(B, device=device)
            sample_floor = torch.tensor(self.min_alpha, device=device).view(1, 1, 1).expand(B, 1, 1)

        warmup_epochs = self.get_warmup_epochs(total_epochs)
        if current_epoch < warmup_epochs:
            time_decay = 1.0
        elif current_epoch >= total_epochs:
            time_decay = 0.0
        else:
            progress = (current_epoch - warmup_epochs) / max(1.0, float(total_epochs - warmup_epochs))
            time_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        alpha_val = time_decay * 1.0 + (1.0 - time_decay) * sample_floor
        alpha_flat = alpha_val.view(B)


        s_float = alpha_flat * 1.0 + (1.0 - alpha_flat) * t_float

        return alpha_val, s_float, uncertainty_metric

    def calc_loss(self, observed_data, cond_mask, observed_mask, time_of_day, prior_mean, side_info, is_train,
                  set_t=-1, true_data=None, prior_uncertainty=None,
                  current_epoch=0, total_epochs=1):
        B, K, L = observed_data.shape

        prior_mean = torch.nan_to_num(prior_mean, nan=0.0)
        observed_data = torch.nan_to_num(observed_data, nan=0.0)
        if true_data is not None:
            true_data = torch.nan_to_num(true_data, nan=0.0)

        if prior_uncertainty is not None and isinstance(prior_uncertainty, torch.Tensor):
            prior_uncertainty = torch.nan_to_num(prior_uncertainty, nan=0.0)

        target_cond = true_data if true_data is not None else observed_data
        target_data = observed_mask * true_data + (1 - observed_mask) * observed_data
        target_mask = observed_mask - cond_mask
        num_eval = target_mask.sum().clamp(min=1)

        epsilon = torch.randn_like(observed_data)
        z0 = prior_mean + epsilon

        if not is_train:
            t_float = (torch.ones(B) * set_t).float() / (self.num_steps - 1)
        else:
            t_float = torch.rand(B)

        t_float = t_float.to(self.device)
        t_indices = (t_float * (self.num_steps - 1)).round().long()

        t_expand = t_float.view(B, 1, 1)
        z_t = (1 - t_expand) * z0 + t_expand * target_data
        v_target = target_data - z0

        input_cond_student = self.set_input_to_diffmodel(z_t, target_cond, cond_mask)
        predicted_v, _ = self.velocity_net(input_cond_student, side_info, t_indices)

        residual_fm = (predicted_v - v_target) * target_mask
        loss_fm = (residual_fm ** 2).sum() / num_eval

        metrics = {
            "loss_fm": loss_fm.item(), "loss_cons": 0.0,
            "cos_tgt": 1.0, "cos_tch": 1.0, "grad_cos": 0.0
        }

        with torch.no_grad():
            flat_pred = (predicted_v * target_mask).reshape(B, -1)
            flat_target = (v_target * target_mask).reshape(B, -1)
            metrics["cos_tgt"] = F.cosine_similarity(flat_pred + 1e-8, flat_target + 1e-8, dim=1).mean().item()

        loss_cons_tensor = None

        if not is_train:
            self.last_loss_components = {"loss_fm": loss_fm, "loss_cons": loss_cons_tensor}
            return loss_fm, metrics

        target_v_final = v_target
        warmup_epochs = self.get_warmup_epochs(total_epochs)
        alpha_val = torch.ones(B, 1, 1, device=self.device)

        consistency_active = self.use_consistency and current_epoch >= warmup_epochs
        consistency_for_diagnostics = self.diagnose_consistency_when_disabled and current_epoch >= warmup_epochs

        if consistency_active:

            alpha_val, s_float, uncertainty_metric_soft = self.get_rectification_schedule(
                t_float, prior_uncertainty, target_mask, current_epoch, total_epochs
            )

            s_float = torch.clamp(s_float, max=1.0)
            s_indices = (s_float * (self.num_steps - 1)).round().long()
            s_expand = s_float.view(B, 1, 1)
            z_s = (1 - s_expand) * z0 + s_expand * target_data

            with torch.no_grad():
                input_cond_teacher = self.set_input_to_diffmodel(z_s, target_cond, cond_mask)
                v_teacher, _ = self.velocity_net(input_cond_teacher, side_info, s_indices)

            if consistency_active:
                target_v_final = alpha_val * v_target + (1.0 - alpha_val) * v_teacher.detach()

            loss_cons_tensor = (((predicted_v - v_teacher.detach()) * target_mask) ** 2).sum() / num_eval

            with torch.no_grad():
                flat_teacher = (v_teacher.detach() * target_mask).reshape(B, -1)
                metrics["cos_tch"] = F.cosine_similarity(flat_pred + 1e-8, flat_teacher + 1e-8, dim=1).mean().item()
                metrics["loss_cons"] = loss_cons_tensor.detach().item()

                g_acc = flat_pred - flat_target
                g_cons = flat_pred - flat_teacher
                metrics["grad_cos"] = F.cosine_similarity(g_acc, g_cons, dim=1, eps=1e-8).mean().item()

                if is_train:
                    cos_sample = torch.zeros(B, device=self.device)
                    sigma_sample = uncertainty_metric_soft.clone()

                    flat_mask = target_mask.reshape(B, -1)
                    if prior_uncertainty is not None and isinstance(prior_uncertainty, torch.Tensor):
                        flat_sigma = prior_uncertainty.reshape(B, -1)
                        for b in range(B):
                            valid_idx = torch.nonzero(flat_mask[b] > 0).squeeze(-1)
                            if valid_idx.numel() > 0:
                                valid_sigmas = flat_sigma[b, valid_idx]
                                k = max(1, int(valid_idx.numel() * 0.25))
                                topk_vals, topk_indices = torch.topk(valid_sigmas, k)

                                hard_idx = valid_idx[topk_indices]
                                v_t = flat_target[b, hard_idx]
                                v_tch = flat_teacher[b, hard_idx]
                                eff_cos = F.cosine_similarity(v_t.unsqueeze(0), v_tch.unsqueeze(0), eps=1e-8).squeeze(0)
                                cos_sample[b] = eff_cos
                            else:
                                cos_sample[b] = 0.0

                    metrics["target_cos_sample"] = cos_sample.detach().cpu().numpy()
                    metrics["sigma_sample"] = sigma_sample.detach().cpu().numpy()

        residual = (predicted_v - target_v_final.detach()) * target_mask

        if consistency_active:
            with torch.no_grad():
                num_eval_per_sample = target_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)
                mse_per_sample = (residual ** 2).sum(dim=(1, 2), keepdim=True) / num_eval_per_sample
                c = 1e-3
                w = alpha_val.view(B, 1, 1) / (mse_per_sample.detach() + c)
                w = w / (w.mean() + 1e-8)
            weighted_sq_err = w * (residual ** 2)
            total_loss = weighted_sq_err.sum() / num_eval
        else:
            total_loss = (residual ** 2).sum() / num_eval

        self.last_loss_components = {"loss_fm": loss_fm, "loss_cons": loss_cons_tensor}
        return total_loss, metrics

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


    def impute(self, observed_data, cond_mask, time_of_day, prior_mean, side_info, n_samples, prior_uncertainty=None,
               true_data=None, observed_mask=None, collect_trace=False, trace_samples=1):
        n_inference_steps = int(self.config["flow_matching"].get("inference_steps", 2))
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)
        timesteps = torch.linspace(0, 1.0, n_inference_steps + 1).to(self.device)
        steps = len(timesteps) - 1

        def get_time_idx(t_float):
            idx = int(round(t_float * (self.num_steps - 1)))
            return min(idx, self.num_steps - 1)


        vmr_all = torch.zeros(B, device=self.device)
        epe_all = torch.zeros(B, device=self.device)
        tgt_cos_all = torch.zeros(B, device=self.device)

        target_data = observed_mask * true_data + (
                    1 - observed_mask) * observed_data if true_data is not None else observed_data
        target_mask = observed_mask - cond_mask if observed_mask is not None else (1 - cond_mask)

        trace_samples = min(trace_samples, n_samples)
        trace_trajectories = []
        trace_velocities = []

        for i in range(n_samples):
            epsilon = torch.randn_like(observed_data)
            current_sample = prior_mean + epsilon
            z0_start = current_sample.clone()


            v_ideal = target_data - z0_start

            sample_vmr = torch.zeros(B, device=self.device)
            sample_epe = torch.zeros(B, device=self.device)
            sample_tgt_cos = torch.zeros(B, device=self.device)
            sample_trace = [current_sample.detach().cpu()] if collect_trace and i < trace_samples else None
            sample_velocity_trace = [] if collect_trace and i < trace_samples else None

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
                if sample_velocity_trace is not None:
                    sample_velocity_trace.append(v1.detach().cpu())


                flat_v1 = (v1 * target_mask).reshape(B, -1)
                flat_v_ideal = (v_ideal * target_mask).reshape(B, -1)
                valid_mask = target_mask.reshape(B, -1).sum(dim=1) > 0


                mag_v1 = torch.norm(flat_v1, dim=1)
                mag_videal = torch.norm(flat_v_ideal, dim=1)
                vmr = mag_v1 / (mag_videal + 1e-8)
                sample_vmr += torch.where(valid_mask, vmr, torch.zeros_like(vmr))


                x_pred = current_sample + (1.0 - t_curr) * v1
                flat_x_pred = (x_pred * target_mask).reshape(B, -1)
                flat_target_data = (target_data * target_mask).reshape(B, -1)
                num_valid_pixels = target_mask.reshape(B, -1).sum(dim=1).clamp(min=1e-5)
                epe = torch.abs(flat_x_pred - flat_target_data).sum(dim=1) / num_valid_pixels
                sample_epe += torch.where(valid_mask, epe, torch.zeros_like(epe))


                tgt_cos = F.cosine_similarity(flat_v1 + 1e-8, flat_v_ideal + 1e-8, dim=1)
                sample_tgt_cos += torch.where(valid_mask, tgt_cos, torch.zeros_like(tgt_cos))


                next_sample = current_sample + dt * v1
                ideal_known = (1 - t_next) * z0_start + t_next * observed_data
                current_sample = cond_mask * ideal_known + (1 - cond_mask) * next_sample
                if sample_trace is not None:
                    sample_trace.append(current_sample.detach().cpu())

            imputed_samples[:, i] = current_sample.detach()
            if sample_trace is not None:
                trace_trajectories.append(torch.stack(sample_trace, dim=1))
                trace_velocities.append(torch.stack(sample_velocity_trace, dim=1))

            vmr_all += sample_vmr / steps
            epe_all += sample_epe / steps
            tgt_cos_all += sample_tgt_cos / steps

        vmr_all /= n_samples
        epe_all /= n_samples
        tgt_cos_all /= n_samples

        inf_metrics = {
            "vmr": vmr_all.cpu().numpy(),
            "epe": epe_all.cpu().numpy(),
            "tgt_cos": tgt_cos_all.cpu().numpy(),

            "sigma_raw": prior_uncertainty.cpu().numpy() if prior_uncertainty is not None else np.zeros((B, K, L))
        }
        if collect_trace and trace_trajectories:
            inf_metrics["trace"] = {
                "trajectory": torch.stack(trace_trajectories, dim=1),
                "velocity": torch.stack(trace_velocities, dim=1),
                "timesteps": timesteps.detach().cpu(),
                "cond_mask": cond_mask.detach().cpu(),
                "target_mask": target_mask.detach().cpu(),
                "observed_data": observed_data.detach().cpu(),
                "target_data": target_data.detach().cpu(),
                "prior_mean": prior_mean.detach().cpu(),
            }

        return imputed_samples, inf_metrics

    def forward(self, batch, is_train=1, current_epoch=0, total_epochs=1):
        (
            observed_data, observed_mask, observed_tp, gt_mask, time_of_day, prior_mean,
            for_pattern_mask, _, true_data, prior_uncertainty,
        ) = self.process_data(batch, is_train)

        if self.target_strategy != "random":
            cond_mask = self.get_hist_mask(observed_mask, for_pattern_mask=for_pattern_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask, time_of_day)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, time_of_day, prior_mean, side_info, is_train,
                         true_data=true_data, prior_uncertainty=prior_uncertainty, current_epoch=current_epoch,
                         total_epochs=total_epochs)

    def evaluate(self, batch, n_samples, collect_trace=False, trace_samples=1):
        (
            observed_data, observed_mask, observed_tp, gt_mask, time_of_day, prior_mean,
            for_pattern_mask, cut_length, true_data, prior_uncertainty,
        ) = self.process_data(batch, 0)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask, time_of_day)

            samples, inf_metrics = self.impute(
                observed_data, cond_mask, time_of_day, prior_mean, side_info, n_samples,
                prior_uncertainty=prior_uncertainty, true_data=true_data, observed_mask=observed_mask,
                collect_trace=collect_trace, trace_samples=trace_samples
            )

            for i in range(len(cut_length)):
                target_mask[i, ..., 0: cut_length[i].item()] = 0

            B = observed_data.size(0)
            sigma_sample = torch.zeros(B, device=self.device)
            flat_mask = target_mask.reshape(B, -1)
            if prior_uncertainty is not None and isinstance(prior_uncertainty, torch.Tensor):
                flat_sigma = prior_uncertainty.reshape(B, -1)
                for b in range(B):
                    valid_idx = torch.nonzero(flat_mask[b] > 0).squeeze(-1)
                    if valid_idx.numel() > 0:
                        valid_sigmas = flat_sigma[b, valid_idx]
                        max_sigma = torch.max(valid_sigmas)
                        scaled_sigmas = (valid_sigmas - max_sigma) / 0.1
                        weights = F.softmax(scaled_sigmas, dim=0)
                        sigma_sample[b] = torch.sum(weights * valid_sigmas)
                    else:
                        sigma_sample[b] = 0.0

            inf_metrics["sigma_sample"] = sigma_sample.cpu().numpy()

        return samples, observed_data, target_mask, observed_mask, observed_tp, inf_metrics


class LOFTTraffic(FlowBase):
    def __init__(self, config, target_dim, device):
        super(LOFTTraffic, self).__init__(target_dim, config, device)
        self.device = device

    def process_data(self, batch, is_train=1):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        time_of_day = batch["time_of_day"]
        prior_mean = batch["prior_mean"]
        true_data = batch["true_data"].to(self.device).float()
        prior_uncertainty = batch["prior_uncertainty"]
        time_of_day = time_of_day.to(self.device).float()
        prior_mean = prior_mean.to(self.device).float()
        prior_uncertainty = prior_uncertainty.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        time_of_day = time_of_day.permute(0, 2, 1)
        prior_mean = prior_mean.permute(0, 2, 1)
        true_data = true_data.permute(0, 2, 1)
        prior_uncertainty = prior_uncertainty.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data, observed_mask, observed_tp, gt_mask, time_of_day,
            prior_mean, for_pattern_mask, cut_length, true_data, prior_uncertainty,
        )
