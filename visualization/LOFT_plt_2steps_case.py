import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import torch
from tqdm import tqdm

plt.rcParams['axes.unicode_minus'] = False

BASE_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LOFT_2steps_inference")


TARGET_LIST = [
    (206, 204), (190, 153), (2202, 82), (1034, 177), (796, 41),
    (2036, 85), (538, 41), (2596, 261), (2052, 169), (572, 148),
    (780, 18), (1288, 164), (794, 41), (2886, 178), (2456, 177),
    (2454, 170), (2440, 153), (1008, 13), (2438, 153), (1002, 13),
    (1226, 285), (436, 178), (1574, 170), (2058, 164), (1020, 13),
    (2450, 169), (2716, 232), (2428, 153), (2714, 232), (2718, 232),
    (940, 82), (990, 13), (3300, 232), (3018, 232), (614, 85),
    (3016, 232), (2048, 85), (2066, 170)
]

def get_batch_info(global_idx, velocities_list):
    current_count = 0
    for b_idx, batch_vel in enumerate(velocities_list):
        batch_size = batch_vel.shape[2]
        if global_idx < current_count + batch_size:
            return b_idx, global_idx - current_count
        current_count += batch_size
    return -1, -1

def reconstruct_one_segment(v_seq, x_final):


    timesteps = torch.linspace(0, 1.0, 3)

    num_steps = len(timesteps)
    L = x_final.shape[0]

    Z_matrix = torch.zeros(L, num_steps)
    Z_matrix[:, -1] = x_final

    curr_x = x_final

    for s in range(num_steps - 2, -1, -1):
        t_curr = timesteps[s].item()
        t_next = timesteps[s + 1].item()
        dt = t_next - t_curr

        v_idx = min(s, v_seq.shape[0] - 1)
        v_s = v_seq[v_idx, :]

        prev_x = curr_x - v_s * dt
        Z_matrix[:, s] = prev_x
        curr_x = prev_x

    return Z_matrix.numpy()

def stitch_trajectory_and_samples(start_global_idx, node_idx, eval_data, velocities_list, num_segments=6, segment_len=12):
    total_samples = eval_data['samples'].shape[0]
    if start_global_idx + (num_segments - 1) * segment_len >= total_samples:
        return None

    stitched_gt = []
    stitched_mask = []
    stitched_cond_mask = []
    stitched_samples = []
    stitched_Z = []

    for i in range(num_segments):
        curr_idx = start_global_idx + i * segment_len


        gt = eval_data['target'][curr_idx, :, node_idx]
        mask = eval_data['observed_points'][curr_idx, :, node_idx]

        if 'eval_points' in eval_data:
            eval_p = eval_data['eval_points'][curr_idx, :, node_idx]
            cond_m = mask - eval_p
            cond_m = (cond_m > 0.5).float()
        else:
            cond_m = mask.clone()

        stitched_gt.append(gt)
        stitched_mask.append(mask)
        stitched_cond_mask.append(cond_m)

        samps = eval_data['samples'][curr_idx, :, :, node_idx].permute(1, 0)
        stitched_samples.append(samps)

        b_idx, s_in_b = get_batch_info(curr_idx, velocities_list)
        if b_idx == -1: return None

        v_seq = velocities_list[b_idx][0, :, s_in_b, node_idx, :]
        x_final_0 = eval_data['samples'][curr_idx, 0, :, node_idx]
        z_seg = reconstruct_one_segment(v_seq, x_final_0)
        stitched_Z.append(z_seg)

    final_gt = torch.cat(stitched_gt, dim=0).numpy()
    final_mask = torch.cat(stitched_mask, dim=0).bool().numpy()
    final_cond_mask = torch.cat(stitched_cond_mask, dim=0).bool().numpy()
    final_samples = torch.cat(stitched_samples, dim=0).numpy()
    final_Z = np.concatenate(stitched_Z, axis=0)

    return final_Z, final_gt, final_mask, final_samples, final_cond_mask

def stitch_trajectory_and_samples_from_trace(start_global_idx, node_idx, eval_data, trace_data, sample_idx=0,
                                             num_segments=6, segment_len=12):
    total_samples = eval_data['samples'].shape[0]
    trace_samples = trace_data['velocity'].shape[1]
    total_nodes = eval_data['samples'].shape[-1]
    if start_global_idx + (num_segments - 1) * segment_len >= total_samples:
        return None
    if start_global_idx + (num_segments - 1) * segment_len >= trace_data['velocity'].shape[0]:
        return None
    if node_idx >= total_nodes:
        return None
    if sample_idx >= trace_samples:
        raise ValueError(f"sample_idx={sample_idx} is out of range for trace_samples={trace_samples}")

    stitched_gt = []
    stitched_mask = []
    stitched_cond_mask = []
    stitched_samples = []
    stitched_Z = []

    for i in range(num_segments):
        curr_idx = start_global_idx + i * segment_len

        gt = eval_data['target'][curr_idx, :, node_idx]
        mask = eval_data['observed_points'][curr_idx, :, node_idx]

        if 'eval_points' in eval_data:
            eval_p = eval_data['eval_points'][curr_idx, :, node_idx]
            cond_m = mask - eval_p
            cond_m = (cond_m > 0.5).float()
        else:
            cond_m = mask.clone()

        stitched_gt.append(gt)
        stitched_mask.append(mask)
        stitched_cond_mask.append(cond_m)

        samps = eval_data['samples'][curr_idx, :, :, node_idx].permute(1, 0)
        stitched_samples.append(samps)

        v_seq = trace_data['velocity'][curr_idx, sample_idx, :, node_idx, :]
        x_final_0 = eval_data['samples'][curr_idx, sample_idx, :, node_idx]
        z_seg = reconstruct_one_segment(v_seq, x_final_0)
        stitched_Z.append(z_seg)

    final_gt = torch.cat(stitched_gt, dim=0).numpy()
    final_mask = torch.cat(stitched_mask, dim=0).bool().numpy()
    final_cond_mask = torch.cat(stitched_cond_mask, dim=0).bool().numpy()
    final_samples = torch.cat(stitched_samples, dim=0).numpy()
    final_Z = np.concatenate(stitched_Z, axis=0)

    return final_Z, final_gt, final_mask, final_samples, final_cond_mask

def plot_3d_vis(Z_matrix, z_init, gt_curve, method_name, save_name, mask=None, samples_matrix=None):
    L_final, num_steps = Z_matrix.shape

    diff_time_axis = np.linspace(0.0, 1.0, num_steps)
    real_time_axis_final = np.arange(L_final)

    fig = plt.figure(figsize=(20, 11))
    ax = fig.add_subplot(111, projection='3d')

    transparent_pane = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(transparent_pane)
    ax.yaxis.set_pane_color(transparent_pane)
    ax.zaxis.set_pane_color(transparent_pane)

    traj_step = max(1, L_final // 6)
    arrow_start = 0
    arrow_end = num_steps - 1
    time_arrow_step = 1

    trajectory_indices = np.linspace(traj_step, L_final-traj_step, 5).astype(int)

    for i in trajectory_indices:
        ts = diff_time_axis
        ys = np.full_like(ts, i)
        zs = Z_matrix[i, :]

        ax.plot(ts, ys, zs, color='black', linewidth=1.0, alpha=0.5, zorder=3)


        for t_idx in range(arrow_start, arrow_end, time_arrow_step):
            if t_idx + 1 >= len(ts): break
            x_curr, y_curr, z_curr = ts[t_idx], ys[t_idx], zs[t_idx]
            x_next, y_next, z_next = ts[t_idx+1], ys[t_idx+1], zs[t_idx+1]

            u, v, w = x_next - x_curr, 0, z_next - z_curr
            norm_vec = np.sqrt(u**2 + v**2 + w**2)
            if norm_vec < 1e-6: continue

            scale = 0.04
            u, v, w = (u/norm_vec)*scale, (v/norm_vec)*scale, (w/norm_vec)*scale
            ax.quiver(x_curr, y_curr, z_curr, u, v, w,
                      color='black', alpha=0.9, arrow_length_ratio=0.5, pivot='tail', linewidth=1.5, zorder=5)

        warm_blue = '#2962FF'
        ax.scatter([0.0], [i], [Z_matrix[i, 0]], color=warm_blue, s=30, alpha=1.0, zorder=6)

    ax.plot(np.zeros(L_final), real_time_axis_final, z_init,
            color=warm_blue, linestyle=':', linewidth=1.5, alpha=0.8, label='Noise ($t=0$)', zorder=3)

    ax.plot(np.ones(L_final), real_time_axis_final, gt_curve,
            color='#191970', linestyle='-', linewidth=2.0, alpha=0.9, label='True Data ($t=1$)', zorder=25)

    if mask is not None:
        observed_indices = np.where(mask)[0]
        ax.scatter(np.ones(len(observed_indices)), observed_indices, gt_curve[mask],
                   color='black', s=25, marker='o', alpha=1.0, zorder=30, label='Observed Input')

    if samples_matrix is not None:
        y_min = np.min(samples_matrix, axis=1)
        y_max = np.max(samples_matrix, axis=1)
        y_median = np.median(samples_matrix, axis=1)
        t_final = 1.0

        verts = []
        for i in range(L_final):
            verts.append((t_final, real_time_axis_final[i], y_min[i]))
        for i in range(L_final-1, -1, -1):
            verts.append((t_final, real_time_axis_final[i], y_max[i]))

        poly = Poly3DCollection([verts], facecolors='#FF9800', alpha=0.4, zorder=15)
        ax.add_collection3d(poly)

        y_median_plot = y_median.copy().astype(float)
        if mask is not None:
            y_median_plot[mask] = np.nan

        ax.plot(np.ones(L_final), real_time_axis_final, y_median_plot,
                color='#D32F2F', linestyle='--', linewidth=3.0, alpha=1.0,
                label='Model Prediction', zorder=20)

    intermediate_times = [0.5]
    z_floor = -3.0

    for idx, tp in enumerate(intermediate_times):
        t_idx = 1
        curve = Z_matrix[:, t_idx]

        xs = np.full(L_final, tp, dtype=float)
        ax.plot(xs, real_time_axis_final, curve,
                color='#FF9800', linestyle='-', linewidth=2.5, alpha=1.0,
                label=f'State ($t={tp}$)', zorder=20)

    ax.set_xlabel('\nIntegration Time $t$', fontsize=24, labelpad=15)
    ax.set_ylabel('\nTemporal Index', fontsize=24, labelpad=15)
    ax.set_zlabel('\nNormalized Value', fontsize=24, labelpad=12)
    ax.set_zlim(z_floor, 3.0)

    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(['0.0', '0.5', '1.0'])

    ax.set_title(f"{method_name} Denoising Process", fontsize=26, pad=20)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.tick_params(axis='z', which='major', labelsize=22)

    ax.grid(True, linestyle=':', alpha=0.1, linewidth=0.3, color='gray')
    ax.view_init(elev=25, azim=-115)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot LOFT 2-step TARGET_LIST cases with the original LOFT_plt_raw.py style."
    )
    parser.add_argument("--trace-file", default=None, help="Full trace pth saved by run.py --trace_file")
    parser.add_argument("--eval-file", default=None, help="evaluation_tensors_*.pth saved by eval")
    parser.add_argument("--vel-file", default=None, help="Legacy velocities_*.pth file")
    parser.add_argument("--output-dir", default=BASE_SAVE_DIR)
    parser.add_argument("--sample", type=int, default=0, help="Stochastic sample index used to reconstruct the denoising path")
    parser.add_argument("--max-cases", type=int, default=0, help="0 means plot all TARGET_LIST entries")
    parser.add_argument("--num-segments", type=int, default=6)
    parser.add_argument("--segment-len", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--stitch-windows", action="store_true", help="Compatibility flag; stitching is always used")
    parser.add_argument("--use-target-list", action="store_true", help="Compatibility flag; TARGET_LIST is always used")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Initializing 2-Step visualization...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.eval_file is None and args.trace_file is None and args.vel_file is None:
        saved_dir = os.path.join(current_dir, "velocities_saved")
        args.eval_file = os.path.join(saved_dir, "evaluation_tensors_PEMS04_SC-TC_0.8_exit0.3_20260203_132346.pth")
        args.vel_file = os.path.join(saved_dir, "velocities_106batches_PEMS04_SC-TC_0.8_exit0.3_20260205_184152.pth")

    if args.eval_file is None:
        raise ValueError("--eval-file is required when using --trace-file or --vel-file")
    if args.trace_file is None and args.vel_file is None:
        raise ValueError("Please provide either --trace-file or --vel-file")

    if not os.path.exists(args.eval_file):
        print(f"Error: eval file not found: {args.eval_file}")
        return
    if args.trace_file is not None and not os.path.exists(args.trace_file):
        print(f"Error: trace file not found: {args.trace_file}")
        return
    if args.vel_file is not None and not os.path.exists(args.vel_file):
        print(f"Error: velocity file not found: {args.vel_file}")
        return

    print(f"Loading data...")
    try:
        eval_data = torch.load(args.eval_file, map_location='cpu')
        trace_data = None
        velocities_list = None
        if args.trace_file is not None:
            trace_data = torch.load(args.trace_file, map_location='cpu')
        else:
            velocities_list = torch.load(args.vel_file, map_location='cpu')
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    targets = TARGET_LIST if args.max_cases <= 0 else TARGET_LIST[:args.max_cases]
    print(f"Processing {len(targets)} specific targets...")

    for i, (global_idx, node_idx) in enumerate(tqdm(targets, desc="Processing")):
        if trace_data is not None:
            res = stitch_trajectory_and_samples_from_trace(
                global_idx, node_idx, eval_data, trace_data, sample_idx=args.sample,
                num_segments=args.num_segments, segment_len=args.segment_len
            )
            b_idx = global_idx // args.batch_size
        else:
            res = stitch_trajectory_and_samples(
                global_idx, node_idx, eval_data, velocities_list,
                num_segments=args.num_segments, segment_len=args.segment_len
            )
            b_idx, _ = get_batch_info(global_idx, velocities_list)
        if res is None:
            print(f"Skipping Invalid Target: Idx {global_idx}, Node {node_idx}")
            continue

        final_Z, final_gt, final_mask, final_samples, final_cond_mask = res

        y_min = np.min(final_samples, axis=1)
        y_max = np.max(final_samples, axis=1)
        is_covered = (final_gt >= y_min) & (final_gt <= y_max)
        coverage_rate = np.mean(is_covered)
        gt_std = np.std(final_gt)
        missing_rate = 1.0 - np.mean(final_cond_mask)

        base_filename = (f"Batch{b_idx}_GlobalIdx{global_idx}_"
                         f"Node{node_idx}_Cov{coverage_rate:.2f}_Std{gt_std:.2f}_"
                         f"Mis{missing_rate:.2f}.png")

        z_init = final_Z[:, 0]

        plot_3d_vis(final_Z, z_init, final_gt, "LOFT",
                    os.path.join(args.output_dir, base_filename),
                    mask=final_cond_mask, samples_matrix=final_samples)

    print(f"\nAll done. PNG plots saved to {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
