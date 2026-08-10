import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    from .LOFT_plt_2steps_case import TARGET_LIST
except ImportError:
    from LOFT_plt_2steps_case import TARGET_LIST


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def conditional_mask(eval_data, global_idx, node_idx):
    observed = eval_data["observed_points"][global_idx, :, node_idx]
    if "eval_points" not in eval_data:
        return observed.bool()
    return ((observed - eval_data["eval_points"][global_idx, :, node_idx]) > 0.5).bool()


def stitch_animation_data(start_global_idx, node_idx, eval_data, trace_data, sample_idx, num_segments, segment_len):
    total_windows = eval_data["samples"].shape[0]
    total_nodes = eval_data["samples"].shape[-1]
    trace_samples = trace_data["trajectory"].shape[1]

    if node_idx < 0 or node_idx >= total_nodes:
        raise ValueError(f"node_idx={node_idx} is outside [0, {total_nodes})")
    if sample_idx < 0 or sample_idx >= trace_samples:
        raise ValueError(f"sample_idx={sample_idx} is outside [0, {trace_samples})")
    last_idx = start_global_idx + (num_segments - 1) * segment_len
    if start_global_idx < 0 or last_idx >= total_windows:
        raise ValueError(f"Cannot stitch {num_segments} windows from global index {start_global_idx}")

    trajectories = []
    targets = []
    samples = []
    cond_masks = []

    for segment in range(num_segments):
        global_idx = start_global_idx + segment * segment_len

        trajectory = trace_data["trajectory"][global_idx, sample_idx, :, node_idx, :].transpose(0, 1)
        trajectories.append(trajectory.numpy())
        targets.append(eval_data["target"][global_idx, :, node_idx].numpy())
        samples.append(eval_data["samples"][global_idx, :, :, node_idx].permute(1, 0).numpy())
        cond_masks.append(conditional_mask(eval_data, global_idx, node_idx).numpy())

    return {
        "trajectory": np.concatenate(trajectories, axis=0),
        "target": np.concatenate(targets, axis=0),
        "samples": np.concatenate(samples, axis=0),
        "cond_mask": np.concatenate(cond_masks, axis=0),
    }


def add_prediction_band(ax, temporal_axis, samples):
    lower = np.min(samples, axis=1)
    upper = np.max(samples, axis=1)
    median = np.median(samples, axis=1)
    vertices = [(1.0, temporal_axis[i], lower[i]) for i in range(len(temporal_axis))]
    vertices.extend((1.0, temporal_axis[i], upper[i]) for i in range(len(temporal_axis) - 1, -1, -1))
    ax.add_collection3d(Poly3DCollection([vertices], facecolors="#FF9800", alpha=0.35, zorder=10))
    ax.plot(np.ones(len(temporal_axis)), temporal_axis, median,
            color="#D32F2F", linestyle="--", linewidth=2.5, alpha=0.95, zorder=25)


def draw_frame(ax, data, state_idx, show_observed_progressively):
    trajectory = data["trajectory"]
    target = data["target"]
    samples = data["samples"]
    cond_mask = data["cond_mask"]
    temporal_axis = np.arange(len(target))
    integration_steps = trajectory.shape[1] - 1
    integration_time = state_idx / integration_steps
    current_state = trajectory[:, state_idx]

    ax.clear()
    transparent_pane = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(transparent_pane)
    ax.yaxis.set_pane_color(transparent_pane)
    ax.zaxis.set_pane_color(transparent_pane)

    ax.plot(
        np.zeros(len(target)), temporal_axis, trajectory[:, 0],
        color="#2962FF", linestyle=":", linewidth=1.5, alpha=0.8, zorder=2,
    )
    trajectory_indices = np.linspace(max(1, len(target) // 6), len(target) - max(1, len(target) // 6), 5).astype(int)
    visible_times = np.arange(state_idx + 1) / integration_steps
    for temporal_idx in trajectory_indices:
        ax.plot(
            visible_times,
            np.full(state_idx + 1, temporal_idx),
            trajectory[temporal_idx, :state_idx + 1],
            color="black", linewidth=1.0, alpha=0.5, zorder=8,
        )

        for segment_idx in range(state_idx):
            start_t = segment_idx / integration_steps
            end_t = (segment_idx + 1) / integration_steps
            start_z = trajectory[temporal_idx, segment_idx]
            end_z = trajectory[temporal_idx, segment_idx + 1]
            dt = end_t - start_t
            dz = end_z - start_z
            norm = np.hypot(dt, dz)
            if norm > 1e-8:
                scale = 0.045
                ax.quiver(
                    start_t, temporal_idx, start_z,
                    dt / norm * scale, 0.0, dz / norm * scale,
                    color="black", alpha=0.9, arrow_length_ratio=0.45, linewidth=1.2, zorder=12,
                )


    last_intermediate_idx = min(state_idx, integration_steps - 1)
    for intermediate_idx in range(1, last_intermediate_idx + 1):
        intermediate_time = intermediate_idx / integration_steps
        ax.plot(
            np.full(len(target), intermediate_time), temporal_axis, trajectory[:, intermediate_idx],
            color="#FF9800", linewidth=2.8, alpha=1.0, zorder=20,
        )


    if state_idx == integration_steps:
        add_prediction_band(ax, temporal_axis, samples)
        ax.plot(
            np.ones(len(target)), temporal_axis, target,
            color="#191970", linewidth=2.0, alpha=0.9, zorder=30,
        )

    if state_idx > 0:
        observed_indices = np.flatnonzero(cond_mask)
        if show_observed_progressively:
            revealed_limit = int(np.ceil((integration_time + 0.08) * len(target)))
            observed_indices = observed_indices[observed_indices < revealed_limit]
        if len(observed_indices):
            ax.scatter(
                np.full(len(observed_indices), integration_time), observed_indices, current_state[observed_indices],
                color="black", s=22, marker="o", alpha=1.0, zorder=40,
            )

    ax.set_xlabel("\nIntegration Time $t$", fontsize=17, labelpad=10)
    ax.set_ylabel("\nTemporal Index", fontsize=17, labelpad=10)
    ax.set_zlabel("\nNormalized Value", fontsize=17, labelpad=8)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, len(target) - 1)
    ax.set_zlim(-3.0, 3.0)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0.0", "0.5", "1.0"])
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="z", which="major", labelsize=13)
    ax.grid(True, linestyle=":", alpha=0.12, linewidth=0.3, color="gray")
    ax.view_init(elev=25, azim=-115)
    ax.set_title(f"LOFT Integration Process  t={integration_time:.2f}", fontsize=19, pad=14)


def save_gif(data, output_path, fps, show_observed_progressively):
    integration_steps = data["trajectory"].shape[1] - 1
    frame_count = integration_steps + 1

    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx):
        draw_frame(ax, data, frame_idx, show_observed_progressively)
        return []

    animation = FuncAnimation(fig, update, frames=frame_count, interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)


def targets_from_png_dir(png_dir):
    pattern = re.compile(r"GlobalIdx(?P<global_idx>\d+)_Node(?P<node_idx>\d+)")
    targets = []
    for png_path in sorted(Path(png_dir).glob("*.png")):
        match = pattern.search(png_path.name)
        if match is None:
            print(f"Skipping PNG with no GlobalIdx/Node pattern: {png_path.name}")
            continue
        targets.append((int(match.group("global_idx")), int(match.group("node_idx"))))
    if not targets:
        raise ValueError(f"No compatible PNG files found in {png_dir}")
    return list(dict.fromkeys(targets))


def parse_args():
    parser = argparse.ArgumentParser(description="Animate LOFT integration traces as GIFs without changing the PNG plot script.")
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--global-index", type=int, default=None)
    parser.add_argument("--node-index", type=int, default=None)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--num-segments", type=int, default=6)
    parser.add_argument("--segment-len", type=int, default=12)
    parser.add_argument("--fps", type=float, default=2.5, help="Playback rate for the three real integration states")
    parser.add_argument("--all-targets", action="store_true", help="Render one GIF for every valid TARGET_LIST pair.")
    parser.add_argument("--png-dir", default=None, help="Render GIFs only for targets represented by PNG files in this directory.")
    parser.add_argument("--no-progressive-observed", action="store_true", help="Show all observed points in every animation frame.")
    return parser.parse_args()


def main():
    args = parse_args()
    trace_data = load_torch(args.trace_file)
    eval_data = load_torch(args.eval_file)

    if args.png_dir is not None:
        targets = targets_from_png_dir(args.png_dir)
    elif args.all_targets:
        targets = TARGET_LIST
    elif args.global_index is not None and args.node_index is not None:
        targets = [(args.global_index, args.node_index)]
    else:
        raise ValueError("Provide both --global-index and --node-index, or use --all-targets.")

    output_dir = os.path.abspath(args.output_dir)
    successful = 0
    for global_idx, node_idx in targets:
        try:
            data = stitch_animation_data(
                global_idx, node_idx, eval_data, trace_data, args.sample,
                args.num_segments, args.segment_len,
            )
        except ValueError as exc:
            print(f"Skipping GlobalIdx={global_idx}, Node={node_idx}: {exc}")
            continue

        output_path = os.path.join(output_dir, f"loft_integration_GlobalIdx{global_idx}_Node{node_idx}_sample{args.sample}.gif")
        print(f"Saving {output_path}")
        save_gif(
            data,
            output_path=Path(output_path),
            fps=args.fps,
            show_observed_progressively=not args.no_progressive_observed,
        )
        successful += 1

    print(f"Saved {successful} GIF file(s) to {output_dir}")


if __name__ == "__main__":
    main()
