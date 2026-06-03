#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def fmt(value: float | int | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def metric_text(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{fmt(value)}{suffix}"


def extract_row(payload: dict, coverage_note: str) -> dict[str, object]:
    overall = payload["metrics"]["overall"]
    dataset = payload["dataset_summary"]
    mode = payload["mode"]
    if mode == "auto":
        primary_iou = overall.get("mean_best_iou")
        recall25 = overall.get("recall@0.25")
        recall50 = overall.get("recall@0.50")
        recall75 = overall.get("recall@0.75")
        dice = None
        boundary_f = None
        masks_per_image = overall.get("mean_num_sam_masks_per_image")
        runtime = overall.get("mean_inference_time_sec_per_image")
        runtime_label = "sec/image"
        label = "SAM auto proposals"
    else:
        primary_iou = overall.get("mean_iou")
        recall25 = None
        recall50 = None
        recall75 = None
        dice = overall.get("mean_dice")
        boundary_f = overall.get("mean_boundary_fscore")
        masks_per_image = None
        runtime = overall.get("mean_prompt_inference_time_sec_per_segment")
        runtime_label = "sec/segment"
        label = "SAM oracle point"
    return {
        "baseline": label,
        "mode": mode,
        "model": payload["model_type"],
        "images": dataset["valid_photos"],
        "segments": dataset["valid_segments"],
        "coverage_note": coverage_note,
        "alignment": payload["alignment_strategy"],
        "primary_iou": primary_iou,
        "mean_dice": dice,
        "mean_boundary_f": boundary_f,
        "recall_025": recall25,
        "recall_050": recall50,
        "recall_075": recall75,
        "mean_masks_per_image": masks_per_image,
        "runtime": runtime,
        "runtime_unit": runtime_label,
    }


def write_csv_table(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "baseline",
        "mode",
        "model",
        "images",
        "segments",
        "coverage_note",
        "alignment",
        "primary_iou",
        "mean_dice",
        "mean_boundary_f",
        "recall_025",
        "recall_050",
        "recall_075",
        "mean_masks_per_image",
        "runtime",
        "runtime_unit",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown_table(rows: list[dict[str, object]], output_path: Path, title: str, description: str) -> None:
    headers = [
        "Baseline",
        "Images",
        "Segments",
        "Primary IoU",
        "Dice",
        "Boundary F",
        "R@0.25",
        "R@0.50",
        "R@0.75",
        "Masks/Image",
        "Runtime",
        "Alignment",
    ]
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["baseline"]),
                    fmt(row["images"]),
                    fmt(row["segments"]),
                    fmt(row["primary_iou"]),
                    fmt(row["mean_dice"]),
                    fmt(row["mean_boundary_f"]),
                    fmt(row["recall_025"]),
                    fmt(row["recall_050"]),
                    fmt(row["recall_075"]),
                    fmt(row["mean_masks_per_image"]),
                    f"{fmt(row['runtime'])} {row['runtime_unit']}",
                    str(row["alignment"]),
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def build_auto_panel_text(payload: dict) -> str:
    overall = payload["metrics"]["overall"]
    dataset = payload["dataset_summary"]
    return "\n".join(
        [
            f"{dataset['valid_photos']} images | {dataset['valid_segments']} segments",
            f"{metric_text(overall.get('mean_num_sam_masks_per_image'))} masks/image",
            f"{metric_text(overall.get('mean_inference_time_sec_per_image'))} sec/image",
        ]
    )


def build_point_panel_text(payload: dict) -> str:
    overall = payload["metrics"]["overall"]
    dataset = payload["dataset_summary"]
    return "\n".join(
        [
            f"{dataset['valid_photos']} images | {dataset['valid_segments']} segments",
            f"{metric_text(overall.get('mean_set_image_time_sec_per_image'))} sec/set_image",
            f"{metric_text(overall.get('mean_prompt_inference_time_sec_per_segment'))} sec/segment",
        ]
    )


def add_bar_labels(axis, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def render_figure(
    auto_payload: dict,
    point_payload: dict,
    auto_example: Path,
    point_example: Path,
    output_path: Path,
    title: str,
) -> None:
    auto_overall = auto_payload["metrics"]["overall"]
    point_overall = point_payload["metrics"]["overall"]

    auto_img = np.array(Image.open(auto_example).convert("RGB"))
    point_img = np.array(Image.open(point_example).convert("RGB"))

    figure = plt.figure(figsize=(16, 12))
    grid = figure.add_gridspec(2, 2, height_ratios=[1.0, 1.5])

    axis_auto = figure.add_subplot(grid[0, 0])
    auto_labels = ["Mean IoU", "R@0.25", "R@0.50", "R@0.75"]
    auto_values = [
        auto_overall["mean_best_iou"],
        auto_overall["recall@0.25"],
        auto_overall["recall@0.50"],
        auto_overall["recall@0.75"],
    ]
    bars = axis_auto.bar(auto_labels, auto_values, color=["#4477AA", "#66CCEE", "#228833", "#AA3377"])
    axis_auto.set_ylim(0.0, 1.05)
    axis_auto.set_title("SAM automatic proposal baseline")
    axis_auto.set_ylabel("score")
    add_bar_labels(axis_auto, bars)
    axis_auto.text(
        0.02,
        0.98,
        build_auto_panel_text(auto_payload),
        transform=axis_auto.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )

    axis_point = figure.add_subplot(grid[0, 1])
    point_metric_specs = [
        ("Mean IoU", point_overall.get("mean_iou"), "#4477AA"),
        ("Mean Dice", point_overall.get("mean_dice"), "#CCBB44"),
        ("Boundary F", point_overall.get("mean_boundary_fscore"), "#EE6677"),
    ]
    point_metric_specs = [spec for spec in point_metric_specs if spec[1] is not None]
    point_labels = [spec[0] for spec in point_metric_specs]
    point_values = [spec[1] for spec in point_metric_specs]
    point_colors = [spec[2] for spec in point_metric_specs]
    bars = axis_point.bar(point_labels, point_values, color=point_colors)
    axis_point.set_ylim(0.0, 1.05)
    axis_point.set_title("SAM oracle point upper bound")
    axis_point.set_ylabel("score")
    add_bar_labels(axis_point, bars)
    axis_point.text(
        0.02,
        0.98,
        build_point_panel_text(point_payload),
        transform=axis_point.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#CCCCCC"),
    )

    axis_auto_vis = figure.add_subplot(grid[1, 0])
    axis_auto_vis.imshow(auto_img)
    axis_auto_vis.set_title("Qualitative example: automatic proposals")
    axis_auto_vis.axis("off")

    axis_point_vis = figure.add_subplot(grid[1, 1])
    axis_point_vis.imshow(point_img)
    axis_point_vis.set_title("Qualitative example: oracle point prompts")
    axis_point_vis.axis("off")

    figure.suptitle(title, fontsize=18, y=0.98)
    figure.tight_layout(rect=[0, 0, 1, 0.965])
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-aggregate", type=Path, required=True)
    parser.add_argument("--point-aggregate", type=Path, required=True)
    parser.add_argument("--auto-example", type=Path, required=True)
    parser.add_argument("--point-example", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-note", type=str, default="available-photo MINC-S test subset")
    parser.add_argument("--title", type=str, default="SAM MINC-S Test-Subset Report")
    parser.add_argument("--figure-title", type=str, default="SAM on the available MINC-S test subset")
    parser.add_argument(
        "--description",
        type=str,
        default="Treat the currently available MINC-S photo set as the held-out test set.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    auto_payload = load_json(args.auto_aggregate)
    point_payload = load_json(args.point_aggregate)
    rows = [extract_row(auto_payload, args.coverage_note), extract_row(point_payload, args.coverage_note)]

    write_csv_table(rows, args.output_dir / "summary_table.csv")
    write_markdown_table(rows, args.output_dir / "summary_table.md", args.title, args.description)
    render_figure(
        auto_payload=auto_payload,
        point_payload=point_payload,
        auto_example=args.auto_example,
        point_example=args.point_example,
        output_path=args.output_dir / "summary_figure.png",
        title=args.figure_title,
    )


if __name__ == "__main__":
    main()
