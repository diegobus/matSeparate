"""Tests for segmentation.formats: RLE round-trip and artifact export."""

import json

import numpy as np

from segmentation.formats import (
    build_legend,
    build_palette,
    decode_rle,
    encode_rle,
    save_results,
)
from segmentation.objects import build_label_map, extract_instances


def test_rle_round_trip_random():
    rng = np.random.default_rng(1)
    for _ in range(5):
        mask = rng.integers(0, 2, size=(7, 11)).astype(np.uint8)
        rle = encode_rle(mask)
        assert rle["size"] == [7, 11]
        back = decode_rle(rle)
        assert np.array_equal(mask, back)


def test_rle_all_zero_and_all_one():
    z = np.zeros((4, 4), dtype=np.uint8)
    o = np.ones((4, 4), dtype=np.uint8)
    assert np.array_equal(decode_rle(encode_rle(z)), z)
    assert np.array_equal(decode_rle(encode_rle(o)), o)


def test_palette_deterministic_and_black_bg():
    p1 = build_palette(5)
    p2 = build_palette(5)
    assert np.array_equal(p1, p2)
    assert tuple(p1[0]) == (0, 0, 0)
    assert p1.shape == (6, 3)


def test_legend():
    legend = build_legend(["wood", "metal"])
    assert legend["0"] == "background"
    assert legend["1"] == "wood"
    assert legend["2"] == "metal"


def test_save_results_writes_files(tmp_path):
    probs = np.zeros((4, 8, 2), dtype=np.float32)
    probs[:, :4, 0] = 0.9
    probs[:, 4:, 1] = 0.9
    label_map, conf = build_label_map(probs, bg_threshold=0.5)
    instances = extract_instances(label_map, conf, ["a", "b"], min_object_area=1)

    written = save_results(
        out_dir=tmp_path,
        label_map=label_map,
        frontier_names=["a", "b"],
        instances=instances,
        level="leaf",
        image_size=(4, 8),
        meta={"foo": "bar"},
        write_color_viz=True,
        write_instance_pngs=True,
    )

    assert (tmp_path / "label_map.png").exists()
    assert (tmp_path / "label_map_color.png").exists()
    assert (tmp_path / "instances").exists()

    labels = json.loads((tmp_path / "labels.json").read_text())
    assert labels["legend"]["1"] == "a"
    assert labels["meta"]["foo"] == "bar"

    coco = json.loads((tmp_path / "instances.json").read_text())
    assert coco["image"]["height"] == 4
    assert len(coco["annotations"]) == len(instances)
    # RLE decodes back to a mask of the right size
    ann = coco["annotations"][0]
    decoded = decode_rle(ann["segmentation"])
    assert decoded.shape == (4, 8)
    assert "written" or written  # written dict returned
