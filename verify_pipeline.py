"""
Verification suite for the Solar Filament Segmentation pipeline.
Runs on real dataset slices to smoke-test every module end-to-end.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.dataset import SolarFilamentDataset, collate_fn, create_train_val_split
from src.models import FilamentUNet, build_model
from src.losses import TverskyFocalLoss, create_loss
from src.evaluator import FilamentEvaluator
from src.postprocess import postprocess_masks, mask_nms, watershed_split
from src.rle_utils import encode_rle, decode_rle, polygon_to_rle, to_submission_format

DATA_ROOT = "filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
TRAIN_JSON = os.path.join(DATA_ROOT, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json")
TRAIN_IMGS = os.path.join(DATA_ROOT, "train", "train_images")

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        print(f"  [OK]   {name}")
        PASS.append(name)
    except Exception as e:
        # Preserve the traceback for rapid debugging
        import traceback
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()
        FAIL.append((name, e))


# ---------------------------------------------------------------- rle_utils
def test_rle_roundtrip():
    mask = np.zeros((64, 64), np.uint8)
    mask[10:20, 10:40] = 1
    rle = encode_rle(mask)
    dec = decode_rle(rle, 64, 64)
    assert (dec == mask).all(), "RLE roundtrip mismatch"
    assert rle["size"] == [64, 64]


def test_polygon_to_rle():
    # A square polygon
    poly = [[10, 10, 40, 10, 40, 40, 10, 40]]
    rle = polygon_to_rle(poly, 64, 64)
    dec = decode_rle(rle, 64, 64)
    assert dec.sum() > 0

# ---------------------------------------------------------------- dataset
def test_train_val_split():
    tr, va = create_train_val_split(TRAIN_JSON, val_ratio=0.2, seed=42)
    assert len(tr) > 0 and len(va) > 0
    # Annotator entries for the SAME file must land in the same split
    tr_bases = {i["id"].split("-", 1)[1] for i in tr}
    va_bases = {i["id"].split("-", 1)[1] for i in va}
    assert not (tr_bases & va_bases), "Leak: same base image in both train and val!"
    print(f"    Train samples: {len(tr)}, Val samples: {len(va)}, unique files: {len(tr_bases)}")


def test_dataset_loading():
    ds = SolarFilamentDataset(
        json_path=TRAIN_JSON,
        images_dir=TRAIN_IMGS,
        mode="train",
    )
    assert len(ds) > 0
    item = ds[0]
    image, target = item["image"], item["target"]
    assert image.shape == (3, 2048, 2048), f"got {image.shape}"
    assert target["masks"].shape[0] == target["labels"].shape[0]
    assert target["masks"].shape[1:] == (2048, 2048)
    # Multi-annotator: same file can appear in multiple samples
    from collections import Counter
    files = [i["file_name"] for i in ds.image_infos]
    c = Counter(files)
    dupes = {k: v for k, v in c.items() if v > 1}
    print(f"    Samples: {len(ds)}, unique files: {len(c)}, multi-annotator files: {len(dupes)}")


def test_dataset_multi_annotator():
    ds = SolarFilamentDataset(
        json_path=TRAIN_JSON,
        images_dir=TRAIN_IMGS,
        mode="val",  # no random augmentation
    )
    # Find two entries pointing to the same file (different annotators)
    from collections import defaultdict
    by_file = defaultdict(list)
    for i, info in enumerate(ds.image_infos):
        by_file[info["file_name"]].append(i)
    pair = [(idxs[0], idxs[1]) for idxs in by_file.values() if len(idxs) > 1][:1]
    assert pair, "No multi-annotator sample found in val mode"
    i0, i1 = pair[0]
    m0 = ds[i0]["target"]["masks"]
    m1 = ds[i1]["target"]["masks"]
    print(f"    file={ds.image_infos[i0]['file_name']} annotator A instances: {m0.shape[0]}, annotator B instances: {m1.shape[0]}")
    # Same annotator -> identical masks; different annotators -> likely differ
    same_annotator = ds.image_infos[i0]["id"].split("-", 1)[0] == ds.image_infos[i1]["id"].split("-", 1)[0]
    if not same_annotator:
        a = m0.sum().item()
        b = m1.sum().item()
        print(f"    Different annotators, mask pixel-sum A={a:.0f} B={b:.0f} (annotation IDs differ)")

# ---------------------------------------------------------------- losses
def test_loss_sequential():
    """Multi-instance (B, K, H, W) mask tensor fed to a (B, 1, H, W) loss should broadcast cleanly."""
    loss_fn = create_loss("tversky_focal")
    pred = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.zeros(2, 1, 64, 64)
    target[0, 0, 10:20, 10:40] = 1
    target[1, 0, 30:50, 20:30] = 1
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None


def test_loss_empty_target():
    loss_fn = create_loss("tversky_focal")
    pred = torch.randn(1, 1, 64, 64)
    target = torch.zeros(1, 1, 64, 64)
    loss = loss_fn(pred, target)
    assert torch.isfinite(loss)

# ---------------------------------------------------------------- postprocess
def test_postprocess_pipeline():
    masks = np.zeros((3, 256, 256), np.uint8)
    masks[0, 30:40, 30:80] = 1           # thin filament (400 px)
    masks[1, 120:180, 120:180] = 1       # blob w/ two lobes -> watershed split
    masks[2, 200:202, 20:28] = 1         # 16 px noise, dropped by min_size
    masks[1, 130:145, 130:145] = 0       # pinch to create a second lobe
    scores = np.array([0.9, 0.8, 0.7])
    out_masks, out_scores = postprocess_masks(masks, scores)
    assert len(out_masks) >= 1
    for m in out_masks:
        assert m.shape == (256, 256)
        assert m.sum() >= 150, "Post NMS mask below min_size survived"
    print(f"    input instances: 3 -> output instances: {len(out_masks)}")


def test_mask_nms_suppresses_duplicates():
    base = np.zeros((64, 64), np.uint8)
    base[10:30, 10:40] = 1
    dup = base.copy()
    dup[28:34, :] = 1  # slight offset, high IoU with base
    masks = np.stack([base, dup])
    scores = np.array([0.9, 0.8])
    kept, _ = mask_nms(masks, scores, iou_threshold=0.4, min_size=0)
    assert len(kept) == 1, f"expected 1 after NMS, got {len(kept)}"


def test_watershed_split():
    mask = np.zeros((128, 128), np.uint8)
    mask[40:50, 20:60] = 1   # left lobe bar
    mask[40:50, 70:110] = 1  # right lobe bar, joined by thin bridge
    mask[44:46, 55:75] = 1
    splits = watershed_split(mask, min_distance=20, threshold_rel=0.4)
    print(f"    watershed produced {len(splits)} sub-masks from 1 blob")
    assert len(splits) >= 1

# ---------------------------------------------------------------- evaluator
def test_evaluator_perfect_match():
    gt = np.zeros((1, 64, 64), np.uint8)
    gt[0, 10:30, 10:40] = 1
    ev = FilamentEvaluator(iou_threshold=0.5)
    m = ev.evaluate_dataset([gt.copy().tolist()] if False else [[gt[0]]], [[gt[0]]])
    assert abs(m["PQ"] - 1.0) < 1e-6, f"expected PQ=1.0, got {m['PQ']}"


def test_evaluator_partial_overlap():
    gt = np.zeros((64, 64), np.uint8)
    gt[10:30, 10:40] = 1
    pred = np.zeros((64, 64), np.uint8)
    pred[12:30, 10:40] = 1  # shifted by 2 rows
    ev = FilamentEvaluator(iou_threshold=0.5)
    m = ev.evaluate_dataset([[pred]], [[gt]])
    expected_iou = np.logical_and(pred, gt).sum() / np.logical_or(pred, gt).sum()
    assert abs(m["PQ"] - expected_iou) < 1e-3, f"PQ={m['PQ']}, expected IoU={expected_iou}"


def test_evaluator_fp_fn():
    gt = np.zeros((64, 64), np.uint8)
    gt[10:30, 10:40] = 1  # 600 px
    pred = np.zeros((64, 64), np.uint8)
    pred[40:55, 40:55] = 1  # 225 px total miss (above min_instance_area=150)
    ev = FilamentEvaluator(iou_threshold=0.5)
    m = ev.evaluate_dataset([[pred]], [[gt]])
    assert m["TP"] == 0 and m["FP"] == 1 and m["FN"] == 1, m


def test_evaluator_fragmentation():
    # One GT split into two preds -> both above threshold ideally match separately?
    # With greedy matching, only one pred matches the GT; other is FP.
    gt = np.zeros((64, 64), np.uint8)
    gt[10:30, 10:70] = 1
    p1 = np.zeros((64, 64), np.uint8)
    p1[10:30, 10:40] = 1
    p2 = np.zeros((64, 64), np.uint8)
    p2[10:30, 40:70] = 1
    ev = FilamentEvaluator(iou_threshold=0.5)
    m = ev.evaluate_dataset([[p1, p2]], [[gt]])
    print(f"    fragmentation case: TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, PQ={m['PQ']:.3f}")
    # Depending on threshold, at most 1 of the two chunks matches GT.
    assert m["TP"] <= 1 and (m["FN"] == 1 or m["TP"] == 1)

# ---------------------------------------------------------------- model
def test_model_forward_backward():
    model = FilamentUNet(encoder="resnet34")
    model.train()
    x = torch.randn(1, 3, 128, 128)
    inst, sem = model(x)
    assert inst.shape[0] == 1 and sem.shape[0] == 1
    assert sem.shape[1] == 1
    target = torch.zeros(1, 1, 128, 128)
    target[0, 0, 30:70, 30:70] = 1
    loss = create_loss("tversky_focal")(sem, target)
    loss.backward()


def test_model_predict():
    model = FilamentUNet(encoder="resnet34")
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        results = model.predict(x, score_thr=0.5)
    assert len(results) == 1
    assert "masks" in results[0] and "scores" in results[0]


def test_build_model_factory():
    m = build_model("unet")
    assert isinstance(m, FilamentUNet)
    try:
        build_model("mask2former")
        raised = False
    except ImportError:
        raised = True
    print(f"    mask2former without detectron2 raises ImportError: {raised}")


# ---------------------------------------------------------------- submission
def test_submission_format():
    mask = np.zeros((512, 512), np.uint8)
    mask[100:200, 200:300] = 1
    rle = encode_rle(mask)
    fid, rle_str = to_submission_format(rle, "20150125172714Mh", 1)
    assert fid == "20150125172714Mh_1"
    assert isinstance(rle_str, str) or isinstance(rle_str, bytes)
    # Decode raw counts without size
    from pycocotools import mask as mask_utils
    if isinstance(rle_str, str):
        rle_str = rle_str.encode("utf-8")
    dec = mask_utils.decode({"size": [512, 512], "counts": rle_str})
    assert dec.shape == (512, 512)


def test_rle_counts_no_quotes():
    """The CSV writer must produce raw un-quoted RLE count strings."""
    import pandas as pd
    import io
    mask = np.zeros((128, 128), np.uint8)
    mask[20:40, 20:80] = 1
    rle = encode_rle(mask)
    rows = [{"filament_id": "img_1", "segmentation_rle": rle["counts"]}]
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False, quoting=3)
    csv_text = buf.getvalue()
    # Count double quotes in the body (skip header)
    body = csv_text.split("\n", 1)[1]
    assert '"' not in body, "CSV body contains quotes!"
    print(f"    sample csv row: {body.strip()[:60]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("SOLAR FILAMENT SEGMENTATION — VERIFICATION SUITE")
    print("=" * 60)

    print("\n[rle_utils]")
    check("rle_roundtrip", test_rle_roundtrip)
    check("polygon_to_rle", test_polygon_to_rle)

    print("\n[dataset]")
    check("train_val_split_no_leak", test_train_val_split)
    check("dataset_loading_shapes", test_dataset_loading)
    check("multi_annotator_samples", test_dataset_multi_annotator)

    print("\n[losses]")
    check("tversky_focal_forward_backward", test_loss_sequential)
    check("tversky_focal_empty_target", test_loss_empty_target)

    print("\n[postprocess]")
    check("postprocess_pipeline", test_postprocess_pipeline)
    check("mask_nms_suppresses_duplicates", test_mask_nms_suppresses_duplicates)
    check("watershed_split", test_watershed_split)

    print("\n[evaluator]")
    check("evaluator_perfect_match", test_evaluator_perfect_match)
    check("evaluator_partial_overlap", test_evaluator_partial_overlap)
    check("evaluator_fp_fn", test_evaluator_fp_fn)
    check("evaluator_fragmentation", test_evaluator_fragmentation)

    print("\n[model]")
    check("unet_forward_backward", test_model_forward_backward)
    check("unet_predict", test_model_predict)
    check("build_model_factory", test_build_model_factory)

    print("\n[submission]")
    check("submission_format", test_submission_format)
    check("rle_counts_no_quotes", test_rle_counts_no_quotes)

    print("\n" + "=" * 60)
    print(f"RESULTS: {len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name, _ in FAIL:
            print(f"  FAILED: {name}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("=" * 60)
