"""
End-to-end submission pipeline for the Solar Filament Segmentation Challenge 2026.

Usage:
    python submission_pipeline.py --model_path checkpoints/best_model.pth \
        --model_type unet --test_dir filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images \
        --output submission.csv

Generates a Kaggle-format CSV with columns:
    filament_id, segmentation_rle
where filament_id is ``<image_id>_<idx>`` (idx starting at 1) and
segmentation_rle is a raw unquoted pycocotools RLE count string at 2048x2048.
"""
import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.augmentations import get_test_augmentations
from src.dataset import _apply_clahe, _apply_solar_limb_mask, _preprocess_image
from src.models import FilamentUNet, build_model
from src.postprocess import postprocess_masks
from src.rle_utils import encode_rle

warnings.filterwarnings("ignore")

SubmissionConfig = Dict[str, float]

DEFAULT_PP_CONFIG: SubmissionConfig = {
    "nms_iou_threshold": 0.4,
    "min_size": 150,
    "watershed_min_distance": 10,
    "watershed_threshold_rel": 0.3,
    "merge_iou_threshold": 0.7,
    "barb_reconnect": True,
    "barb_max_gap": 5,
}


def load_model(
    model_path: str,
    model_type: str = "unet",
    device: Optional[torch.device] = None,
    **model_kwargs,
):
    """Load a trained model checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "unet":
        model = FilamentUNet(**model_kwargs)
    else:
        model = build_model(model_type, **model_kwargs)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    sd = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint
    )
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    model,
    image_gray: np.ndarray,
    transform,
    device: torch.device,
    score_thr: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a single-image forward pass and return (binary masks, scores)."""
    if hasattr(model, "predict"):
        # Models exposing their own instance-head (e.g. SOLOv2, FilamentUNet)
        image_3ch = np.stack([image_gray] * 3, axis=-1)
        tensor = transform(image=image_3ch)["image"].unsqueeze(0).to(device)
        results = model.predict(tensor, score_thr=score_thr)
        masks = results[0]["masks"]
        scores = results[0]["scores"]
        if masks.size == 0:
            masks = np.zeros((0, image_gray.shape[0], image_gray.shape[1]), np.uint8)
        return masks.astype(np.uint8), np.asarray(scores, dtype=np.float64)

    # Generic semantic-head fallback (UNet produces instance+semantic logits)
    image_3ch = np.stack([image_gray] * 3, axis=-1)
    tensor = transform(image=image_3ch)["image"].unsqueeze(0).to(device)
    instance_logits, semantic_logits = model(tensor)
    instance_probs = torch.sigmoid(instance_logits)
    semantic_probs = torch.sigmoid(semantic_logits)[0, 0]
    combined = (instance_probs[0] * semantic_probs).cpu().numpy()

    masks = (combined > score_thr).astype(np.uint8)
    scores = combined.mean(axis=(1, 2))
    return masks, scores


def run_test_inference(
    model,
    test_images_dir: str,
    transform,
    device: torch.device,
    pp_config: Optional[Dict] = None,
    apply_clahe: bool = True,
    apply_limb_mask: bool = True,
    score_thr: float = 0.5,
) -> List[Dict]:
    """Run inference over a whole test-image directory."""
    image_files = sorted(
        f for f in os.listdir(test_images_dir)
        if f.lower().endswith((".jpeg", ".jpg", ".png"))
    )
    predictions = []
    for file_name in tqdm(image_files, desc="Inference"):
        path = os.path.join(test_images_dir, file_name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: failed to load {path}, skipping.")
            continue

        # Inference-time preprocessing must mirror the training loader
        prepped = _preprocess_image(image, apply_clahe, apply_limb_mask)

        raw_masks, raw_scores = predict_image(
            model, prepped, transform, device, score_thr=score_thr
        )
        masks, scores = postprocess_masks(
            raw_masks, raw_scores, prepped, pp_config
        )

        image_id = os.path.splitext(file_name)[0]
        predictions.append({
            "image_id": image_id,
            "file_name": file_name,
            "masks": masks,
            "scores": scores,
        })
    return predictions


def create_submission(predictions: List[Dict], output_path: str = "submission.csv") -> pd.DataFrame:
    """Convert per-image predictions into the final Kaggle CSV."""
    rows = []
    for pred in predictions:
        image_id = pred["image_id"]
        masks = pred["masks"]
        if masks is None or len(masks) == 0:
            continue
        for idx, mask in enumerate(masks):
            rle = encode_rle(mask)
            filament_id = f"{image_id}_{idx + 1}"  # unique index suffix, starts at 1
            rows.append({
                "filament_id": filament_id,
                "segmentation_rle": rle["counts"],
            })
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(output_path, index=False, quoting=3)  # 3 = QUOTE_NONE (raw RLE, no quotes)
    print(f"Submission saved to {output_path} ({len(df)} filaments)")
    return df


def validate_submission_format(submission_path: str, height: int = 2048, width: int = 2048) -> bool:
    """Verify the exported CSV obeys the PRD formatting rules."""
    df = pd.read_csv(submission_path)
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        print("[KO] Wrong columns:", list(df.columns))
        return False
    if df.empty:
        print("[OK] Submission is empty (no filaments predicted).")
        return True

    bad_suffix = 0
    bad_rle = 0
    sample_ok = 0
    for fid, rle_str in df.head(200).itertuples(index=False):
        # filament_id must end with a unique index suffix
        base, _, suffix = fid.rpartition("_")
        if not base or not suffix.isdigit() or int(suffix) < 1:
            bad_suffix += 1
        # RLE must decode to a 2048x2048 binary mask
        try:
            from pycocotools import mask as mask_utils
            rle = {"size": [height, width], "counts": rle_str.encode("utf-8")}
            decoded = mask_utils.decode(rle)
            if decoded.shape == (height, width):
                sample_ok += 1
            else:
                bad_rle += 1
        except Exception:
            bad_rle += 1

    print(f"[OK] {len(df)} rows validated. Sample decoded OK: {sample_ok}/200. "
          f"Bad suffixes: {bad_suffix}. Bad RLEs: {bad_rle}.")
    return bad_suffix == 0 and bad_rle == 0


def main():
    parser = argparse.ArgumentParser(description="Solar Filament Submission Pipeline")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="unet")
    parser.add_argument("--test_dir", type=str, default="filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score_thr", type=float, default=0.5)
    parser.add_argument("--no_limb_mask", action="store_true")
    parser.add_argument("--no_clahe", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model_path, args.model_type, device)
    transform = get_test_augmentations()

    pp_config = dict(DEFAULT_PP_CONFIG)
    predictions = run_test_inference(
        model,
        args.test_dir,
        transform,
        device,
        pp_config=pp_config,
        apply_clahe=not args.no_clahe,
        apply_limb_mask=not args.no_limb_mask,
        score_thr=args.score_thr,
    )
    create_submission(predictions, args.output)
    validate_submission_format(args.output)


if __name__ == "__main__":
    main()
