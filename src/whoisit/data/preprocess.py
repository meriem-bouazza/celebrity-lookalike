"""Align all images of a metadata CSV once and store them as JPEG (task 4.1).

Input:  CSV with columns ``image_path,identity`` (paths absolute or relative to ``--image-root``).
Output in ``--out-dir``:
    images/<identity>/<file>.jpg   aligned 112 x 112 faces
    meta.csv     image_path (relative to out-dir), identity, source_path, confidence, 5 landmarks
    failed.csv   source_path, identity, reason ("no_face" or "unreadable: ...")

The script can be interrupted and started again: images already listed in meta.csv or
failed.csv are skipped.

Usage:
    python -m whoisit.data.preprocess --metadata data/lfw_meta.csv --out-dir data/aligned/lfw
"""

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from whoisit.data.align import align_face, detect_largest_face, load_image

LANDMARK_COLUMNS = [f"{axis}{i}" for i in range(1, 6) for axis in ("lm_x", "lm_y")]
META_COLUMNS = ["image_path", "identity", "source_path", "confidence", *LANDMARK_COLUMNS]
FAILED_COLUMNS = ["source_path", "identity", "reason"]


def _read_rows(path: Path) -> list[dict]:
    """Read a CSV as list of dicts; empty list if the file does not exist."""
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _open_for_append(path: Path, columns: list[str]):
    """Open a CSV for appending and write the header if the file is new.

    Line buffering (buffering=1) writes every row to disk immediately, so an interrupted
    run loses at most the image that was being processed.
    """
    is_new = not path.exists()
    f = open(path, "a", newline="", encoding="utf-8", buffering=1)
    writer = csv.DictWriter(f, fieldnames=columns)
    if is_new:
        writer.writeheader()
    return f, writer


def preprocess(
    metadata: Path,
    out_dir: Path,
    image_root: Path | None = None,
    min_confidence: float = 0.9,
    device: str = "cpu",
    jpeg_quality: int = 95,
) -> dict:
    """Detect, align and save every image listed in ``metadata``.

    Args:
        metadata: CSV with columns ``image_path,identity``.
        out_dir: Output folder for images/, meta.csv and failed.csv.
        image_root: Base folder for relative ``image_path`` values (None = as given).
        min_confidence: Minimum MTCNN face probability.
        device: "cpu" or "cuda".
        jpeg_quality: JPEG quality of the stored crops (0-100).

    Returns:
        Dict with counts ``total``, ``aligned``, ``failed``, ``new`` (processed in this call)
        and ``detection_rate`` (aligned / total).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path, failed_path = out_dir / "meta.csv", out_dir / "failed.csv"

    rows = _read_rows(metadata)
    if rows and not {"image_path", "identity"} <= rows[0].keys():
        raise ValueError(f"{metadata} needs the columns image_path,identity")

    # Resume: everything that already has a result is skipped.
    meta_rows = _read_rows(meta_path)
    done = {r["source_path"] for r in meta_rows + _read_rows(failed_path)}
    used_names = {r["image_path"] for r in meta_rows}
    todo = [r for r in rows if r["image_path"] not in done]

    meta_file, meta_writer = _open_for_append(meta_path, META_COLUMNS)
    failed_file, failed_writer = _open_for_append(failed_path, FAILED_COLUMNS)
    try:
        for row in tqdm(todo, desc="aligning", unit="img"):
            source, identity = row["image_path"], row["identity"]
            full_path = Path(source) if image_root is None else image_root / source

            try:
                image = load_image(full_path)
            except Exception as error:  # broken or missing file: record it, keep going
                failed_writer.writerow({"source_path": source, "identity": identity,
                                        "reason": f"unreadable: {error}"})
                continue

            face = detect_largest_face(image, min_confidence=min_confidence, device=device)
            if face is None:
                failed_writer.writerow({"source_path": source, "identity": identity, "reason": "no_face"})
                continue
            landmarks, confidence = face

            # Output name: images/<identity>/<original file name>.jpg
            relative = Path("images") / identity / (Path(source).stem + ".jpg")
            if relative.as_posix() in used_names:
                raise ValueError(f"two input images of '{identity}' map to {relative}; rename one")
            used_names.add(relative.as_posix())
            # A file that exists but is not in meta.csv is a leftover of an interrupted run
            # and is simply overwritten.
            target = out_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            aligned = align_face(image, landmarks)
            cv2.imwrite(str(target), cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

            meta_writer.writerow({
                "image_path": relative.as_posix(),
                "identity": identity,
                "source_path": source,
                "confidence": f"{confidence:.4f}",
                **{col: f"{value:.2f}" for col, value in zip(LANDMARK_COLUMNS, landmarks.flatten())},
            })
    finally:
        meta_file.close()
        failed_file.close()

    aligned_count = len(_read_rows(meta_path))
    failed_count = len(_read_rows(failed_path))
    total = aligned_count + failed_count
    return {
        "total": total,
        "aligned": aligned_count,
        "failed": failed_count,
        "new": len(todo),
        "detection_rate": aligned_count / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Align all images of a metadata CSV (image_path,identity).")
    parser.add_argument("--metadata", type=Path, required=True, help="input CSV with image_path,identity")
    parser.add_argument("--out-dir", type=Path, required=True, help="output folder")
    parser.add_argument("--image-root", type=Path, default=None, help="base folder for relative image paths")
    parser.add_argument("--min-confidence", type=float, default=0.9, help="minimum MTCNN probability")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    stats = preprocess(args.metadata, args.out_dir, args.image_root, args.min_confidence, args.device)
    print(f"processed now: {stats['new']}")
    print(f"aligned: {stats['aligned']} / {stats['total']}  failed: {stats['failed']}")
    print(f"detection rate: {stats['detection_rate']:.2%}")


if __name__ == "__main__":
    main()
