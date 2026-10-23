"""Discover State Farm files, validate metadata/images, and write driver-isolated splits."""
import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from config import DATA_DIR, RANDOM_SEED
from src.dataset import describe_splits, discover_dataset, grouped_driver_split, make_manifest
from src.utils import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "raw", help="Root containing extracted labelled class folders and driver_imgs_list.csv")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--download", action="store_true", help="Download the competition archive with the Kaggle CLI, then extract it")
    args = parser.parse_args()
    if args.download:
        if shutil.which("kaggle") is None:
            parser.error("Kaggle CLI is missing. Install it with pip install kaggle, configure credentials, and accept competition rules first.")
        args.data_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["kaggle", "competitions", "download", "-c", "state-farm-distracted-driver-detection", "-p", str(args.data_dir)], capture_output=True, text=True)
        if result.returncode:
            parser.error("Kaggle download failed. Verify credentials, network access, and that you accepted the competition rules on Kaggle.")
        pending_archives = list(args.data_dir.rglob("*.zip"))
        extracted_archives = set()
        root = args.data_dir.resolve()
        while pending_archives:
            archive = pending_archives.pop(0).resolve()
            if archive in extracted_archives or not archive.is_file():
                continue
            extracted_archives.add(archive)
            try:
                with zipfile.ZipFile(archive) as bundle:
                    for member in bundle.infolist():
                        destination = (archive.parent / member.filename).resolve()
                        if destination != root and root not in destination.parents:
                            parser.error(f"Archive contains an unsafe path: {member.filename!r}")
                    bundle.extractall(archive.parent)
            except zipfile.BadZipFile as exc:
                parser.error(f"Downloaded archive is invalid: {archive.name}: {exc}")
            pending_archives.extend(p for p in args.data_dir.rglob("*.zip") if p.resolve() not in extracted_archives)
        print("Downloaded and safely extracted competition archives, including nested image archives.")
    try:
        images, metadata = discover_dataset(args.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Found {len(images)} labelled images and official metadata at {metadata}.")
    try:
        manifest = make_manifest(images, metadata)
        manifest = grouped_driver_split(manifest, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    split_dir = DATA_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        output = split_dir / f"{split}.csv"
        manifest.loc[manifest.split == split].to_csv(output, index=False)
        print(f"Wrote {output}")
    combined = split_dir / "manifest.csv"
    manifest.to_csv(combined, index=False)
    stats = {"seed": args.seed, "requested_driver_proportions": {"train": 0.70, "validation": 0.15, "test": 0.15}, "actual_splits": describe_splits(manifest), "num_images": len(manifest), "num_drivers": manifest.driver_id.nunique(), "metadata_path": str(metadata.resolve())}
    save_json(DATA_DIR / "processed" / "dataset_stats.json", stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
