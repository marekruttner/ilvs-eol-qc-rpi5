#!/usr/bin/env python3

import os
import glob
import json
import argparse
import gc
import platform
import urllib.parse

import torch
from ultralytics import YOLO
from PIL import Image

def normalize_path_for_docker(abs_path):
    """
    If you're using Docker on Windows with a WSL-based setup,
    convert Windows-style paths (C:\\...) to /mnt/c/... format.
    Otherwise, return unchanged.
    """
    if platform.system() == "Windows":
        drive, rest = os.path.splitdrive(abs_path)
        drive = drive.lower().replace(':', '')
        rest = rest.replace('\\', '/')
        docker_path = f"/mnt/{drive}{rest}"
        return docker_path
    return abs_path

def create_coco_annotations(images, predictions, class_names):
    """
    Build a COCO-style annotation dictionary from:
      - A list of image paths
      - Corresponding YOLO predictions (same order)
      - A list of class names

    Returns a dict that follows the COCO format:
      {
        "images": [...],
        "annotations": [...],
        "categories": [...]
      }

    You can then dump this dictionary to a JSON file, and
    import it into CVAT via "Import dataset -> COCO".
    """

    # Create a list of categories with 1-based IDs (COCO convention)
    categories = []
    for i, name in enumerate(class_names):
        categories.append({
            "id": i + 1,
            "name": name
        })

    # Prepare containers for images & annotations
    images_coco = []
    annotations_coco = []

    annotation_id = 1  # COCO annotation IDs start at 1 (arbitrary but common practice)

    for idx, (img_path, res) in enumerate(zip(images, predictions)):
        # 1) Basic info about the image
        abs_path = os.path.abspath(img_path)
        # If you plan to mount this into Docker, you might do:
        # abs_path = normalize_path_for_docker(abs_path)

        # Use PIL to get image dimensions
        with Image.open(img_path) as im:
            width, height = im.size

        images_coco.append({
            "id": idx,  # zero-based image ID
            "file_name": os.path.basename(img_path),
            "width": width,
            "height": height
        })

        # 2) For each box in YOLO results, convert to COCO annotation
        for box in res.boxes:
            cls_id = int(box.cls[0])
            # YOLO normalized xywh in 0..1
            xywhn = box.xywhn[0]  # [cx, cy, w, h]
            cx, cy, bw, bh = xywhn.tolist()

            # Convert center-based coords => top-left-based in pixel space
            x_tl = (cx - bw / 2) * width
            y_tl = (cy - bh / 2) * height
            w_box = bw * width
            h_box = bh * height

            # COCO bounding box is [x, y, width, height]
            bbox = [x_tl, y_tl, w_box, h_box]

            # Make sure category_id matches COCO's 1-based indexing
            category_id = cls_id + 1

            annotations_coco.append({
                "id": annotation_id,
                "image_id": idx,
                "category_id": category_id,
                "bbox": bbox,
                # "area" is recommended for COCO, though not strictly mandatory
                "area": w_box * h_box,
                "iscrowd": 0
            })
            annotation_id += 1

    coco_dict = {
        "images": images_coco,
        "annotations": annotations_coco,
        "categories": categories
    }
    return coco_dict

def main():
    parser = argparse.ArgumentParser(description="Pseudo-label images (YOLO) => generate COCO JSON for CVAT import.")
    # Model + data
    parser.add_argument("--model-weights", type=str, required=True,
                        help="Path to YOLO model weights (e.g. best.pt)")
    parser.add_argument("--images-dir", type=str, required=True,
                        help="Folder with images to pseudo-label")
    parser.add_argument("--classes-file", type=str, required=True,
                        help="Path to classes.txt containing class names, one per line")

    # YOLO inference settings
    parser.add_argument("--chunk-size", type=int, default=16, help="How many images to process at once")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="IoU threshold")

    # Output
    parser.add_argument("--out-json", type=str, default="pseudo_labels_coco.json",
                        help="Output filename for COCO annotations")

    args = parser.parse_args()

    # 1) Read class names
    with open(args.classes_file, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(class_names)} classes from {args.classes_file}")

    # 2) Gather images
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_images = []
    for ext in exts:
        all_images.extend(glob.glob(os.path.join(args.images_dir, f"*{ext}")))
    all_images.sort()
    if not all_images:
        raise ValueError(f"No images found in {args.images_dir}")

    print(f"Found {len(all_images)} images in {args.images_dir}")

    # 3) Load YOLO model
    model = YOLO(args.model_weights)
    print(f"Model loaded: {args.model_weights}")

    # 4) Inference in batches
    total_imgs = len(all_images)
    print(f"Running inference on {total_imgs} images (chunk_size={args.chunk_size})...")

    predictions_accum = []
    for start_idx in range(0, total_imgs, args.chunk_size):
        end_idx = start_idx + args.chunk_size
        batch_paths = all_images[start_idx:end_idx]
        print(f"Inferencing on images {start_idx + 1}..{min(end_idx, total_imgs)} of {total_imgs}.")

        # YOLO inference
        batch_results = model.predict(
            source=batch_paths,
            imgsz=args.imgsz,
            conf=args.conf_thres,
            iou=args.iou_thres,
            verbose=False,
            save=False
        )
        predictions_accum.extend(batch_results)

        # Clean up
        del batch_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5) Convert predictions to COCO format
    coco_dict = create_coco_annotations(
        images=all_images,
        predictions=predictions_accum,
        class_names=class_names
    )

    # 6) Write COCO JSON
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, indent=2)
    print(f"COCO annotations saved to {args.out_json}")

    print("Done. To use in CVAT:\n"
          "1) Create a new Task in CVAT (upload the same images)\n"
          "2) Open the task, click 'Menu' -> 'Import dataset' -> 'COCO 1.0'\n"
          f"3) Upload the file {args.out_json}.")

if __name__ == "__main__":
    main()
