import os
import json
import shutil
import random
import yaml
import glob

from ultralytics import YOLO


# -----------------------------
# 1. DATA PREPARATION FUNCTION
# -----------------------------
def prepare_dataset_structure(
        images_dir: str,
        labels_dir: str,
        classes_file: str,
        notes_file: str,
        output_dir: str = "dataset_yolo",
        val_ratio: float = 0.2,
        test_ratio: float = 0.0
):
    """
    Prepares a YOLO-style dataset folder with train/val(/test) splits.

    Args:
        images_dir (str): Path to the folder containing all images.
        labels_dir (str): Path to the folder containing all label TXT files.
        classes_file (str): Path to classes.txt (one class per line).
        notes_file (str): Path to notes.json (optional usage).
        output_dir (str): Output folder for the structured dataset.
        val_ratio (float): Fraction of data to reserve for validation.
        test_ratio (float): Fraction of data to reserve for testing.
            If you don't want a test set, leave it at 0.0.
    Returns:
        data_yaml_path (str): Path to the generated data.yaml file for YOLO training.
    """

    # 1) Read classes
    with open(classes_file, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]
    num_classes = len(class_names)
    print(f"Found {num_classes} classes in {classes_file}.")

    # 2) Optionally read notes.json (if needed)
    if os.path.exists(notes_file):
        with open(notes_file, "r") as f:
            notes_data = json.load(f)
        print(f"Loaded notes.json with keys: {list(notes_data.keys())}")
    else:
        print(f"notes.json not found at {notes_file}, continuing without it.")

    # 3) Gather all image paths
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    all_images.sort()

    if not all_images:
        raise ValueError(f"No images found in {images_dir} with extensions {image_extensions}")

    # 4) Shuffle and split data
    random.shuffle(all_images)
    total_count = len(all_images)
    val_count = int(val_ratio * total_count)
    test_count = int(test_ratio * total_count)

    train_images = all_images[: (total_count - val_count - test_count)]
    val_images = all_images[(total_count - val_count - test_count): (total_count - test_count)]
    test_images = all_images[(total_count - test_count):]

    print(f"Data split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

    # 5) Create output YOLO directories
    subsets = []
    if train_images:
        subsets.append("train")
    if val_images:
        subsets.append("val")
    if test_images:
        subsets.append("test")

    for subset in subsets:
        os.makedirs(os.path.join(output_dir, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, "labels"), exist_ok=True)

    # 6) Helper to copy images and labels
    def copy_images_and_labels(image_list, subset_name):
        for img_path in image_list:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, base_name + ".txt")

            dest_img_path = os.path.join(output_dir, subset_name, "images", os.path.basename(img_path))
            shutil.copy2(img_path, dest_img_path)

            if os.path.exists(label_path):
                dest_label_path = os.path.join(output_dir, subset_name, "labels", os.path.basename(label_path))
                shutil.copy2(label_path, dest_label_path)
            else:
                # YOLO requires a label file per image. Create an empty file if none exists.
                with open(os.path.join(output_dir, subset_name, "labels", base_name + ".txt"), "w"):
                    pass

    # Copy data into the new structure
    if train_images:
        copy_images_and_labels(train_images, "train")
    if val_images:
        copy_images_and_labels(val_images, "val")
    if test_images:
        copy_images_and_labels(test_images, "test")

    # 7) Use absolute paths in data.yaml
    train_path = os.path.abspath(os.path.join(output_dir, "train", "images")) if train_images else None
    val_path = os.path.abspath(os.path.join(output_dir, "val", "images")) if val_images else None
    test_path = os.path.abspath(os.path.join(output_dir, "test", "images")) if test_images else None

    data_dict = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": num_classes,
        "names": class_names,
    }

    data_yaml_path = os.path.join(output_dir, "data.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_dict, f)

    print(f"Created {data_yaml_path}")
    return data_yaml_path


# -----------------------------
# 2. TRAINING FUNCTION (YOLOv8)
# -----------------------------
def train_yolov8(
        data_yaml: str,
        model_arch: str = "yolov8n.pt",
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        project_name: str = "runs",
        run_name: str = "train_yolov8"
):
    """
    Trains a YOLOv8 model using the Ultralytics library.

    Args:
        data_yaml (str): Path to the data.yaml file.
        model_arch (str): Base model architecture, e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'.
        epochs (int): Number of epochs.
        imgsz (int): Image resolution.
        batch (int): Batch size.
        project_name (str): Where to save results (folder).
        run_name (str): Subfolder name for this run.
    """
    # Load the pretrained YOLOv8 model
    model = YOLO(model_arch)
    # Start training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_name,
        name=run_name,
        #device=0,       # <---- Force GPU 0. Use 'device=0' or 'device=\"cuda:0\"'
        optimizer='Adam',
        lr0=0.01
    )
    print(f"Training finished. Results saved to: {results}")


# -----------------------------
# 3. MAIN (PUT IT ALL TOGETHER)
# -----------------------------
if __name__ == "__main__":
    # 1) Paths to your raw dataset
    IMAGES_DIR = "ilvs-eol-qc-annoted/images"
    LABELS_DIR = "ilvs-eol-qc-annoted/labels"
    CLASSES_FILE = "ilvs-eol-qc-annoted/classes.txt"
    NOTES_FILE = "ilvs-eol-qc-annoted/notes.json"

    # 2) Prepare the dataset
    DATA_YOLO_DIR = "dataset_yolo"
    data_yaml_path = prepare_dataset_structure(
        images_dir=IMAGES_DIR,
        labels_dir=LABELS_DIR,
        classes_file=CLASSES_FILE,
        notes_file=NOTES_FILE,
        output_dir=DATA_YOLO_DIR,
        val_ratio=0.2,  # 20% val
        test_ratio=0.1  # 10% test
    )

    # 3) Train YOLOv8 with GPU
    train_yolov8(
        data_yaml=data_yaml_path,
        model_arch="yolov8s.pt",  # Switch to yolo8s for better accuracy than nano
        epochs=50,
        imgsz=640,
        batch=16,
        project_name="runs",
        run_name="train_yolov8"
    )
