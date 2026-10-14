"""
FACIAL PALSY SEVERITY CLASSIFICATION - TRAINING
================================================

Leakage-safe training pipeline.

Dataset:
    outputs/train.csv
    outputs/val.csv
    outputs/test.csv

Classes:
    0 -> Mild
    1 -> Moderate
    2 -> Moderate Severe
    3 -> Severe

IMPORTANT:
    dataset.py is responsible for leakage-safe case-group splitting.

    This script:
        - loads the generated CSV files
        - uses identical preprocessing for all splits
        - applies augmentation ONLY to training images
        - uses class-weighted CrossEntropyLoss
        - uses a pretrained ResNet18
        - fine-tunes layer3/layer4 + classifier
        - monitors validation Macro F1
        - saves the best model
        - evaluates on the untouched test set
        - generates classification report
        - generates confusion matrix
        - saves training history
"""

from __future__ import annotations

import os
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"

TRAIN_CSV = OUTPUT_DIR / "train.csv"
VAL_CSV = OUTPUT_DIR / "val.csv"
TEST_CSV = OUTPUT_DIR / "test.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------

CLASSES = [
    "Mild",
    "Moderate",
    "Moderate Severe",
    "Severe",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASSES)
}

NUM_CLASSES = len(CLASSES)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

IMAGE_SIZE = 224

BATCH_SIZE = 8

NUM_EPOCHS = 15

# Backbone is deliberately conservative.
BACKBONE_LR = 1e-5

# Classifier learns faster.
CLASSIFIER_LR = 1e-4

WEIGHT_DECAY = 1e-4

# Early stopping.
PATIENCE = 4

# Number of DataLoader workers.
# Windows can be problematic with many workers, so keep this conservative.
NUM_WORKERS = 0

PIN_MEMORY = torch.cuda.is_available()

# Gradient clipping.
GRADIENT_CLIP = 1.0


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# IMAGE FILE VALIDATION
# ============================================================================

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ============================================================================
# DATASET
# ============================================================================

class FacialPalsyDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.transform = transform

        if self.df.empty:
            raise RuntimeError(
                "Dataset is empty."
            )

        # Verify required columns.
        required_columns = {
            "image_path",
            "severity",
        }

        missing = (
            required_columns
            - set(self.df.columns)
        )

        if missing:
            raise RuntimeError(
                f"Missing required CSV columns: "
                f"{missing}"
            )

        # Validate labels.
        invalid_labels = set(
            self.df["severity"].unique()
        ) - set(CLASSES)

        if invalid_labels:
            raise RuntimeError(
                f"Unknown severity labels found: "
                f"{invalid_labels}"
            )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = Path(
            row["image_path"]
        )

        label_name = row["severity"]

        label = CLASS_TO_INDEX[
            label_name
        ]

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as exc:

            raise RuntimeError(
                f"Could not load image:\n"
                f"{image_path}\n"
                f"Error: {exc}"
            )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return image, label


# ============================================================================
# TRANSFORMS
# ============================================================================

def create_transforms():

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Validation and test DO NOT receive random augmentation.
    #
    # Training gets only mild geometric/photometric augmentation.
    #
    # Pre-generated augmentations already exist in the training CSV, but
    # those images are treated as ordinary training samples here.
    # ------------------------------------------------------------------------

    train_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        # Mild online augmentation.
        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=8
        ),

        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    eval_transform = transforms.Compose([

        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])

    return (
        train_transform,
        eval_transform,
    )


# ============================================================================
# LOAD CSV
# ============================================================================

def load_dataframe(
    csv_path: Path,
    split_name: str,
) -> pd.DataFrame:

    if not csv_path.exists():

        raise FileNotFoundError(
            f"{split_name} CSV not found:\n"
            f"{csv_path}\n\n"
            f"Run dataset.py first."
        )

    df = pd.read_csv(
        csv_path
    )

    if df.empty:

        raise RuntimeError(
            f"{split_name} dataset is empty:\n"
            f"{csv_path}"
        )

    # ------------------------------------------------------------------------
    # Verify split
    # ------------------------------------------------------------------------

    if "split" in df.columns:

        actual_splits = set(
            df["split"]
            .dropna()
            .unique()
        )

        if split_name == "train":

            if actual_splits != {"train"}:

                print(
                    f"WARNING: train.csv contains "
                    f"split values: {actual_splits}"
                )

        elif split_name == "validation":

            if actual_splits != {"validation"}:

                print(
                    f"WARNING: val.csv contains "
                    f"split values: {actual_splits}"
                )

        elif split_name == "test":

            if actual_splits != {"test"}:

                print(
                    f"WARNING: test.csv contains "
                    f"split values: {actual_splits}"
                )

    # ------------------------------------------------------------------------
    # Verify image paths
    # ------------------------------------------------------------------------

    missing_paths = []

    for path in df["image_path"]:

        if not Path(path).exists():

            missing_paths.append(path)

            if len(missing_paths) >= 10:
                break

    if missing_paths:

        raise FileNotFoundError(
            "Some image paths do not exist.\n"
            f"Examples:\n"
            + "\n".join(
                missing_paths
            )
        )

    return df


# ============================================================================
# DATASET SUMMARY
# ============================================================================

def print_dataset_summary(
    train_df,
    val_df,
    test_df,
):

    print()
    print("=" * 78)
    print("DATASET SUMMARY")
    print("=" * 78)

    for name, df in [
        ("TRAIN", train_df),
        ("VALIDATION", val_df),
        ("TEST", test_df),
    ]:

        print()
        print(
            f"{name}: {len(df):,} images"
        )

        counts = (
            df["severity"]
            .value_counts()
            .reindex(
                CLASSES,
                fill_value=0,
            )
        )

        for cls in CLASSES:

            print(
                f"  {cls:<18} "
                f"{int(counts[cls]):>6}"
            )


# ============================================================================
# CLASS WEIGHTS
# ============================================================================

def calculate_class_weights(
    train_df: pd.DataFrame,
):

    counts = (
        train_df["severity"]
        .value_counts()
        .reindex(
            CLASSES,
            fill_value=0,
        )
        .astype(float)
    )

    if (counts <= 0).any():

        raise RuntimeError(
            "One or more classes have zero "
            "training samples."
        )

    # ------------------------------------------------------------------------
    # Effective inverse-frequency weighting.
    #
    # sqrt(N / class_count) is intentionally used instead of pure
    # inverse frequency because Severe is substantially smaller than
    # Moderate/Moderate Severe.
    #
    # Pure inverse frequency would make the minority classes excessively
    # influential.
    # ------------------------------------------------------------------------

    total = counts.sum()

    weights = np.sqrt(
        total / counts
    )

    # Normalize mean weight to 1.
    weights = weights / weights.mean()

    weights_tensor = torch.tensor(
        weights.values,
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print("=" * 78)
    print("CLASS WEIGHTS")
    print("=" * 78)

    for cls, count, weight in zip(
        CLASSES,
        counts.values,
        weights.values,
    ):

        print(
            f"{cls:<18} "
            f"count={int(count):>6} "
            f"weight={weight:.4f}"
        )

    return weights_tensor


# ============================================================================
# MODEL
# ============================================================================

class FacialPalsyResNet18(
    nn.Module
):

    def __init__(self):

        super().__init__()

        print()
        print(
            "Loading pretrained ResNet18..."
        )

        try:

            weights = (
                models.ResNet18_Weights.DEFAULT
            )

            backbone = models.resnet18(
                weights=weights
            )

        except Exception:

            print(
                "WARNING: pretrained weights "
                "could not be loaded."
            )

            print(
                "Using randomly initialized "
                "ResNet18 instead."
            )

            backbone = models.resnet18(
                weights=None
            )

        # Keep the feature extractor.
        self.backbone = backbone

        # Replace classifier.
        in_features = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Sequential(

            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                in_features,
                NUM_CLASSES,
            ),
        )

        # --------------------------------------------------------------------
        # Freeze everything initially.
        # --------------------------------------------------------------------

        for parameter in (
            self.backbone.parameters()
        ):

            parameter.requires_grad = False

        # --------------------------------------------------------------------
        # Fine-tune layer3 + layer4.
        # --------------------------------------------------------------------

        for parameter in (
            self.backbone.layer3.parameters()
        ):

            parameter.requires_grad = True

        for parameter in (
            self.backbone.layer4.parameters()
        ):

            parameter.requires_grad = True

        # Classifier.
        for parameter in (
            self.backbone.fc.parameters()
        ):

            parameter.requires_grad = True

    def forward(self, x):

        return self.backbone(x)


# ============================================================================
# OPTIMIZER
# ============================================================================

def create_optimizer(
    model: nn.Module,
):

    backbone_parameters = []
    classifier_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if name.startswith(
            "backbone.fc"
        ):

            classifier_parameters.append(
                parameter
            )

        else:

            backbone_parameters.append(
                parameter
            )

    optimizer = torch.optim.AdamW(

        [
            {
                "params":
                    backbone_parameters,
                "lr":
                    BACKBONE_LR,
            },

            {
                "params":
                    classifier_parameters,
                "lr":
                    CLASSIFIER_LR,
            },
        ],

        weight_decay=WEIGHT_DECAY,
    )

    return optimizer


# ============================================================================
# SCHEDULER
# ============================================================================

def create_scheduler(
    optimizer,
):

    # Reduce LR if validation Macro F1 stops improving.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=1,

        min_lr=1e-7,
    )

    return scheduler


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    y_true,
    y_pred,
):

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(
                range(NUM_CLASSES)
            ),
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy":
            balanced_accuracy,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


# ============================================================================
# TRAIN ONE EPOCH
# ============================================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
):

    model.train()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    start_time = time.time()

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels,
        )

        loss.backward()

        # Gradient clipping.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP,
        )

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs
            .argmax(dim=1)
        )

        all_targets.extend(
            labels.detach()
            .cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    elapsed = (
        time.time()
        - start_time
    )

    return (
        epoch_loss,
        metrics,
        elapsed,
    )


# ============================================================================
# VALIDATE
# ============================================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    start_time = time.time()

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels,
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs
            .argmax(dim=1)
        )

        all_targets.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    elapsed = (
        time.time()
        - start_time
    )

    return (
        epoch_loss,
        metrics,
        elapsed,
    )


# ============================================================================
# PRINT PER-CLASS METRICS
# ============================================================================

def print_per_class_metrics(
    metrics,
):

    print()
    print(
        "Validation Per-Class Metrics:"
    )

    for i, cls in enumerate(
        CLASSES
    ):

        print(
            f"  {cls:<18} "
            f"P={metrics['precision'][i]:.3f} "
            f"R={metrics['recall'][i]:.3f} "
            f"F1={metrics['f1'][i]:.3f} "
            f"N={int(metrics['support'][i])}"
        )


# ============================================================================
# LEARNING RATE
# ============================================================================

def get_learning_rates(
    optimizer,
):

    if len(
        optimizer.param_groups
    ) >= 2:

        return (
            optimizer.param_groups[0]["lr"],
            optimizer.param_groups[1]["lr"],
        )

    lr = optimizer.param_groups[0]["lr"]

    return lr, lr


# ============================================================================
# SAVE CHECKPOINT
# ============================================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    val_macro_f1,
    history,
):

    checkpoint_path = (
        MODEL_DIR
        / "best_model.pth"
    )

    torch.save(

        {
            "epoch":
                epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "val_macro_f1":
                val_macro_f1,

            "classes":
                CLASSES,

            "class_to_index":
                CLASS_TO_INDEX,

            "image_size":
                IMAGE_SIZE,

            "history":
                history,
        },

        checkpoint_path,
    )

    return checkpoint_path


# ============================================================================
# SAVE TRAINING HISTORY
# ============================================================================

def save_history(
    history,
):

    history_df = pd.DataFrame(
        history
    )

    path = (
        OUTPUT_DIR
        / "training_history.csv"
    )

    history_df.to_csv(
        path,
        index=False,
    )

    return path


# ============================================================================
# LOAD BEST MODEL
# ============================================================================

def load_best_model(
    model,
):

    checkpoint_path = (
        MODEL_DIR
        / "best_model.pth"
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Best model not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    # Support our checkpoint format.
    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        best_epoch = checkpoint.get(
            "epoch",
            None,
        )

        best_f1 = checkpoint.get(
            "val_macro_f1",
            None,
        )

    else:

        # Fallback if checkpoint is a raw state dict.
        model.load_state_dict(
            checkpoint
        )

        best_epoch = None
        best_f1 = None

    return (
        model,
        best_epoch,
        best_f1,
    )


# ============================================================================
# FINAL TEST EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_test(
    model,
    loader,
    criterion,
):

    model.eval()

    running_loss = 0.0

    all_targets = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels,
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = (
            outputs
            .argmax(dim=1)
        )

        all_targets.extend(
            labels.cpu()
            .numpy()
            .tolist()
        )

        all_predictions.extend(
            predictions.cpu()
            .numpy()
            .tolist()
        )

    test_loss = (
        running_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_targets,
        all_predictions,
    )

    return (
        test_loss,
        metrics,
        np.array(all_targets),
        np.array(all_predictions),
    )


# ============================================================================
# SAVE CLASSIFICATION REPORT
# ============================================================================

def save_classification_report(
    y_true,
    y_pred,
):

    report = classification_report(

        y_true,

        y_pred,

        labels=list(
            range(NUM_CLASSES)
        ),

        target_names=CLASSES,

        digits=4,

        zero_division=0,
    )

    path = (
        OUTPUT_DIR
        / "classification_report.txt"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    print()
    print(
        "Classification report:"
    )
    print()
    print(report)

    print(
        f"✓ Saved: {path}"
    )

    return path


# ============================================================================
# SAVE CONFUSION MATRIX
# ============================================================================

def save_confusion_matrix(
    y_true,
    y_pred,
):

    cm = confusion_matrix(

        y_true,

        y_pred,

        labels=list(
            range(NUM_CLASSES)
        ),
    )

    figure = plt.figure(
        figsize=(8, 7)
    )

    plt.imshow(
        cm,
        interpolation="nearest",
    )

    plt.title(
        "Facial Palsy Severity - Test Confusion Matrix"
    )

    plt.colorbar()

    tick_marks = np.arange(
        NUM_CLASSES
    )

    plt.xticks(
        tick_marks,
        CLASSES,
        rotation=35,
        ha="right",
    )

    plt.yticks(
        tick_marks,
        CLASSES,
    )

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    # Write values.
    threshold = (
        cm.max() / 2.0
        if cm.size
        else 0
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                ),
                fontsize=11,
            )

    plt.tight_layout()

    path = (
        OUTPUT_DIR
        / "confusion_matrix.png"
    )

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    print(
        f"✓ Confusion matrix saved: "
        f"{path}"
    )

    return path


# ============================================================================
# SAVE TEST PREDICTIONS
# ============================================================================

def save_test_predictions(
    test_df,
    y_true,
    y_pred,
):

    result = test_df.copy()

    result[
        "true_index"
    ] = y_true

    result[
        "predicted_index"
    ] = y_pred

    result[
        "true_label"
    ] = [
        CLASSES[x]
        for x in y_true
    ]

    result[
        "predicted_label"
    ] = [
        CLASSES[x]
        for x in y_pred
    ]

    result[
        "correct"
    ] = (
        y_true == y_pred
    )

    path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    result.to_csv(
        path,
        index=False,
    )

    print(
        f"✓ Test predictions saved: "
        f"{path}"
    )

    return path


# ============================================================================
# SAVE FINAL MODEL
# ============================================================================

def save_final_model(
    model,
):

    path = (
        MODEL_DIR
        / "final_model.pth"
    )

    torch.save(

        {
            "model_state_dict":
                model.state_dict(),

            "classes":
                CLASSES,

            "class_to_index":
                CLASS_TO_INDEX,

            "image_size":
                IMAGE_SIZE,
        },

        path,
    )

    print(
        f"✓ Final model saved: {path}"
    )

    return path


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():

    total_start = time.time()

    print()
    print("=" * 78)
    print("FACIAL PALSY SEVERITY CLASSIFICATION")
    print("=" * 78)

    print()
    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Image size: "
        f"{IMAGE_SIZE} x {IMAGE_SIZE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {NUM_EPOCHS}"
    )

    print(
        f"Backbone LR: {BACKBONE_LR}"
    )

    print(
        f"Classifier LR: {CLASSIFIER_LR}"
    )

    # ------------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("LOADING DATA")
    print("=" * 78)

    train_df = load_dataframe(
        TRAIN_CSV,
        "train",
    )

    val_df = load_dataframe(
        VAL_CSV,
        "validation",
    )

    test_df = load_dataframe(
        TEST_CSV,
        "test",
    )

    print_dataset_summary(
        train_df,
        val_df,
        test_df,
    )

    # ------------------------------------------------------------------------
    # 2. TRANSFORMS
    # ------------------------------------------------------------------------

    train_transform, eval_transform = (
        create_transforms()
    )

    # ------------------------------------------------------------------------
    # 3. DATASETS
    # ------------------------------------------------------------------------

    train_dataset = FacialPalsyDataset(
        train_df,
        transform=train_transform,
    )

    val_dataset = FacialPalsyDataset(
        val_df,
        transform=eval_transform,
    )

    test_dataset = FacialPalsyDataset(
        test_df,
        transform=eval_transform,
    )

    # ------------------------------------------------------------------------
    # 4. DATA LOADERS
    # ------------------------------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=NUM_WORKERS,

        pin_memory=PIN_MEMORY,

    )

    print()
    print(
        "DataLoaders created:"
    )

    print(
        f"  Train batches: "
        f"{len(train_loader):,}"
    )

    print(
        f"  Validation batches: "
        f"{len(val_loader):,}"
    )

    print(
        f"  Test batches: "
        f"{len(test_loader):,}"
    )

    # ------------------------------------------------------------------------
    # 5. CLASS WEIGHTS
    # ------------------------------------------------------------------------

    class_weights = (
        calculate_class_weights(
            train_df
        )
    )

    # ------------------------------------------------------------------------
    # 6. MODEL
    # ------------------------------------------------------------------------

    model = FacialPalsyResNet18()

    model = model.to(
        DEVICE
    )

    # ------------------------------------------------------------------------
    # 7. LOSS
    # ------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss(

        weight=class_weights,

        label_smoothing=0.05,
    )

    # ------------------------------------------------------------------------
    # 8. OPTIMIZER
    # ------------------------------------------------------------------------

    optimizer = create_optimizer(
        model
    )

    scheduler = create_scheduler(
        optimizer
    )

    # ------------------------------------------------------------------------
    # 9. TRAINING
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STARTING TRAINING")
    print("=" * 78)

    history = []

    best_val_f1 = -float(
        "inf"
    )

    best_epoch = 0

    epochs_without_improvement = 0

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print()
        print("=" * 78)
        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}"
        )
        print("=" * 78)

        # --------------------------------------------------------------------
        # TRAIN
        # --------------------------------------------------------------------

        (
            train_loss,
            train_metrics,
            train_time,
        ) = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,
        )

        # --------------------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------------------

        (
            val_loss,
            val_metrics,
            val_time,
        ) = validate(

            model,

            val_loader,

            criterion,
        )

        epoch_time = (
            train_time
            + val_time
        )

        # --------------------------------------------------------------------
        # Scheduler
        # --------------------------------------------------------------------

        scheduler.step(
            val_metrics["macro_f1"]
        )

        backbone_lr, classifier_lr = (
            get_learning_rates(
                optimizer
            )
        )

        # --------------------------------------------------------------------
        # Print
        # --------------------------------------------------------------------

        print()
        print(
            f"  Train Loss:          "
            f"{train_loss:.4f}"
        )

        print(
            f"  Train Accuracy:      "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"  Train Balanced Acc:  "
            f"{train_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"  Train Macro F1:      "
            f"{train_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Val Loss:            "
            f"{val_loss:.4f}"
        )

        print(
            f"  Val Accuracy:        "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"  Val Balanced Acc:    "
            f"{val_metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"  Val Macro F1:        "
            f"{val_metrics['macro_f1']:.4f}"
        )

        print(
            f"  Backbone LR:         "
            f"{backbone_lr:.7f}"
        )

        print(
            f"  Classifier LR:       "
            f"{classifier_lr:.7f}"
        )

        print(
            f"  Time:                "
            f"{epoch_time:.1f}s"
        )

        print_per_class_metrics(
            val_metrics
        )

        # --------------------------------------------------------------------
        # Save history
        # --------------------------------------------------------------------

        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_metrics["accuracy"],

            "train_balanced_accuracy":
                train_metrics[
                    "balanced_accuracy"
                ],

            "train_macro_f1":
                train_metrics[
                    "macro_f1"
                ],

            "val_loss":
                val_loss,

            "val_accuracy":
                val_metrics[
                    "accuracy"
                ],

            "val_balanced_accuracy":
                val_metrics[
                    "balanced_accuracy"
                ],

            "val_macro_f1":
                val_metrics[
                    "macro_f1"
                ],

            "backbone_lr":
                backbone_lr,

            "classifier_lr":
                classifier_lr,

            "epoch_time":
                epoch_time,
        })

        history_path = save_history(
            history
        )

        # --------------------------------------------------------------------
        # BEST MODEL
        # --------------------------------------------------------------------

        current_f1 = (
            val_metrics["macro_f1"]
        )

        if current_f1 > best_val_f1:

            best_val_f1 = current_f1

            best_epoch = epoch

            epochs_without_improvement = 0

            checkpoint_path = (
                save_checkpoint(

                    model,

                    optimizer,

                    scheduler,

                    epoch,

                    best_val_f1,

                    history,
                )
            )

            print()
            print(
                "✓ BEST MODEL SAVED"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                f"No improvement "
                f"({epochs_without_improvement}/"
                f"{PATIENCE})"
            )

        # --------------------------------------------------------------------
        # EARLY STOPPING
        # --------------------------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print("=" * 78)
            print("EARLY STOPPING")
            print("=" * 78)

            break

    # ------------------------------------------------------------------------
    # TRAINING FINISHED
    # ------------------------------------------------------------------------

    total_training_time = (
        time.time()
        - total_start
    )

    print()
    print("=" * 78)
    print("TRAINING PHASE FINISHED")
    print("=" * 78)

    print()
    print(
        f"Total training time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"History saved:\n"
        f"{OUTPUT_DIR / 'training_history.csv'}"
    )

    print(
        f"Best model saved:\n"
        f"{MODEL_DIR / 'best_model.pth'}"
    )

    # ------------------------------------------------------------------------
    # 10. LOAD BEST MODEL
    # ------------------------------------------------------------------------

    model, loaded_epoch, loaded_f1 = (
        load_best_model(
            model
        )
    )

    model = model.to(
        DEVICE
    )

    print()
    print(
        f"✓ Best model loaded "
        f"from epoch {loaded_epoch}"
    )

    print(
        f"✓ Validation Macro F1: "
        f"{loaded_f1:.4f}"
    )

    # ------------------------------------------------------------------------
    # 11. FINAL TEST
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("FINAL TEST EVALUATION")
    print("=" * 78)

    (
        test_loss,
        test_metrics,
        y_true,
        y_pred,
    ) = evaluate_test(

        model,

        test_loader,

        criterion,
    )

    print()
    print(
        f"Test Loss:             "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy:         "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Balanced Acc:     "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Macro F1:         "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print()
    print(
        "TEST PER-CLASS METRICS"
    )

    for i, cls in enumerate(
        CLASSES
    ):

        print(
            f"  {cls:<18} "
            f"P={test_metrics['precision'][i]:.4f} "
            f"R={test_metrics['recall'][i]:.4f} "
            f"F1={test_metrics['f1'][i]:.4f} "
            f"N={int(test_metrics['support'][i])}"
        )

    # ------------------------------------------------------------------------
    # 12. CLASSIFICATION REPORT
    # ------------------------------------------------------------------------

    save_classification_report(
        y_true,
        y_pred,
    )

    # ------------------------------------------------------------------------
    # 13. CONFUSION MATRIX
    # ------------------------------------------------------------------------

    save_confusion_matrix(
        y_true,
        y_pred,
    )

    # ------------------------------------------------------------------------
    # 14. TEST PREDICTIONS
    # ------------------------------------------------------------------------

    save_test_predictions(
        test_df,
        y_true,
        y_pred,
    )

    # ------------------------------------------------------------------------
    # 15. FINAL MODEL
    # ------------------------------------------------------------------------

    save_final_model(
        model
    )

    # ------------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)

    print()
    print(
        f"Best Validation Macro F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Test Accuracy:          "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"Test Macro F1:          "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print()
    print(
        "GENERATED FILES"
    )

    print(
        f"✓ {MODEL_DIR / 'best_model.pth'}"
    )

    print(
        f"✓ {MODEL_DIR / 'final_model.pth'}"
    )

    print(
        f"✓ {OUTPUT_DIR / 'training_history.csv'}"
    )

    print(
        f"✓ {OUTPUT_DIR / 'test_predictions.csv'}"
    )

    print(
        f"✓ {OUTPUT_DIR / 'confusion_matrix.png'}"
    )

    print(
        f"✓ {OUTPUT_DIR / 'classification_report.txt'}"
    )

    print()
    print(
        "NEXT PROJECT STAGE:"
    )

    print(
        "1. Analyze confusion matrix"
    )

    print(
        "2. Grad-CAM explainability"
    )

    print(
        "3. Facial-region analysis"
    )

    print(
        "4. Severity inference pipeline"
    )

    print(
        "5. Streamlit/UI integration"
    )

    print()
    print("=" * 78)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    main()