"""Data loading utilities for WikiChurches SSL attention analysis.

This package provides:
- Annotation parsing (building_parts.json with 139 expert-annotated images)
- Dataset classes for both annotated subset and full 9,502 image collection
- Collate functions for DataLoader integration

Example:
    from ssl_attention.data import AnnotatedSubset, load_annotations
    from ssl_attention.config import DATASET_PATH

    # Load just annotations
    annotations = load_annotations(DATASET_PATH / "building_parts.json")

    # Or use the dataset class
    dataset = AnnotatedSubset(DATASET_PATH)
    sample = dataset[0]
    print(sample["annotation"].num_bboxes)
"""

from ssl_attention.data.annotations import (
    BoundingBox,
    FeatureType,
    ImageAnnotation,
    load_annotations,
    load_annotations_with_features,
    load_feature_types,
)
from ssl_attention.data.wikichurches import (
    AnnotatedSubset,
    FullDataset,
    collate_annotated,
    collate_classification,
)

__all__ = [
    # Dataclasses
    "BoundingBox",
    "FeatureType",
    "ImageAnnotation",
    # Loading functions
    "load_annotations",
    "load_annotations_with_features",
    "load_feature_types",
    # Dataset classes
    "AnnotatedSubset",
    "FullDataset",
    # Collate functions
    "collate_annotated",
    "collate_classification",
]
