"""
Material merging / segmentation pipeline (final stage of matSeparate).

Turns a single RGB image into:
  1. a MINC-style semantic material label map, and
  2. per-object instance masks (connected components of a single material),
segmented at a customizable level of the taxonomy.

See MERGE.md for the full design.
"""

from segmentation.config import SegmentationConfig

__all__ = ["SegmentationConfig"]
