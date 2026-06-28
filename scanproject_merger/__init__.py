"""Register and merge ScannerConsolidator ``.scanproject`` point clouds.

A library port of the standalone ``scanproject-merger`` tool. The high-level
:func:`merge_scan_projects` runs the full discover → register → export pipeline;
the underlying ``register_scans`` and ``export_*`` functions are exposed for callers
that need finer control.
"""

from .export import (
    export_merged_cloud,
    export_original_scans,
    export_transformed_scans,
    write_registration_report,
)
from .format import (
    GeoReference,
    PointBatch,
    ScanProject,
    ScanProjectError,
    transform_points,
)
from .merge import MergeOutputs, discover, merge_scan_projects
from .registration import (
    RegistrationEdge,
    RegistrationResult,
    RejectedRegistrationEdge,
    register_scans,
)

__all__ = [
    "GeoReference",
    "MergeOutputs",
    "PointBatch",
    "RegistrationEdge",
    "RegistrationResult",
    "RejectedRegistrationEdge",
    "ScanProject",
    "ScanProjectError",
    "discover",
    "export_merged_cloud",
    "export_original_scans",
    "export_transformed_scans",
    "merge_scan_projects",
    "register_scans",
    "transform_points",
    "write_registration_report",
]
