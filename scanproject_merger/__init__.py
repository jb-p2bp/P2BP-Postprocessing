"""Register and merge ScannerConsolidator ``.scanproject`` point clouds.

A library port of the standalone ``scanproject-merger`` tool. The high-level
:func:`merge_scan_projects` runs the full discover → register → export pipeline;
the underlying ``register_scans`` and ``export_*`` functions are exposed for callers
that need finer control.

Provenance: ported from ``scanproject-merger`` branch ``feat/scanproject-merger``
at commit ``a4a33b8``. ``format``/``registration``/``visual``/``export`` started as
verbatim copies; later commits in this repository harden them (keyframe path and
size validation, run-scoped feature/grid caches, point-count limit, atomic and
collision-free output). When re-syncing from upstream, diff against that revision
rather than assuming these files are still identical to it.
"""

from .export import (
    export_merged_cloud,
    export_merged_cloud_bin,
    export_merged_cloud_outputs,
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
    RegistrationParams,
    RegistrationResult,
    RejectedRegistrationEdge,
    register_scans,
)

__all__ = [
    "GeoReference",
    "MergeOutputs",
    "PointBatch",
    "RegistrationEdge",
    "RegistrationParams",
    "RegistrationResult",
    "RejectedRegistrationEdge",
    "ScanProject",
    "ScanProjectError",
    "discover",
    "export_merged_cloud",
    "export_merged_cloud_bin",
    "export_merged_cloud_outputs",
    "export_original_scans",
    "export_transformed_scans",
    "merge_scan_projects",
    "register_scans",
    "write_registration_report",
]
