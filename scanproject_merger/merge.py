"""High-level entry points for registering and merging ``.scanproject`` packages.

This module replaces the standalone tool's command-line interface with a
programmatic API. ``merge_scan_projects`` performs the same discover → register →
export pipeline as the CLI but returns structured results instead of printing and
exiting, so it can be embedded in the P2BP postprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .export import (
    export_merged_cloud_outputs,
    export_original_scans,
    export_transformed_scans,
    write_registration_report,
)
from .format import ScanProject, ScanProjectError
from .registration import RegistrationParams, RegistrationResult, register_scans


def discover(inputs: Iterable[str | Path]) -> list[Path]:
    """Resolve scan packages from explicit ``.scanproject`` paths or parent directories.

    Each input may be an individual ``.scanproject`` directory or a directory whose
    immediate children are scan packages. Duplicates are removed and the result is
    sorted for deterministic ordering. Raises :class:`ScanProjectError` when nothing
    resolves to a scan package.
    """
    packages: dict[Path, None] = {}
    for value in (Path(item) for item in inputs):
        if value.suffix == ".scanproject":
            packages[value.resolve()] = None
        elif value.is_dir():
            for package in value.glob("*.scanproject"):
                packages[package.resolve()] = None
        else:
            raise ScanProjectError(f"input does not exist or is not a scan package: {value}")
    if not packages:
        raise ScanProjectError("no .scanproject packages found")
    return sorted(packages)


@dataclass(frozen=True)
class MergeOutputs:
    """Paths and statistics produced by :func:`merge_scan_projects`."""

    output: Path
    report: Path
    point_count: int
    result: RegistrationResult
    bin_output: Path | None = None
    transformed_scans: list[Path] = field(default_factory=list)
    original_scans: list[Path] = field(default_factory=list)


def merge_scan_projects(
    inputs: Iterable[str | Path],
    output: str | Path,
    *,
    bin_output: str | Path | None = None,
    report: str | Path | None = None,
    transformed_scans_dir: str | Path | None = None,
    original_scans_dir: str | Path | None = None,
    registration: RegistrationParams = RegistrationParams(),  # Frozen, so a shared default is safe.
    deduplicate_voxel: float = 0.02,  # 2 cm output grid.
    export_minimum_confidence: int = 0,  # Preserve all output points.
) -> MergeOutputs:
    """Register overlapping scan packages and write a merged LAS/LAZ point cloud.

    ``inputs`` may mix individual ``.scanproject`` directories and parent directories
    containing them; they are resolved with :func:`discover`. The merged cloud is
    written to ``output`` and an audit report to ``report`` (defaulting to
    ``output`` with a ``.registration.json`` suffix). When ``bin_output`` is given,
    the same points are also written using ScannerConsolidator's packed ``.bin``
    point format. When ``transformed_scans_dir`` or ``original_scans_dir`` is
    given, one LAZ file per source scan is also written there. Source packages
    are never modified.

    ``registration`` carries the alignment tuning (see :class:`RegistrationParams`);
    ``deduplicate_voxel`` and ``export_minimum_confidence`` control output only.
    Registration and export confidence are intentionally separate: the defaults
    align on medium+ points while writing every point to the outputs.

    Raises :class:`ScanProjectError` for malformed packages and :class:`ValueError`
    for inconsistent or unregisterable inputs.
    """
    projects = [ScanProject.open(path) for path in discover(inputs)]
    result = register_scans(projects, registration)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bin_output_path = Path(bin_output) if bin_output else None
    if bin_output_path:
        bin_output_path.parent.mkdir(parents=True, exist_ok=True)
    point_count = export_merged_cloud_outputs(
        result,
        laz_output=output_path,
        bin_output=bin_output_path,
        minimum_confidence=export_minimum_confidence,
        deduplicate_voxel=deduplicate_voxel,
    )
    report_path = Path(report) if report else output_path.with_suffix(".registration.json")
    write_registration_report(result, report_path, point_count)
    transformed_scans = (
        export_transformed_scans(result, transformed_scans_dir, export_minimum_confidence)
        if transformed_scans_dir
        else []
    )
    original_scans = (
        export_original_scans(result, original_scans_dir, export_minimum_confidence)
        if original_scans_dir
        else []
    )
    return MergeOutputs(
        output=output_path,
        report=report_path,
        point_count=point_count,
        result=result,
        bin_output=bin_output_path,
        transformed_scans=transformed_scans,
        original_scans=original_scans,
    )
