"""Pydantic models for messages produced by p2bp-cf-worker on the MESH_JOBS queue.

Each message carries a `type` for dispatch and a `version` that is pinned per
variant: a consumer validates only the exact schema version it was built for
and rejects any other version, so an incompatible producer fails loudly rather
than being silently mis-parsed. The discriminated union below mirrors the
`MeshJobQueueMessage` type defined in
`p2bp-cf-worker/src/routes/api/mesh.jobs.builder.ts`.

Field names are intentionally camelCase to match that JSON wire contract
exactly. Do not rename them to snake_case -- it would break deserialization
of messages produced by the worker.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

# Identifier that is stripped of surrounding whitespace and must be non-empty,
# so values like " " are rejected rather than treated as valid ids.
NonEmptyId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

__all__ = [
    "MESH_GENERATE_TYPE",
    "MESH_REFINE_TYPE",
    "MESH_GENERATE_VERSION",
    "MESH_REFINE_VERSION",
    "MeshGenerateJob",
    "MeshRefineJob",
    "MeshJobMessage",
    "parse_mesh_job_message",
]

MESH_GENERATE_TYPE = "mesh.generate"
MESH_REFINE_TYPE = "mesh.refine"

MESH_GENERATE_VERSION = 1
MESH_REFINE_VERSION = 1


class _MeshJobBase(BaseModel):
    """Fields and config shared by every mesh job message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    organizationId: NonEmptyId
    projectId: NonEmptyId


class MeshGenerateJob(_MeshJobBase):
    """Mesh generation job: build a mesh from uploaded zone scan archives."""

    type: Literal[MESH_GENERATE_TYPE] = MESH_GENERATE_TYPE
    version: Literal[MESH_GENERATE_VERSION] = MESH_GENERATE_VERSION
    zoneScanObjectKeys: list[str] = Field(min_length=1)


class MeshRefineJob(_MeshJobBase):
    """Mesh refinement job: refine an existing mesh for a project."""

    type: Literal[MESH_REFINE_TYPE] = MESH_REFINE_TYPE
    version: Literal[MESH_REFINE_VERSION] = MESH_REFINE_VERSION


# `version` is an exact pin per variant, so a new schema revision is added as a
# new model in this union (e.g. a MeshGenerateJobV2 with version=2) rather than
# by editing the existing classes. The discriminator is `type`; if two revisions
# ever need to share a `type`, switch to a nested (type, version) discriminator.
MeshJobMessage = Annotated[
    MeshGenerateJob | MeshRefineJob,
    Field(discriminator="type"),
]

mesh_job_message_adapter: TypeAdapter[MeshJobMessage] = TypeAdapter(MeshJobMessage)


def parse_mesh_job_message(data: object) -> MeshJobMessage:
    """Validate arbitrary data into a :class:`MeshJobMessage` variant."""

    return mesh_job_message_adapter.validate_python(data)