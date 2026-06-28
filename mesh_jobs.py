"""Pydantic models for messages produced by p2bp-cf-worker on the MESH_JOBS queue.

Each message carries a `type` and `version` so the EC2 consumer can dispatch
on type and evolve its schema over time. The discriminated union below mirrors
the `MeshJobQueueMessage` type defined in
`p2bp-cf-worker/src/routes/api/mesh.jobs.builder.ts`.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

MESH_GENERATE_TYPE = "mesh.generate"
MESH_REFINE_TYPE = "mesh.refine"

MESH_GENERATE_VERSION = 1
MESH_REFINE_VERSION = 1


class MeshGenerateJob(BaseModel):
    """Mesh generation job: build a mesh from uploaded zone scan archives."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal[MESH_GENERATE_TYPE] = MESH_GENERATE_TYPE
    version: Literal[MESH_GENERATE_VERSION] = MESH_GENERATE_VERSION
    organizationId: str = Field(min_length=1)
    projectId: str = Field(min_length=1)
    zoneScanObjectKeys: list[str] = Field(default_factory=list)


class MeshRefineJob(BaseModel):
    """Mesh refinement job: refine an existing mesh for a project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal[MESH_REFINE_TYPE] = MESH_REFINE_TYPE
    version: Literal[MESH_REFINE_VERSION] = MESH_REFINE_VERSION
    organizationId: str = Field(min_length=1)
    projectId: str = Field(min_length=1)


MeshJobMessage = Annotated[
    Union[MeshGenerateJob, MeshRefineJob],
    Field(discriminator="type"),
]

mesh_job_message_adapter: TypeAdapter[MeshJobMessage] = TypeAdapter(MeshJobMessage)


def parse_mesh_job_message(data: object) -> MeshJobMessage:
    """Validate arbitrary data into a :class:`MeshJobMessage` variant."""

    return mesh_job_message_adapter.validate_python(data)