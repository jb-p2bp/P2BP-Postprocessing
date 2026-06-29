"""Tests for the MESH_JOBS queue message models."""

import pytest
from pydantic import ValidationError

from mesh_jobs import (
    MESH_GENERATE_TYPE,
    MESH_GENERATE_VERSION,
    MESH_REFINE_TYPE,
    MESH_REFINE_VERSION,
    MeshGenerateJob,
    MeshRefineJob,
    parse_mesh_job_message,
)


class TestMeshGenerateJob:
    def test_valid_generate_message(self) -> None:
        message = MeshGenerateJob(
            organizationId="org_123",
            projectId="proj_456",
            zoneScanObjectKeys=["a/b.zip", "c/d.zip"],
        )

        assert message.type == MESH_GENERATE_TYPE
        assert message.version == MESH_GENERATE_VERSION
        assert message.organizationId == "org_123"
        assert message.projectId == "proj_456"
        assert message.zoneScanObjectKeys == ["a/b.zip", "c/d.zip"]

    def test_defaults_type_and_version(self) -> None:
        message = MeshGenerateJob(
            organizationId="org",
            projectId="proj",
            zoneScanObjectKeys=["a/b.zip"],
        )

        assert message.type == MESH_GENERATE_TYPE
        assert message.version == MESH_GENERATE_VERSION

    def test_empty_scan_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeshGenerateJob(
                organizationId="org",
                projectId="proj",
                zoneScanObjectKeys=[],
            )

    def test_missing_scan_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeshGenerateJob(
                organizationId="org",
                projectId="proj",
            )

    def test_explicit_type_must_match(self) -> None:
        with pytest.raises(ValidationError):
            MeshGenerateJob(
                type=MESH_REFINE_TYPE,
                organizationId="org",
                projectId="proj",
                zoneScanObjectKeys=["a/b.zip"],
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeshGenerateJob(
                organizationId="org",
                projectId="proj",
                zoneScanObjectKeys=["a/b.zip"],
                unexpected="value",  # type: ignore[call-arg]
            )

    def test_empty_organization_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeshGenerateJob(
                organizationId="",
                projectId="proj",
                zoneScanObjectKeys=["a/b.zip"],
            )

    def test_empty_project_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeshGenerateJob(
                organizationId="org",
                projectId="",
                zoneScanObjectKeys=["a/b.zip"],
            )

    def test_whitespace_only_ids_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeshGenerateJob(
                organizationId="   ",
                projectId="proj",
                zoneScanObjectKeys=["a/b.zip"],
            )

    def test_ids_are_stripped(self) -> None:
        message = MeshGenerateJob(
            organizationId="  org  ",
            projectId="  proj  ",
            zoneScanObjectKeys=["a/b.zip"],
        )

        assert message.organizationId == "org"
        assert message.projectId == "proj"


class TestMeshRefineJob:
    def test_valid_refine_message(self) -> None:
        message = MeshRefineJob(
            organizationId="org_123",
            projectId="proj_456",
        )

        assert message.type == MESH_REFINE_TYPE
        assert message.version == MESH_REFINE_VERSION
        assert message.organizationId == "org_123"
        assert message.projectId == "proj_456"

    def test_defaults_type_and_version(self) -> None:
        message = MeshRefineJob(organizationId="org", projectId="proj")

        assert message.type == MESH_REFINE_TYPE
        assert message.version == MESH_REFINE_VERSION

    def test_explicit_type_must_match(self) -> None:
        with pytest.raises(ValidationError):
            MeshRefineJob(
                type=MESH_GENERATE_TYPE,
                organizationId="org",
                projectId="proj",
            )

    def test_zone_scan_keys_rejected_on_refine(self) -> None:
        with pytest.raises(ValidationError):
            MeshRefineJob(
                organizationId="org",
                projectId="proj",
                zoneScanObjectKeys=["a/b.zip"],  # type: ignore[call-arg]
            )

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MeshRefineJob(
                organizationId="org",
                projectId="proj",
                unexpected="value",  # type: ignore[call-arg]
            )


class TestParseMeshJobMessage:
    def test_round_trips_through_serialization(self) -> None:
        original = MeshGenerateJob(
            organizationId="org",
            projectId="proj",
            zoneScanObjectKeys=["a/b.zip", "c/d.zip"],
        )

        dumped = original.model_dump()

        # The serialized payload keeps the camelCase wire field names...
        assert set(dumped) == {
            "type",
            "version",
            "organizationId",
            "projectId",
            "zoneScanObjectKeys",
        }
        # ...and parses back into an equal model.
        assert parse_mesh_job_message(dumped) == original

    def test_parses_generate_variant(self) -> None:
        result = parse_mesh_job_message(
            {
                "type": MESH_GENERATE_TYPE,
                "version": MESH_GENERATE_VERSION,
                "organizationId": "org",
                "projectId": "proj",
                "zoneScanObjectKeys": ["k1", "k2"],
            }
        )

        assert isinstance(result, MeshGenerateJob)
        assert result.zoneScanObjectKeys == ["k1", "k2"]

    def test_parses_refine_variant(self) -> None:
        result = parse_mesh_job_message(
            {
                "type": MESH_REFINE_TYPE,
                "version": MESH_REFINE_VERSION,
                "organizationId": "org",
                "projectId": "proj",
            }
        )

        assert isinstance(result, MeshRefineJob)

    def test_missing_type_rejected_by_discriminator(self) -> None:
        with pytest.raises(ValidationError):
            parse_mesh_job_message(
                {
                    "organizationId": "org",
                    "projectId": "proj",
                    "zoneScanObjectKeys": ["a"],
                }
            )

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parse_mesh_job_message(
                {
                    "type": "mesh.unknown",
                    "version": 1,
                    "organizationId": "org",
                    "projectId": "proj",
                }
            )

    def test_wrong_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parse_mesh_job_message(
                {
                    "type": MESH_GENERATE_TYPE,
                    "version": 2,
                    "organizationId": "org",
                    "projectId": "proj",
                    "zoneScanObjectKeys": ["a/b.zip"],
                }
            )

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            [],
            "not-an-object",
            42,
        ],
    )
    def test_non_object_inputs_rejected(self, payload: object) -> None:
        with pytest.raises(ValidationError):
            parse_mesh_job_message(payload)

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parse_mesh_job_message(
                {
                    "type": MESH_REFINE_TYPE,
                    "version": MESH_REFINE_VERSION,
                    "organizationId": "org",
                }
            )

    def test_refine_with_generate_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            parse_mesh_job_message(
                {
                    "type": MESH_REFINE_TYPE,
                    "version": MESH_REFINE_VERSION,
                    "organizationId": "org",
                    "projectId": "proj",
                    "zoneScanObjectKeys": ["a"],
                }
            )


class TestContractSnapshot:
    """Pin the field names and constants to a manual transcription of the
    p2bp-cf-worker contract in `src/routes/api/mesh.jobs.builder.ts`.

    These assertions only catch drift on the Python side -- they do not read
    the TypeScript file, so a change to the worker contract must be mirrored
    here by hand. Keep this class in sync when the worker contract changes."""

    def test_generate_field_names_match_worker(self) -> None:
        fields = set(MeshGenerateJob.model_fields)

        assert fields == {
            "type",
            "version",
            "organizationId",
            "projectId",
            "zoneScanObjectKeys",
        }

    def test_refine_field_names_match_worker(self) -> None:
        fields = set(MeshRefineJob.model_fields)

        assert fields == {"type", "version", "organizationId", "projectId"}

    def test_constants_match_worker(self) -> None:
        assert MESH_GENERATE_TYPE == "mesh.generate"
        assert MESH_REFINE_TYPE == "mesh.refine"
        assert MESH_GENERATE_VERSION == 1
        assert MESH_REFINE_VERSION == 1