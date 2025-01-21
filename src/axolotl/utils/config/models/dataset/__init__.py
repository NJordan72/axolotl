"""
Dataset configuration models for Axolotl.

This module provides Pydantic models for configuring dataset storage backends, file formats,
and data shapes. It supports various storage locations including Google Cloud Storage (GCS),
Amazon S3, and Hugging Face Hub. The module uses discriminated unions to automatically select
the appropriate configuration model based on the dataset path.

Available config models:
- BaseStorageConfig: Common configuration fields for all storage types
- HuggingFaceStorageConfig: Configuration for Hugging Face Hub storage
- GCSStorageConfig: Configuration for Google Cloud Storage
- S3StorageConfig: Configuration for Amazon S3
"""

from pathlib import Path
from typing import Annotated, Any, Dict, Literal, Mapping, Optional, Sequence, Union

from datasets import load_dataset
from huggingface_hub.errors import HFValidationError
from pydantic import BaseModel, Discriminator, Field, ValidationInfo, field_validator

FileType = Literal["parquet", "arrow", "csv", "text", "json"]
StorageBackend = Literal["gcs", "s3", "hf_hub", "local"]


# Storage Configuration
class BaseStorageConfig(BaseModel):
    """Common fields for all storage configs."""

    storage_backend: StorageBackend  # Discriminator for storage backends
    file_type: Optional[FileType] = Field(validation_alias="ds_type", default=None)
    path: str
    data_files: Optional[
        Union[str, Sequence[str], Mapping[str, str], Sequence[Mapping[str, str]]]
    ] = None
    name: Optional[str] = None

    @field_validator("file_type", mode="after")
    @classmethod
    def determine_file_type(
        cls, v: Optional[FileType], info: ValidationInfo
    ) -> FileType:
        """Infer file type from the file extension if not explicitly provided."""
        if v:
            return v

        path = info.data["path"]

        match path.lower():
            case p if ".parquet" in p:
                return "parquet"
            case p if ".arrow" in p:
                return "arrow"
            case p if ".csv" in p:
                return "csv"
            case p if ".txt" in p:
                return "text"

        return "json"


class HuggingFaceStorageConfig(BaseStorageConfig):
    """
    Configuration for datasets stored on the Hugging Face Hub.

    Attributes:
        storage_backend: Fixed as "hf_hub" to identify Hugging Face Hub storage
        revision: Optional Git revision (branch name, tag, or commit hash) to use
        trust_remote_code: Whether to trust and execute remote code from the dataset repository
    """

    storage_backend: Literal["hf_hub"] = "hf_hub"
    revision: Optional[str] = None
    trust_remote_code: bool = False
    use_auth_token: bool = False


class GCSStorageConfig(BaseStorageConfig):
    """
    Configuration for datasets stored in Google Cloud Storage (GCS).

    Attributes:
        storage_backend: Fixed as "gcs" to identify Google Cloud Storage
    """

    storage_backend: Literal["gcs"] = "gcs"


class S3StorageConfig(BaseStorageConfig):
    """
    Configuration for datasets stored in Amazon S3.

    Attributes:
        storage_backend: Fixed as "s3" to identify Amazon S3 storage
    """

    storage_backend: Literal["s3"] = "s3"


class LocalStorageConfig(BaseStorageConfig):
    """
    Configuration for datasets stored locally.

    Attributes:
        storage_backend: Fixed as "local" to identify local storage
    """

    storage_backend: Literal["local"] = "local"


def _get_storage_backend(v: Any) -> StorageBackend:
    """
    Determine the storage backend based on the dataset path.

    Args:
        v: Either a dict or BaseModel containing a path field

    Returns:
        str: Storage backend identifier ('gcs', 's3', 'hf_hub', or 'local')

    Raises:
        ValueError: If path is invalid or missing

    Notes:
        - A callable discriminator function may pass in either
          a dict or a BaseModel. so we have to handle both cases.
    """
    match v:
        case {"path": p}:
            path = p
        case BaseModel() as model if hasattr(model, "path"):
            path = model.path

    if not isinstance(path, str):
        raise ValueError(f"Invalid path: {path}")

    match path.lower():
        case p if p.startswith(("gcs://", "gs://")):
            return "gcs"
        case p if p.startswith("s3://"):
            return "s3"
        case p if _check_if_exists_locally(path, v):
            # Prefer local datasets over HF Hub
            return "local"
        case p if _check_if_exists_on_hf_hub(path, v):
            return "hf_hub"

    raise ValueError(f"Unable to determine storage backend for path: {path}")


def _check_if_exists_locally(path: str, v: Union[BaseModel, Dict[str, Any]]) -> bool:
    """Check if the dataset exists locally."""
    local_path = Path(path)

    if isinstance(v, BaseModel):
        name = getattr(v, "name", None)
        data_files = getattr(v, "data_files", None)
    else:
        name = v.get("name")
        data_files = v.get("data_files")

    if local_path.exists():
        try:
            load_dataset(
                path, name=name, data_files=data_files, streaming=True, split=None
            )
            return True
        except (FileNotFoundError, ConnectionError, HFValidationError, ValueError):
            return False

    return False


def _check_if_exists_on_hf_hub(
    path: str,
    v: Union[BaseModel, Dict[str, Any]],
) -> bool:
    """
    Check if the dataset exists on the Hugging Face Hub.  The key
    is that Streaming=True makes this a lazy evaluation, so we get
    the validation for free without actually downloading the dataset.
    """
    if isinstance(v, BaseModel):
        name = getattr(v, "name", None)
        use_auth_token = getattr(v, "use_auth_token", False)
        revision = getattr(v, "revision", None)
        trust_remote_code = getattr(v, "trust_remote_code", False)
    else:
        name = v.get("name")
        use_auth_token = v.get("use_auth_token")
        revision = v.get("revision")
        trust_remote_code = v.get("trust_remote_code")

    try:
        load_dataset(
            path,
            name=name,
            streaming=True,
            token=use_auth_token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        return True
    except (FileNotFoundError, ConnectionError, HFValidationError, ValueError):
        return False


# Dataset configuration using discriminated unions
DatasetStorageConfig = Annotated[
    Union[
        HuggingFaceStorageConfig,
        GCSStorageConfig,
        S3StorageConfig,
    ],
    Discriminator(_get_storage_backend),
]
