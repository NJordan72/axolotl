"""
tests for dataset storage config
"""

import tempfile
from pathlib import Path

from pydantic import BaseModel

from axolotl.utils.config.models.dataset.dataset_storage_config import (
    DatasetStorageConfig,
)


class TestStorageConfig(BaseModel):
    """
    Test class for storage config
    """

    config: DatasetStorageConfig


class TestDatasetStorageConfig:
    """
    Test class for dataset storage config
    """

    def test_base_storage_config_file_type_inference(self):
        config = TestStorageConfig(config={"path": "NovaSky-AI/Sky-T1_data_17k"})
        assert config.config.storage_backend == "hf_hub"

    def test_local_storage_backend(self):
        """Usual use case.  Verify a directory of parquet files can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_ds_dir = Path(tmp_dir) / "tmp_dataset"
            tmp_ds_dir.mkdir()
            tmp_ds_path = tmp_ds_dir / "shard1.parquet"

            import pandas as pd

            df = pd.DataFrame({"col1": [1, 2, 3]})
            df.to_parquet(tmp_ds_path)

            config = TestStorageConfig(config={"path": str(tmp_ds_dir)})
            assert config.config.storage_backend == "local"
