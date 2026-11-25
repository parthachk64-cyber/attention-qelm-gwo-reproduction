import os
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("IIOT_QELM_REFERENCES", str(Path(__file__).resolve().parents[1] / "references" / "reported"))

from iiot_qelm.data.prepare import PrepareConfig, prepare_frame  # noqa: E402
from iiot_qelm.data.synthetic import make_fixture  # noqa: E402
from iiot_qelm.models.pipeline import TrainConfig  # noqa: E402


@pytest.fixture(scope="session")
def raw_df():
    return make_fixture(1200, seed=3)


@pytest.fixture(scope="session")
def prepared(raw_df):
    return prepare_frame(raw_df, PrepareConfig(task="multiclass"), source={"fixture": True})


@pytest.fixture(scope="session")
def prepared_binary(raw_df):
    return prepare_frame(raw_df, PrepareConfig(task="binary"), source={"fixture": True})


@pytest.fixture
def tiny_train():
    return TrainConfig(epochs=2, patience=2, batch_size=64)
