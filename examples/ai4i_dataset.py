"""
Provides tools to fetch, verify, and load the UCI AI4I 2020 Predictive Maintenance dataset.

This dataset consists of 10,000 rows of process logs from a milling machine, where 3.39%
of the rows represent failures.

Source URL:
    https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Citation:
    Matzka, S. (2020). AI4I 2020 Predictive Maintenance Dataset [Dataset].
    UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

Licence:
    Creative Commons Attribution 4.0 International (CC BY 4.0)

The dataset is downloaded on first use and cached as a CSV file under `examples/data/`.
This directory is excluded via `.gitignore` because this repository never commits
process data. Trust in the cached file is established through the SHA256 checksum
hardcoded in this module, rather than by relying on the repository.
"""
from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

AI4I_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
AI4I_ZIP_SHA256 = "f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252"
AI4I_CSV_MEMBER = "ai4i2020.csv"
EXPECTED_ROWS = 10_000
TARGET_COLUMN = "Machine failure"
ASSET_COLUMN = "Type"
IDENTIFIER_COLUMNS = ["UDI", "Product ID"]
# These five columns are individual failure modes (tool wear, heat dissipation, power,
# overstrain, random). Each is a label recorded when the machine has ALREADY failed.
# Training on them is a trap, because the model will score almost perfectly by simply
# reading the answer, while predicting nothing of value. This dataset ships that trap
# in the file; this example explicitly removes it by default to prevent leakage.
LEAKY_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def default_cache_path() -> Path:
    """
    Constructs the default path to the cached dataset file.

    Returns the absolute path to `examples/data/ai4i2020.csv` relative to this module.
    """
    return Path(__file__).resolve().parent / "data" / AI4I_CSV_MEMBER


def _verify_sha256(payload: bytes) -> None:
    """
    Verifies that the downloaded zip archive payload matches the expected SHA256 digest.

    This ensures the integrity and authenticity of the dataset before it is extracted.
    It raises a RuntimeError with a descriptive message if the digest is wrong, ensuring
    nothing is written to disk.
    """
    actual = hashlib.sha256(payload).hexdigest()
    if actual != AI4I_ZIP_SHA256:
        raise RuntimeError(
            f"Checksum mismatch for AI4I payload. Expected {AI4I_ZIP_SHA256}, "
            f"but got {actual}. Nothing was written to disk."
        )


def download_ai4i(dest: Path | None = None, *, timeout: float = 60.0) -> Path:
    """
    Downloads and extracts the AI4I dataset to the given destination path.

    Creates the parent directory if it does not exist. The function downloads the zip
    archive into memory, verifies its SHA256 checksum, and extracts the target CSV to
    a sibling temporary file before atomically replacing the destination path. This
    prevents interrupted downloads from leaving half-written CSVs.
    """
    dest_path = dest if dest is not None else default_cache_path()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(AI4I_URL, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeError(
            f"Failed to download the AI4I dataset from {AI4I_URL} to {dest_path}. "
            "Please download the zip archive manually and extract ai4i2020.csv to "
            "that path."
        ) from exc

    _verify_sha256(payload)

    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        csv_data = zf.read(AI4I_CSV_MEMBER)

    # The checksum already proves the archive is the right one; this counts the
    # rows of what came out of it, so a future release that repackages the file
    # under the same name is caught here rather than halfway through a run.
    extracted_rows = len(csv_data.splitlines()) - 1
    if extracted_rows != EXPECTED_ROWS:
        raise RuntimeError(
            f"{AI4I_CSV_MEMBER} holds {extracted_rows} rows, expected {EXPECTED_ROWS}. "
            "Nothing was written to disk."
        )

    temp_dest = dest_path.with_suffix(".tmp")
    temp_dest.write_bytes(csv_data)
    temp_dest.replace(dest_path)

    return dest_path


def load_ai4i(
    cache_path: str | Path | None = None,
    *,
    download: bool = True,
    drop_leaky: bool = True,
) -> pd.DataFrame:
    """
    Loads the AI4I dataset into a pandas DataFrame from the cache path.

    It validates the presence of expected columns to ensure data integrity.
    Dropping the identifier columns is not cosmetic: `UDI` is a row counter and
    `Product ID` is unique per row, so both would be memorised by a model.
    """
    resolved_path = Path(cache_path) if cache_path is not None else default_cache_path()

    if not resolved_path.is_file():
        if download:
            download_ai4i(resolved_path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {resolved_path} and download is disabled. "
                f"It is available at {AI4I_URL}"
            )

    df = pd.read_csv(resolved_path)

    required_cols = [TARGET_COLUMN] + IDENTIFIER_COLUMNS + LEAKY_COLUMNS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"The downloaded CSV is missing required columns: {missing}")

    if drop_leaky:
        return df.drop(columns=IDENTIFIER_COLUMNS + LEAKY_COLUMNS)
    return df


if __name__ == "__main__":
    path = download_ai4i()
    df_clean = load_ai4i(path)
    failure_rate = (df_clean[TARGET_COLUMN].sum() / len(df_clean)) * 100
    dropped = IDENTIFIER_COLUMNS + LEAKY_COLUMNS

    print(f"Cache path: {path}")
    print(f"Shape: {df_clean.shape}")
    print(f"Failure rate: {failure_rate:.2f}%")
    print(f"Dropped columns: {dropped}")
