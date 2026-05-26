#!/usr/bin/env python3
"""Download VectorDBBench OpenAI500K parquet files and convert them for Knowhere."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from parquet_to_fbin import write_fbin, write_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-local-dir", type=Path, default=Path("/tmp/vectordb_bench/dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", choices=["s3", "aliyun"], default="s3")
    parser.add_argument("--use-shuffled-data", choices=["true", "false"], default="false")
    parser.add_argument("--base-limit", type=int, default=-1)
    parser.add_argument("--query-limit", type=int, default=-1)
    parser.add_argument("--truth-topk", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["DATASET_LOCAL_DIR"] = str(args.dataset_local_dir)
    os.environ["USE_SHUFFLED_DATA"] = args.use_shuffled_data

    try:
        from vectordb_bench.backend.data_source import DatasetSource
        from vectordb_bench.backend.dataset import Dataset
    except ImportError as exc:
        raise SystemExit(
            "VectorDBBench is not importable. Install it first, for example: "
            "python -m pip install -e /path/to/VectorDBBench"
        ) from exc

    source = DatasetSource.S3 if args.source == "s3" else DatasetSource.AliyunOSS
    manager = Dataset.OPENAI.manager(500_000)

    data_dir = Path(manager.data_dir)
    download_files = [*manager.data.train_files, manager.data.test_file, "neighbors.parquet"]
    source.reader().read(
        dataset=manager.data.dir_name.lower(),
        files=download_files,
        local_ds_root=data_dir,
    )

    train_files = [data_dir / name for name in manager.data.train_files]
    test_file = data_dir / manager.data.test_file
    neighbors_file = data_dir / "neighbors.parquet"
    for path in [*train_files, test_file, neighbors_file]:
        if not path.exists():
            raise SystemExit(f"expected VectorDBBench dataset file does not exist: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_fbin(train_files, args.output_dir / "openai.fbin", "id", "emb", args.base_limit, args.batch_size)
    write_fbin([test_file], args.output_dir / "openai_query.fbin", "id", "emb", args.query_limit, args.batch_size)
    write_truth(
        neighbors_file,
        args.output_dir / "openai_query_COSINE_0.99_100.truth",
        "id",
        "neighbors_id",
        None,
        args.truth_topk,
        args.query_limit,
    )

    print(f"VectorDBBench parquet directory: {data_dir}", file=sys.stderr)
    print(f"Knowhere SVS reproduce data directory: {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
