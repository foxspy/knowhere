#!/usr/bin/env python3
"""Convert VectorDBBench OpenAI parquet files to fbin/truth files.

This is adapted from vecTool's python/convertor/parquet2fbin.py. It keeps the
same fbin layout used by vecTool:

    int32 rows, int32 dim, float32 vectors[rows][dim]

The truth output follows vecTool SearchResult layout:

    int32 nq, int32 topk, int64 ids[nq][topk], float32 distances[nq][topk]
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import polars as pl


def _vector_dim(parquet_paths: list[Path], vector_col: str) -> int:
    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1, columns=[vector_col]):
            vectors = batch.column(0).to_pylist()
            if vectors:
                return len(vectors[0])
    raise ValueError("unable to infer vector dimension")


def _row_count(parquet_paths: list[Path], limit: int) -> int:
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in parquet_paths)
    if limit > 0:
        return min(rows, limit)
    return rows


def write_fbin(
    parquet_paths: list[Path],
    output_path: Path,
    id_col: str,
    vector_col: str,
    limit: int,
    batch_size: int,
) -> None:
    if not parquet_paths:
        raise ValueError("no parquet input files")

    num_points = _row_count(parquet_paths, limit)
    if num_points == 0:
        raise ValueError("empty vector dataframe")

    dim = _vector_dim(parquet_paths, vector_col)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as writer:
        writer.write(struct.pack("<ii", num_points, dim))
        writer.truncate(8 + num_points * dim * np.dtype(np.float32).itemsize)
        seen = np.zeros(num_points, dtype=np.bool_)
        written = 0
        for path in parquet_paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size, columns=[id_col, vector_col]):
                ids = batch.column(0).to_pylist()
                vectors = batch.column(1).to_pylist()
                for row_id, vector in zip(ids, vectors, strict=True):
                    row_id = int(row_id)
                    if row_id < 0:
                        raise ValueError(f"{path} contains negative id {row_id}")
                    if row_id >= num_points:
                        if limit > 0:
                            continue
                        raise ValueError(f"{path} contains id {row_id}, but expected ids < {num_points}")
                    if seen[row_id]:
                        raise ValueError(f"{path} contains duplicate id {row_id}")
                    if len(vector) != dim:
                        raise ValueError(f"inconsistent vector dim: expected {dim}, got {len(vector)}")
                    writer.seek(8 + row_id * dim * np.dtype(np.float32).itemsize)
                    writer.write(np.asarray(vector, dtype=np.float32).tobytes(order="C"))
                    seen[row_id] = True
                    written += 1
        if written != num_points:
            missing = np.flatnonzero(~seen)[:10].tolist()
            raise ValueError(f"missing {num_points - written} vectors; first missing ids: {missing}")


def write_truth(
    parquet_path: Path,
    output_path: Path,
    id_col: str,
    neighbors_col: str,
    distance_col: str | None,
    topk: int,
    limit: int,
) -> None:
    columns = [id_col, neighbors_col]
    if distance_col:
        columns.append(distance_col)
    df = pl.read_parquet(parquet_path, columns=columns).sort(id_col)
    if limit > 0:
        df = df.head(limit)
    if df.height == 0:
        raise ValueError("empty truth dataframe")

    neighbors = df[neighbors_col].to_list()
    truth_k = topk if topk > 0 else len(neighbors[0])
    if truth_k <= 0:
        raise ValueError("truth topk must be positive")

    distances = None
    if distance_col:
        distances = df[distance_col].to_list()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as writer:
        writer.write(struct.pack("<ii", len(neighbors), truth_k))
        for row in neighbors:
            if len(row) < truth_k:
                raise ValueError(f"truth row has fewer than {truth_k} neighbors")
            writer.write(np.asarray(row[:truth_k], dtype=np.int64).tobytes(order="C"))
        for i, row in enumerate(neighbors):
            if distances is None:
                writer.write(np.zeros(truth_k, dtype=np.float32).tobytes(order="C"))
                continue
            dist_row = distances[i]
            if len(dist_row) < truth_k:
                raise ValueError(f"distance row has fewer than {truth_k} values")
            writer.write(np.asarray(dist_row[:truth_k], dtype=np.float32).tobytes(order="C"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", nargs="+", type=Path, required=True, help="VectorDBBench train parquet file(s)")
    parser.add_argument("--test", type=Path, required=True, help="VectorDBBench test.parquet file")
    parser.add_argument("--neighbors", type=Path, required=True, help="VectorDBBench neighbors.parquet file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for fbin/truth files")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--vector-col", default="emb")
    parser.add_argument("--neighbors-col", default="neighbors_id")
    parser.add_argument("--distance-col", default="", help="Optional neighbors distance column")
    parser.add_argument("--base-limit", type=int, default=-1)
    parser.add_argument("--query-limit", type=int, default=-1)
    parser.add_argument("--truth-topk", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--base-name", default="openai.fbin")
    parser.add_argument("--query-name", default="openai_query.fbin")
    parser.add_argument("--truth-name", default="openai_query_COSINE_0.99_100.truth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_fbin(args.train, args.output_dir / args.base_name, args.id_col, args.vector_col, args.base_limit, args.batch_size)
    write_fbin([args.test], args.output_dir / args.query_name, args.id_col, args.vector_col, args.query_limit, args.batch_size)
    write_truth(
        args.neighbors,
        args.output_dir / args.truth_name,
        args.id_col,
        args.neighbors_col,
        args.distance_col or None,
        args.truth_topk,
        args.query_limit,
    )


if __name__ == "__main__":
    main()
