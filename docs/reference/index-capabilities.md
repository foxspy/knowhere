# Index Capabilities

This document describes the capabilities of each index type in Knowhere.

## Index Families

| Family | Type | Metrics | Build Option |
|--------|------|---------|--------------|
| Flat | Brute force | L2, IP, COSINE | - |
| HNSW | Graph | L2, IP, COSINE | - |
| IVF | Inverted file | L2, IP, COSINE | - |
| SCANN | IVF + Anisotropic | L2, IP, COSINE | - |
| DISKANN | Disk graph | L2, IP, COSINE | `with_diskann` |
| AISAQ | Disk + PQ | L2, IP, COSINE | `with_diskann` |
| SPARSE_INVERTED_INDEX | Inverted | IP, BM25 | - |
| SPARSE_WAND | Inverted (WAND) | IP, BM25 | - |
| MinHash | LSH | Jaccard | - |
| GPU (CUVS) | CUDA | L2, IP | `with_cuvs` (see GPU Series below) |

## HNSW Series

| Variant | Quantization | Data Types |
|---------|--------------|------------|
| HNSW | None | fp32, fp16, bf16, int8 |
| HNSW_SQ | Scalar (8-bit) | fp32, fp16, bf16, int8 |
| HNSW_PQ | Product | fp32, fp16, bf16, int8 |
| HNSW_PRQ | Product Residual | fp32, fp16, bf16, int8 |

## IVF Series

Growing variants (concurrent read/write): IVF_FLAT_CC, IVF_SQ_CC support real-time insertion.

| Variant | Quantization | Data Types |
|---------|--------------|------------|
| IVF_FLAT | None | fp32, fp16, bf16, int8 |
| IVF_SQ | Scalar | fp32, fp16, bf16, int8 |
| IVF_PQ | Product | fp32, fp16, bf16, int8 |
| IVF_RABITQ | RaBitQ | fp32, fp16, bf16 |
| BIN_IVF_FLAT | None | binary |

## Quantization Types

| Type | Abbreviation | Description | Memory Reduction |
|------|--------------|-------------|------------------|
| Scalar 4-bit | SQ4U | 4-bit uniform scalar quantization | ~8x |
| Scalar 6-bit | SQ6 | 6-bit scalar quantization | ~5x |
| Scalar 8-bit | SQ8 | 8-bit scalar quantization | ~4x |
| Product | PQ | Vector split into subvectors | ~8-32x |
| Product Residual | PRQ | PQ applied to residuals | ~8-32x |
| RaBitQ | RaBitQ | Random Binary Quantization | ~32x |

## Other Indexes

| Index | Data Types | Notes |
|-------|------------|-------|
| Flat | fp32, fp16, bf16, int8, binary | Brute force, 100% recall |
| SCANN | fp32, fp16, bf16, int8 | Google's anisotropic quantization (SCANN_DVR variant with optimized reranking) |
| DISKANN | fp32, fp16, bf16 | Billion-scale on SSD |
| AISAQ | fp32, fp16, bf16 | DISKANN + PQ optimization |
| SPARSE_INVERTED_INDEX | sparse_u32_f32 | Standard inverted index, supports BM25 |
| SPARSE_WAND | sparse_u32_f32 | WAND algorithm for optimized top-k retrieval |
| MinHash | binary | Jaccard similarity |

## GPU (CUVS) Series

| Variant | Data Types | Notes |
|---------|------------|-------|
| GPU_CAGRA | fp32, fp16, int8, binary | Graph-based (CAGRA algorithm) |
| GPU_BRUTE_FORCE | fp32, fp16 | Exact search |
| GPU_IVF_FLAT | fp32, fp16, int8 | IVF without quantization |
| GPU_IVF_PQ | fp32, fp16, int8 | IVF with product quantization |

---

*Last updated: 2026-01-26*
