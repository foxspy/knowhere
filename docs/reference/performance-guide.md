# Performance Guide

This document provides performance characteristics and resource usage guidelines for Knowhere indexes.

## Index Selection Guide

| Use Case | Recommended Index | Reason |
|----------|-------------------|--------|
| Small dataset (<1M), need 100% recall | Flat | Exact search, no index overhead |
| Low latency, high recall | HNSW | Fast graph traversal |
| Large dataset, balanced performance | IVF_FLAT | Good trade-off |
| Memory constrained | IVF_PQ, IVF_SQ8 | Compressed vectors |
| Billion-scale, limited RAM | DISKANN, AISAQ | Disk-based storage |
| Sparse vectors, text search | Sparse | Native sparse support |
| High throughput, GPU available | GPU (CUVS) | Parallel processing |

## Resource Usage

### Memory

| Index | Memory Formula | Notes |
|-------|---------------|-------|
| Flat | `n * d * sizeof(type)` | Full vectors in memory |
| HNSW | `n * (d * sizeof(type) + M * 8)` | Vectors + graph |
| IVF_FLAT | `n * d * sizeof(type) + overhead` | Vectors + cluster info |
| IVF_PQ | `n * m + codebook` | Compressed (m = subquantizers) |
| DISKANN | `graph on disk + cache` | Minimal RAM |

### Build Time Factors

- Dataset size (n)
- Vector dimension (d)
- Index parameters (M, nlist, etc.)
- Available CPU cores

---

*Last updated: 2026-01-26*
