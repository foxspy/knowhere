# SVS OpenAI500K Reproduction Workflow

This workflow is intended for SVS providers to debug the OpenAI500K `SVS_VAMANA_LEANVEC` recall issue directly in Knowhere. It does not require Milvus or a VectorDBBench benchmark run. It only uses VectorDBBench as the source of the OpenAI500K parquet dataset, converts that dataset to vecTool/Knowhere binary files, and runs a hidden Knowhere UT that directly builds and searches `SVS_VAMANA_LEANVEC`.

The reported issue is recall@10 drifting as low as `0.8933`, while this configuration is expected to be in the high-recall band. The target parameters are:

- `index_type=SVS_VAMANA_LEANVEC`
- `metric_type=COSINE`
- `dim=1536`
- `svs_graph_max_degree=64`
- `svs_construction_window_size=200`
- `svs_storage_kind=leanvec4x8`
- `svs_leanvec_dim=768`
- `svs_search_window_size=1000`
- `svs_search_buffer_capacity=1000`
- `topk=10`

## 1. Prepare VectorDBBench

Use the same VectorDBBench revision as the original reproduction if possible. This is only needed to download the parquet dataset; the actual reproduction test calls Knowhere directly.

```bash
git clone https://github.com/zilliztech/VectorDBBench.git
cd VectorDBBench
git checkout 0c20701725a84fbcd2a14b5d628c77cac2beb071
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## 2. Download and Convert OpenAI500K

From the Knowhere repository:

```bash
export DATA_DIR=/tmp/knowhere_svs_openai500k
. /path/to/VectorDBBench/.venv/bin/activate

python3 scripts/svs_reproduce/prepare_openai500k_from_vdbbench.py \
  --dataset-local-dir /tmp/vectordb_bench/dataset \
  --output-dir "$DATA_DIR" \
  --source s3 \
  --use-shuffled-data false
```

This downloads VectorDBBench's OpenAI500K parquet files under:

```text
/tmp/vectordb_bench/dataset/openai/openai_medium_500k/
```

and writes:

```text
$DATA_DIR/openai.fbin
$DATA_DIR/openai_query.fbin
$DATA_DIR/openai_query_COSINE_0.99_100.truth
```

The converter is adapted from vecTool's `python/convertor/parquet2fbin.py`. It streams parquet batches into fbin and writes each vector at row offset `id`, so Knowhere internal ids match VectorDBBench ground-truth ids even when VectorDBBench provides `shuffle_train.parquet`.

If the parquet files are already present, the lower-level converter can be run directly:

```bash
python3 scripts/svs_reproduce/parquet_to_fbin.py \
  --train /tmp/vectordb_bench/dataset/openai/openai_medium_500k/shuffle_train.parquet \
  --test /tmp/vectordb_bench/dataset/openai/openai_medium_500k/test.parquet \
  --neighbors /tmp/vectordb_bench/dataset/openai/openai_medium_500k/neighbors.parquet \
  --output-dir "$DATA_DIR"
```

## 3. Build Knowhere with SVS and UT

SVS LeanVec requires an Intel CPU and an SVS-enabled build.

```bash
make WITH_SVS=True WITH_UT=True
```

If your local Makefile does not pass `WITH_SVS` through to Conan/CMake, configure CMake with `-DWITH_SVS=ON -DWITH_UT=ON` or use the same build flow used by your Knowhere CI.

## 4. Run the Hidden Reproduction Test

```bash
export KNOWHERE_SVS_REPRO_DATA_DIR="$DATA_DIR"
export KNOWHERE_SVS_REPRO_ZERO_BITSET=true
export KNOWHERE_SVS_REPRO_MIN_RECALL=0.0

./build/Release/tests/ut/knowhere_tests \
  "Reproduce OpenAI500K SVS Vamana LeanVec recall"
```

The test prints a line similar to:

```text
SVS OpenAI500K reproduce: rows=500000, dim=1536, nq=1000, topk=10, recall=0.8933, ...
```

With `KNOWHERE_SVS_REPRO_ZERO_BITSET=true`, the test passes a vecTool/Milvus-style all-zero bitset with 500000 bits instead of an empty bitset. This is the condition that reproduces the reported low recall. With an empty bitset, the same index can return the high recall band around `0.9914`.

The test fails the final assertion if recall is below `KNOWHERE_SVS_REPRO_MIN_RECALL`:

```text
REQUIRE(recall >= min_recall)
```

Set `KNOWHERE_SVS_REPRO_MIN_RECALL=0.0` if you want the test to complete without failing while still printing the measured recall.

### m6id Validation Note

This branch was validated from scratch on an Intel m6id host using the steps above:

- this reproduction branch with empty bitset: direct Knowhere runs returned `recall=0.9914`;
- comparison at Knowhere `c55ad73b0fbed93cab2d5ff1874b358e7b4bc8ab`: direct build/search with empty bitset also returned `recall=0.9914`;
- adding vecTool-like string parameters, `data_path`, `Serialize -> Deserialize -> Search`, and vecTool-like load params still returned `recall=0.9914` with empty bitset;
- loading a previously saved vecTool index that vecTool reports as `0.893302` returned `recall=0.9914` with empty bitset, even when forcing the UT to load vecTool's `libknowhere.so` and `libsvs_runtime.so.0`;
- loading the same index with `KNOWHERE_SVS_REPRO_ZERO_BITSET=true` returned `recall=0.8933`, matching vecTool.
- running the full `Build -> Serialize -> Deserialize -> Search` path with `KNOWHERE_SVS_REPRO_ZERO_BITSET=true` returned `recall=0.8933`.

So the low recall is reproducible in the direct Knowhere test when the search call includes a non-empty all-zero bitset. The explicit search JSON is unchanged; the behavior difference comes from the bitset selector path.

## What This Rules Out

This reproduction path does not create a Milvus collection, does not compact segments, does not load QueryNode segments, and does not run VectorDBBench search. If it returns recall around `0.8933` with `KNOWHERE_SVS_REPRO_ZERO_BITSET=true`, the low-recall state is reproduced inside the Knowhere/SVS search path. If it returns the high band around `0.9914` with the same setting, compare the generated search line and confirm the test did not fall back to an empty bitset.

## Useful Overrides

```bash
export KNOWHERE_SVS_REPRO_BASE_FILE=openai.fbin
export KNOWHERE_SVS_REPRO_QUERY_FILE=openai_query.fbin
export KNOWHERE_SVS_REPRO_TRUTH_FILE=openai_query_COSINE_0.99_100.truth
export KNOWHERE_SVS_REPRO_TOPK=10
export KNOWHERE_SVS_REPRO_INDEX_VERSION=9
export KNOWHERE_SVS_REPRO_MAX_QUERIES=1000
export KNOWHERE_SVS_REPRO_STRING_PARAMS=true
export KNOWHERE_SVS_REPRO_ROUNDTRIP=true
export KNOWHERE_SVS_REPRO_INCLUDE_DATA_PATH=true
export KNOWHERE_SVS_REPRO_EXEC_OVER_BUILD_POOL=true
export KNOWHERE_SVS_REPRO_ZERO_BITSET=true
export KNOWHERE_SVS_REPRO_SEARCH_WINDOW_SIZE=1000
export KNOWHERE_SVS_REPRO_SEARCH_BUFFER_CAPACITY=1000
export KNOWHERE_SVS_REPRO_LOAD_INDEX_FILE=/path/to/existing_kw.index
```

`KNOWHERE_SVS_REPRO_MAX_BASE_ROWS` exists for smoke checks, but recall is only meaningful for the full 500K base set because the ground truth ids refer to the full VectorDBBench dataset.
