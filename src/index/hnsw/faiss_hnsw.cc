// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#include <faiss/utils/Heap.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "common/metric.h"
#include "faiss/IndexBinaryHNSW.h"
#include "faiss/IndexCosine.h"
#include "faiss/IndexHNSW.h"
#include "faiss/IndexRefine.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "faiss/index_io.h"
#include "index/hnsw/faiss_hnsw_config.h"
#include "index/hnsw/impl/IndexBruteForceWrapper.h"
#include "index/hnsw/impl/IndexHNSWWrapper.h"
#include "index/hnsw/impl/IndexWrapperCosine.h"
#include "io/memory_io.h"
#include "knowhere/bitsetview_idselector.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/comp/thread_pool.h"
#include "knowhere/comp/time_recorder.h"
#include "knowhere/config.h"
#include "knowhere/expected.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/index/index_node_data_mock_wrapper.h"
#include "knowhere/log.h"
#include "knowhere/range_util.h"
#include "knowhere/utils.h"

#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
#include "knowhere/prometheus_client.h"
#endif

namespace knowhere {

//
class BaseFaissIndexNode : public IndexNode {
 public:
    BaseFaissIndexNode(const int32_t& /*version*/, const Object& object) {
        build_pool = ThreadPool::GetGlobalBuildThreadPool();
        search_pool = ThreadPool::GetGlobalSearchThreadPool();
    }

    //
    Status
    Train(const DataSetPtr dataset, const Config& cfg) override {
        // config
        const BaseConfig& base_cfg = static_cast<const FaissHnswConfig&>(cfg);

        // use build_pool_ to make sure the OMP threads spawned by index_->train etc
        // can inherit the low nice value of threads in build_pool_.
        auto tryObj = build_pool
                          ->push([&] {
                              std::unique_ptr<ThreadPool::ScopedOmpSetter> setter;
                              if (base_cfg.num_build_thread.has_value()) {
                                  setter =
                                      std::make_unique<ThreadPool::ScopedOmpSetter>(base_cfg.num_build_thread.value());
                              } else {
                                  setter = std::make_unique<ThreadPool::ScopedOmpSetter>();
                              }

                              return TrainInternal(dataset, cfg);
                          })
                          .getTry();

        if (!tryObj.hasValue()) {
            LOG_KNOWHERE_WARNING_ << "faiss internal error: " << tryObj.exception().what();
            return Status::faiss_inner_error;
        }

        return tryObj.value();
    }

    Status
    Add(const DataSetPtr dataset, const Config& cfg) override {
        const BaseConfig& base_cfg = static_cast<const FaissHnswConfig&>(cfg);

        // use build_pool_ to make sure the OMP threads spawned by index_->train etc
        // can inherit the low nice value of threads in build_pool_.
        auto tryObj = build_pool
                          ->push([&] {
                              std::unique_ptr<ThreadPool::ScopedOmpSetter> setter;
                              if (base_cfg.num_build_thread.has_value()) {
                                  setter =
                                      std::make_unique<ThreadPool::ScopedOmpSetter>(base_cfg.num_build_thread.value());
                              } else {
                                  setter = std::make_unique<ThreadPool::ScopedOmpSetter>();
                              }

                              return AddInternal(dataset, cfg);
                          })
                          .getTry();

        if (!tryObj.hasValue()) {
            LOG_KNOWHERE_WARNING_ << "faiss internal error: " << tryObj.exception().what();
            return Status::faiss_inner_error;
        }

        return tryObj.value();
    }

    int64_t
    Size() const override {
        // todo
        return 0;
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
        // todo
        return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
    }

 protected:
    std::shared_ptr<ThreadPool> build_pool;
    std::shared_ptr<ThreadPool> search_pool;

    // train impl
    virtual Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) = 0;

    // add impl
    virtual Status
    AddInternal(const DataSetPtr dataset, const Config& cfg) = 0;
};

//
class BaseFaissRegularIndexNode : public BaseFaissIndexNode {
 public:
    BaseFaissRegularIndexNode(const int32_t& version, const Object& object)
        : BaseFaissIndexNode(version, object), index{nullptr} {
    }

    Status
    Serialize(BinarySet& binset) const override {
        if (index == nullptr) {
            return Status::empty_index;
        }

        try {
            MemoryIOWriter writer;
            faiss::write_index(index.get(), &writer);

            std::shared_ptr<uint8_t[]> data(writer.data());
            binset.Append(Type(), data, writer.tellg());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        auto binary = binset.GetByName(Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }

        MemoryIOReader reader(binary->data.get(), binary->size);
        try {
            auto read_index = std::unique_ptr<faiss::Index>(faiss::read_index(&reader));
            index.reset(read_index.release());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) override {
        auto cfg = static_cast<const knowhere::BaseConfig&>(config);

        int io_flags = 0;
        if (cfg.enable_mmap.value()) {
            io_flags |= faiss::IO_FLAG_MMAP;
        }

        try {
            auto read_index = std::unique_ptr<faiss::Index>(faiss::read_index(filename.data(), io_flags));
            index.reset(read_index.release());
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    //
    int64_t
    Dim() const override {
        if (index == nullptr) {
            return -1;
        }

        return index->d;
    }

    int64_t
    Count() const override {
        if (index == nullptr) {
            return -1;
        }

        // total number of indexed vectors
        return index->ntotal;
    }

 protected:
    std::unique_ptr<faiss::Index> index;

    Status
    AddInternal(const DataSetPtr dataset, const Config&) override {
        if (this->index == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to an empty index.";
            return Status::empty_index;
        }

        auto data = dataset->GetTensor();
        auto rows = dataset->GetRows();
        try {
            this->index->add(rows, reinterpret_cast<const float*>(data));
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }
};

//
enum class DataFormatEnum { fp32, fp16, bf16 };

template <typename T>
struct DataType2EnumHelper {};

template <>
struct DataType2EnumHelper<knowhere::fp32> {
    static constexpr DataFormatEnum value = DataFormatEnum::fp32;
};
template <>
struct DataType2EnumHelper<knowhere::fp16> {
    static constexpr DataFormatEnum value = DataFormatEnum::fp16;
};
template <>
struct DataType2EnumHelper<knowhere::bf16> {
    static constexpr DataFormatEnum value = DataFormatEnum::bf16;
};

template <typename T>
static constexpr DataFormatEnum datatype_v = DataType2EnumHelper<T>::value;

//
namespace {

//
bool
convert_rows_to_fp32(const void* const __restrict src_in, float* const __restrict dst,
                     const DataFormatEnum src_data_format, const size_t start_row, const size_t nrows,
                     const size_t dim) {
    if (src_data_format == DataFormatEnum::fp16) {
        const knowhere::fp16* const src = reinterpret_cast<const knowhere::fp16*>(src_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i] = (float)(src[i + start_row * dim]);
        }

        return true;
    } else if (src_data_format == DataFormatEnum::bf16) {
        const knowhere::bf16* const src = reinterpret_cast<const knowhere::bf16*>(src_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i] = (float)(src[i + start_row * dim]);
        }

        return true;
    } else if (src_data_format == DataFormatEnum::fp32) {
        const knowhere::fp32* const src = reinterpret_cast<const knowhere::fp32*>(src_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i] = src[i + start_row * dim];
        }

        return true;
    } else {
        // unknown
        return false;
    }
}

bool
convert_rows_from_fp32(const float* const __restrict src, void* const __restrict dst_in,
                       const DataFormatEnum dst_data_format, const size_t start_row, const size_t nrows,
                       const size_t dim) {
    if (dst_data_format == DataFormatEnum::fp16) {
        knowhere::fp16* const dst = reinterpret_cast<knowhere::fp16*>(dst_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i + start_row * dim] = (knowhere::fp16)src[i];
        }

        return true;
    } else if (dst_data_format == DataFormatEnum::bf16) {
        knowhere::bf16* const dst = reinterpret_cast<knowhere::bf16*>(dst_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i + start_row * dim] = (knowhere::bf16)src[i];
        }

        return true;
    } else if (dst_data_format == DataFormatEnum::fp32) {
        knowhere::fp32* const dst = reinterpret_cast<knowhere::fp32*>(dst_in);
        for (size_t i = 0; i < nrows * dim; i++) {
            dst[i + start_row * dim] = src[i];
        }

        return true;
    } else {
        // unknown
        return false;
    }
}

//
DataSetPtr
convert_ds_to_float(const DataSetPtr& src, DataFormatEnum data_format) {
    if (data_format == DataFormatEnum::fp32) {
        return src;
    } else if (data_format == DataFormatEnum::fp16) {
        return ConvertFromDataTypeIfNeeded<knowhere::fp16>(src);
    } else if (data_format == DataFormatEnum::bf16) {
        return ConvertFromDataTypeIfNeeded<knowhere::bf16>(src);
    }

    return nullptr;
}

Status
add_to_index(faiss::Index* const __restrict index, const DataSetPtr& dataset, const DataFormatEnum data_format) {
    const auto* data = dataset->GetTensor();
    const auto rows = dataset->GetRows();
    const auto dim = dataset->GetDim();

    if (data_format == DataFormatEnum::fp32) {
        // add as is
        index->add(rows, reinterpret_cast<const float*>(data));
    } else {
        // convert data into float in pieces and add to the index
        constexpr int64_t n_tmp_rows = 4096;
        std::unique_ptr<float[]> tmp = std::make_unique<float[]>(n_tmp_rows * dim);

        for (int64_t irow = 0; irow < rows; irow += n_tmp_rows) {
            const int64_t start_row = irow;
            const int64_t end_row = std::min(rows, start_row + n_tmp_rows);
            const int64_t count_rows = end_row - start_row;

            if (!convert_rows_to_fp32(data, tmp.get(), data_format, start_row, count_rows, dim)) {
                LOG_KNOWHERE_ERROR_ << "Unsupported data format";
                return Status::invalid_args;
            }

            // add
            index->add(count_rows, tmp.get());
        }
    }

    return Status::success;
}

// IndexFlat and IndexFlatCosine contain raw fp32 data
// IndexScalarQuantizer and IndexScalarQuantizerCosine may contain rar bf16 and fp16 data
//
// returns nullopt if an input index does not contain raw bf16, fp16 or fp32 data
std::optional<DataFormatEnum>
get_index_data_format(const faiss::Index* index) {
    // empty
    if (index == nullptr) {
        return std::nullopt;
    }

    // is it flat?
    // note: IndexFlatCosine preserves the original data, no cosine norm is applied
    auto index_flat = dynamic_cast<const faiss::IndexFlat*>(index);
    if (index_flat != nullptr) {
        return DataFormatEnum::fp32;
    }

    // is it sq?
    // note: IndexScalarQuantizerCosine preserves the original data, no cosine norm is appliesd
    auto index_sq = dynamic_cast<const faiss::IndexScalarQuantizer*>(index);
    if (index_sq != nullptr) {
        if (index_sq->sq.qtype == faiss::ScalarQuantizer::QT_bf16) {
            return DataFormatEnum::bf16;
        } else if (index_sq->sq.qtype == faiss::ScalarQuantizer::QT_fp16) {
            return DataFormatEnum::fp16;
        } else {
            return std::nullopt;
        }
    }

    // some other index
    return std::nullopt;
}

}  // namespace

//
class BaseFaissRegularIndexHNSWNode : public BaseFaissRegularIndexNode {
 public:
    BaseFaissRegularIndexHNSWNode(const int32_t& version, const Object& object, DataFormatEnum data_format_in)
        : BaseFaissRegularIndexNode(version, object), data_format{data_format_in} {
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        if (this->index == nullptr) {
            return false;
        }

        // check whether there is an index to reconstruct a raw data from
        return (GetIndexToReconstructRawDataFrom() != nullptr);
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSetPtr dataset) const override {
        if (index == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        if (!index->is_trained) {
            return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
        }

        // an index that is used for reconstruction
        const faiss::Index* index_to_reconstruct_from = GetIndexToReconstructRawDataFrom();

        // check whether raw data is available
        if (index_to_reconstruct_from == nullptr) {
            return expected<DataSetPtr>::Err(
                Status::invalid_index_error,
                "The index does not contain a raw data, cannot proceed with GetVectorByIds");
        }

        // perform reconstruction
        auto dim = Dim();
        auto rows = dataset->GetRows();
        auto ids = dataset->GetIds();

        try {
            if (data_format == DataFormatEnum::fp32) {
                // perform a direct reconstruction for fp32 data
                auto data = std::make_unique<float[]>(dim * rows);

                for (int64_t i = 0; i < rows; i++) {
                    const int64_t id = ids[i];
                    assert(id >= 0 && id < index->ntotal);
                    index_to_reconstruct_from->reconstruct(id, data.get() + i * dim);
                }

                return GenResultDataSet(rows, dim, std::move(data));
            } else if (data_format == DataFormatEnum::fp16) {
                auto data = std::make_unique<knowhere::fp16[]>(dim * rows);

                // faiss produces fp32 data format, we need some other format.
                // Let's create a temporary fp32 buffer for this.
                auto tmp = std::make_unique<float[]>(dim);

                for (int64_t i = 0; i < rows; i++) {
                    const int64_t id = ids[i];
                    assert(id >= 0 && id < index->ntotal);
                    index_to_reconstruct_from->reconstruct(id, tmp.get());

                    if (!convert_rows_from_fp32(tmp.get(), data.get(), data_format, i, 1, dim)) {
                        return expected<DataSetPtr>::Err(Status::invalid_args, "Unsupported data format");
                    }
                }

                return GenResultDataSet(rows, dim, std::move(data));
            } else if (data_format == DataFormatEnum::bf16) {
                auto data = std::make_unique<knowhere::bf16[]>(dim * rows);

                // faiss produces fp32 data format, we need some other format.
                // Let's create a temporary fp32 buffer for this.
                auto tmp = std::make_unique<float[]>(dim);

                for (int64_t i = 0; i < rows; i++) {
                    const int64_t id = ids[i];
                    assert(id >= 0 && id < index->ntotal);
                    index_to_reconstruct_from->reconstruct(id, tmp.get());

                    if (!convert_rows_from_fp32(tmp.get(), data.get(), data_format, i, 1, dim)) {
                        return expected<DataSetPtr>::Err(Status::invalid_args, "Unsupported data format");
                    }
                }

                return GenResultDataSet(rows, dim, std::move(data));
            } else {
                return expected<DataSetPtr>::Err(Status::invalid_args, "Unsupported data format");
            }
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }
    }

    expected<DataSetPtr>
    Search(const DataSetPtr dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (this->index == nullptr) {
            return expected<DataSetPtr>::Err(Status::empty_index, "index not loaded");
        }
        if (!this->index->is_trained) {
            return expected<DataSetPtr>::Err(Status::index_not_trained, "index not trained");
        }

        const auto dim = dataset->GetDim();
        const auto rows = dataset->GetRows();
        const auto* data = dataset->GetTensor();

        const auto hnsw_cfg = static_cast<const FaissHnswConfig&>(cfg);
        const auto k = hnsw_cfg.k.value();
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), knowhere::metric::COSINE);

        const bool whether_bf_search = WhetherPerformBruteForceSearch(hnsw_cfg, bitset);

        feder::hnsw::FederResultUniq feder_result;
        if (hnsw_cfg.trace_visit.value()) {
            if (rows != 1) {
                return expected<DataSetPtr>::Err(Status::invalid_args, "a single query vector is required");
            }
            feder_result = std::make_unique<feder::hnsw::FederResult>();
        }

        auto ids = std::make_unique<faiss::idx_t[]>(rows * k);
        auto distances = std::make_unique<float[]>(rows * k);
        try {
            std::vector<folly::Future<folly::Unit>> futs;
            futs.reserve(rows);
            for (int64_t i = 0; i < rows; ++i) {
                futs.emplace_back(search_pool->push([&, idx = i] {
                    // 1 thread per element
                    ThreadPool::ScopedOmpSetter setter(1);

                    // set up a query
                    // const float* cur_query = (const float*)data + idx * dim;

                    const float* cur_query = nullptr;

                    std::vector<float> cur_query_tmp(dim);
                    if (data_format == DataFormatEnum::fp32) {
                        cur_query = (const float*)data + idx * dim;
                    } else {
                        convert_rows_to_fp32(data, cur_query_tmp.data(), data_format, idx, 1, dim);
                        cur_query = cur_query_tmp.data();
                    }

                    // set up local results
                    faiss::idx_t* const __restrict local_ids = ids.get() + k * idx;
                    float* const __restrict local_distances = distances.get() + k * idx;

                    // set up faiss search parameters
                    knowhere::SearchParametersHNSWWrapper hnsw_search_params;
                    if (hnsw_cfg.ef.has_value()) {
                        hnsw_search_params.efSearch = hnsw_cfg.ef.value();
                    }
                    // do not collect HNSW stats
                    hnsw_search_params.hnsw_stats = nullptr;
                    // set up feder
                    hnsw_search_params.feder = feder_result.get();
                    // set up kAlpha
                    hnsw_search_params.kAlpha = bitset.filter_ratio() * 0.7f;

                    // set up a selector
                    BitsetViewIDSelector bw_idselector(bitset);
                    faiss::IDSelector* id_selector = (bitset.empty()) ? nullptr : &bw_idselector;

                    hnsw_search_params.sel = id_selector;

                    // use knowhere-based search by default
                    const bool override_faiss_search = hnsw_cfg.override_faiss_search.value_or(true);

                    // check if we have a refine available.
                    faiss::IndexRefine* const index_refine = dynamic_cast<faiss::IndexRefine*>(index.get());

                    if (index_refine != nullptr) {
                        // yes, it is possible to refine results.

                        // cast a base index to IndexHNSW-based index
                        faiss::IndexHNSW* const index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index_refine->base_index);

                        if (index_hnsw == nullptr) {
                            // this is unexpected
                            throw std::runtime_error("Expecting faiss::IndexHNSW");
                        }

                        // pick a wrapper for hnsw which does not own indices
                        knowhere::IndexHNSWWrapper wrapper_hnsw_search(index_hnsw);
                        knowhere::IndexBruteForceWrapper wrapper_bf(index_hnsw);

                        faiss::Index* base_wrapper = nullptr;
                        if (!override_faiss_search) {
                            // use the original index, no wrappers
                            base_wrapper = index_hnsw;
                        } else if (whether_bf_search) {
                            // use brute-force wrapper
                            base_wrapper = &wrapper_bf;
                        } else {
                            // use hnsw-search wrapper
                            base_wrapper = &wrapper_hnsw_search;
                        }

                        // check if used wants a refined result
                        if (hnsw_cfg.refine_k.has_value()) {
                            // yes, a user wants to perform a refine

                            // set up search parameters
                            faiss::IndexRefineSearchParameters refine_params;
                            refine_params.k_factor = hnsw_cfg.refine_k.value();
                            // a refine procedure itself does not need to care about filtering
                            refine_params.sel = nullptr;
                            refine_params.base_index_params = &hnsw_search_params;

                            // is it a cosine index?
                            if (index_hnsw->storage->is_cosine && is_cosine) {
                                // yes, wrap both base and refine index
                                knowhere::IndexWrapperCosine cosine_wrapper(
                                    index_refine->refine_index,
                                    dynamic_cast<faiss::HasInverseL2Norms*>(index_hnsw->storage)
                                        ->get_inverse_l2_norms());

                                // create a temporary refine index which does not own
                                faiss::IndexRefine tmp_refine(base_wrapper, &cosine_wrapper);

                                // perform a search
                                tmp_refine.search(1, cur_query, k, local_distances, local_ids, &refine_params);
                            } else {
                                // no, wrap base index only.

                                // create a temporary refine index which does not own
                                faiss::IndexRefine tmp_refine(base_wrapper, index_refine->refine_index);

                                // perform a search
                                tmp_refine.search(1, cur_query, k, local_distances, local_ids, &refine_params);
                            }
                        } else {
                            // no, a user wants to skip a refine

                            // perform a search
                            base_wrapper->search(1, cur_query, k, local_distances, local_ids, &hnsw_search_params);
                        }
                    } else {
                        // there's no refining available

                        // check if refine is required
                        if (hnsw_cfg.refine_k.has_value()) {
                            // this is not possible, throw an error
                            throw std::runtime_error("Refine is not provided by the index.");
                        }

                        // cast to IndexHNSW-based index
                        faiss::IndexHNSW* const index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index.get());

                        if (index_hnsw == nullptr) {
                            // this is unexpected
                            throw std::runtime_error("Expecting faiss::IndexHNSW");
                        }

                        // pick a wrapper for hnsw which does not own indices
                        knowhere::IndexHNSWWrapper wrapper_hnsw_search(index_hnsw);
                        knowhere::IndexBruteForceWrapper wrapper_bf(index_hnsw);

                        faiss::Index* wrapper = nullptr;
                        if (!override_faiss_search) {
                            // use the original index, no wrappers
                            wrapper = index_hnsw;
                        } else if (whether_bf_search) {
                            // use brute-force wrapper
                            wrapper = &wrapper_bf;
                        } else {
                            // use hnsw-search wrapper
                            wrapper = &wrapper_hnsw_search;
                        }

                        // perform a search
                        wrapper->search(1, cur_query, k, local_distances, local_ids, &hnsw_search_params);
                    }
                }));
            }

            // wait for the completion
            WaitAllSuccess(futs);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return expected<DataSetPtr>::Err(Status::faiss_inner_error, e.what());
        }

        auto res = GenResultDataSet(rows, k, std::move(ids), std::move(distances));

        // set visit_info json string into result dataset
        if (feder_result != nullptr) {
            Json json_visit_info, json_id_set;
            nlohmann::to_json(json_visit_info, feder_result->visit_info_);
            nlohmann::to_json(json_id_set, feder_result->id_set_);
            res->SetJsonInfo(json_visit_info.dump());
            res->SetJsonIdSet(json_id_set.dump());
        }

        return res;
    }

 protected:
    DataFormatEnum data_format;

    // Decides whether a brute force should be used instead of a regular HNSW search.
    // This may be applicable in case of very large topk values or
    //   extremely high filtering levels.
    bool
    WhetherPerformBruteForceSearch(const BaseConfig& cfg, const BitsetView& bitset) const {
        constexpr float kHnswSearchKnnBFFilterThreshold = 0.93f;
        // constexpr float kHnswSearchRangeBFFilterThreshold = 0.97f;
        constexpr float kHnswSearchBFTopkThreshold = 0.5f;

        auto k = cfg.k.value();

        if (k >= (index->ntotal * kHnswSearchBFTopkThreshold)) {
            return true;
        }

        if (!bitset.empty()) {
            const size_t filtered_out_num = bitset.count();
#if defined(NOT_COMPILE_FOR_SWIG) && !defined(KNOWHERE_WITH_LIGHT)
            double ratio = ((double)filtered_out_num) / bitset.size();
            knowhere::knowhere_hnsw_bitset_ratio.Observe(ratio);
#endif
            if (filtered_out_num >= (index->ntotal * kHnswSearchKnnBFFilterThreshold) ||
                k >= (index->ntotal - filtered_out_num) * kHnswSearchBFTopkThreshold) {
                return true;
            }
        }

        // the default value
        return false;
    }

    Status
    AddInternal(const DataSetPtr dataset, const Config&) override {
        if (index == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to an empty index.";
            return Status::empty_index;
        }

        auto rows = dataset->GetRows();
        try {
            LOG_KNOWHERE_INFO_ << "Adding " << rows << " to HNSW Index";

            auto status = add_to_index(index.get(), dataset, data_format);
            if (status != Status::success) {
                return status;
            }

        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }

    const faiss::Index*
    GetIndexToReconstructRawDataFrom() const {
        if (index == nullptr) {
            return nullptr;
        }

        // an index that is used for reconstruction
        const faiss::Index* index_to_reconstruct_from = nullptr;

        // check whether our index uses refine
        auto index_refine = dynamic_cast<const faiss::IndexRefine*>(index.get());
        if (index_refine == nullptr) {
            // non-refined index

            // cast as IndexHNSW
            auto index_hnsw = dynamic_cast<const faiss::IndexHNSW*>(index.get());
            if (index_hnsw == nullptr) {
                // this is unexpected, we expect IndexHNSW
                return nullptr;
            }

            // storage index is the one that holds the raw data
            auto index_data_format = get_index_data_format(index_hnsw->storage);

            // make sure that its data format matches our input format
            if (index_data_format.has_value() && index_data_format.value() == data_format) {
                index_to_reconstruct_from = index_hnsw->storage;
            }
        } else {
            // refined index

            // refine index holds the raw data
            auto index_data_format = get_index_data_format(index_refine->refine_index);

            // make sure that its data format matches our input format
            if (index_data_format.has_value() && index_data_format.value() == data_format) {
                index_to_reconstruct_from = index_refine->refine_index;
            }
        }

        // done
        return index_to_reconstruct_from;
    }
};

//
class BaseFaissRegularIndexHNSWFlatNode : public BaseFaissRegularIndexHNSWNode {
 public:
    BaseFaissRegularIndexHNSWFlatNode(const int32_t& version, const Object& object, DataFormatEnum data_format)
        : BaseFaissRegularIndexHNSWNode(version, object, data_format) {
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswFlatConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FAISS_HNSW_FLAT;
    }

 protected:
    Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) override {
        // number of rows
        auto rows = dataset->GetRows();
        // dimensionality of the data
        auto dim = dataset->GetDim();
        // data
        auto data = dataset->GetTensor();

        // config
        auto hnsw_cfg = static_cast<const FaissHnswFlatConfig&>(cfg);

        auto metric = Str2FaissMetricType(hnsw_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }

        // create an index
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE);

        std::unique_ptr<faiss::IndexHNSW> hnsw_index;
        if (is_cosine) {
            if (data_format == DataFormatEnum::fp32) {
                hnsw_index = std::make_unique<faiss::IndexHNSWFlatCosine>(dim, hnsw_cfg.M.value());
            } else if (data_format == DataFormatEnum::fp16) {
                hnsw_index = std::make_unique<faiss::IndexHNSWSQCosine>(dim, faiss::ScalarQuantizer::QT_fp16,
                                                                        hnsw_cfg.M.value());
            } else if (data_format == DataFormatEnum::bf16) {
                hnsw_index = std::make_unique<faiss::IndexHNSWSQCosine>(dim, faiss::ScalarQuantizer::QT_bf16,
                                                                        hnsw_cfg.M.value());
            } else {
                LOG_KNOWHERE_ERROR_ << "Unsupported metric type: " << hnsw_cfg.metric_type.value();
                return Status::invalid_metric_type;
            }
        } else {
            if (data_format == DataFormatEnum::fp32) {
                hnsw_index = std::make_unique<faiss::IndexHNSWFlat>(dim, hnsw_cfg.M.value(), metric.value());
            } else if (data_format == DataFormatEnum::fp16) {
                hnsw_index = std::make_unique<faiss::IndexHNSWSQ>(dim, faiss::ScalarQuantizer::QT_fp16,
                                                                  hnsw_cfg.M.value(), metric.value());
            } else if (data_format == DataFormatEnum::bf16) {
                hnsw_index = std::make_unique<faiss::IndexHNSWSQ>(dim, faiss::ScalarQuantizer::QT_bf16,
                                                                  hnsw_cfg.M.value(), metric.value());
            } else {
                LOG_KNOWHERE_ERROR_ << "Unsupported metric type: " << hnsw_cfg.metric_type.value();
                return Status::invalid_metric_type;
            }
        }

        hnsw_index->hnsw.efConstruction = hnsw_cfg.efConstruction.value();

        // train
        LOG_KNOWHERE_INFO_ << "Training HNSW Index";

        // this function does nothing for the given parameters and indices.
        //   as a result, I'm just keeping it to have is_trained set to true.
        // WARNING: this may cause problems if ->train() performs some action
        //   based on the data in the future. Otherwise, data needs to be
        //   converted into float*.
        hnsw_index->train(rows, (const float*)data);

        // done
        index = std::move(hnsw_index);
        return Status::success;
    }
};

template <typename DataType>
class BaseFaissRegularIndexHNSWFlatNodeTemplate : public BaseFaissRegularIndexHNSWFlatNode {
 public:
    BaseFaissRegularIndexHNSWFlatNodeTemplate(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWFlatNode(version, object, datatype_v<DataType>) {
    }
};

namespace {

// a supporting function
expected<faiss::ScalarQuantizer::QuantizerType>
get_sq_quantizer_type(const std::string& sq_type) {
    std::map<std::string, faiss::ScalarQuantizer::QuantizerType> sq_types = {{"sq6", faiss::ScalarQuantizer::QT_6bit},
                                                                             {"sq8", faiss::ScalarQuantizer::QT_8bit},
                                                                             {"fp16", faiss::ScalarQuantizer::QT_fp16},
                                                                             {"bf16", faiss::ScalarQuantizer::QT_bf16}};

    // todo: tolower
    auto sq_type_tolower = str_to_lower(sq_type);
    auto itr = sq_types.find(sq_type_tolower);
    if (itr == sq_types.cend()) {
        return expected<faiss::ScalarQuantizer::QuantizerType>::Err(
            Status::invalid_args, fmt::format("invalid scalar quantizer type ({})", sq_type_tolower));
    }

    return itr->second;
}

expected<bool>
is_flat_refine(const std::optional<std::string>& refine_type) {
    // grab a type of a refine index
    if (!refine_type.has_value()) {
        return true;
    };

    // todo: tolower
    std::string refine_type_tolower = str_to_lower(refine_type.value());
    if (refine_type_tolower == "fp32" || refine_type_tolower == "flat") {
        return true;
    };

    // parse
    auto refine_sq_type = get_sq_quantizer_type(refine_type_tolower);
    if (!refine_sq_type.has_value()) {
        LOG_KNOWHERE_ERROR_ << "Invalid refine type: " << refine_type.value();
        return expected<bool>::Err(Status::invalid_args, fmt::format("invalid refine type ({})", refine_type.value()));
    }

    return false;
}

// pick a refine index
expected<std::unique_ptr<faiss::Index>>
pick_refine_index(const DataFormatEnum data_format, const std::optional<std::string>& refine_type,
                  std::unique_ptr<faiss::IndexHNSW>&& hnsw_index) {
    // yes

    // grab a type of a refine index
    expected<bool> is_fp32_flat = is_flat_refine(refine_type);
    if (!is_fp32_flat.has_value()) {
        return expected<std::unique_ptr<faiss::Index>>::Err(Status::invalid_args, "");
    }

    const bool is_fp32_flat_v = is_fp32_flat.value();

    // check input data_format
    if (data_format == DataFormatEnum::fp16) {
        // make sure that we're using fp16 refine
        auto refine_sq_type = get_sq_quantizer_type(refine_type.value());
        if (!(refine_sq_type.has_value() &&
              (refine_sq_type.value() != faiss::ScalarQuantizer::QT_bf16 && !is_fp32_flat_v))) {
            LOG_KNOWHERE_ERROR_ << "fp16 input data does not accept bf16 or fp32 as a refine index.";
            return expected<std::unique_ptr<faiss::Index>>::Err(
                Status::invalid_args, "fp16 input data does not accept bf16 or fp32 as a refine index.");
        }
    }

    if (data_format == DataFormatEnum::bf16) {
        // make sure that we're using bf16 refine
        auto refine_sq_type = get_sq_quantizer_type(refine_type.value());
        if (!(refine_sq_type.has_value() &&
              (refine_sq_type.value() != faiss::ScalarQuantizer::QT_fp16 && !is_fp32_flat_v))) {
            LOG_KNOWHERE_ERROR_ << "bf16 input data does not accept fp16 or fp32 as a refine index.";
            return expected<std::unique_ptr<faiss::Index>>::Err(
                Status::invalid_args, "bf16 input data does not accept fp16 or fp32 as a refine index.");
        }
    }

    // build
    std::unique_ptr<faiss::IndexHNSW> local_hnsw_index = std::move(hnsw_index);

    // either build flat or sq
    if (is_fp32_flat_v) {
        // build IndexFlat as a refine
        auto refine_index = std::make_unique<faiss::IndexRefineFlat>(local_hnsw_index.get());

        // let refine_index to own everything
        refine_index->own_fields = true;
        local_hnsw_index.release();

        // reassign
        return refine_index;
    } else {
        // being IndexScalarQuantizer as a refine
        auto refine_sq_type = get_sq_quantizer_type(refine_type.value());

        // a redundant check
        if (!refine_sq_type.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid refine type: " << refine_type.value();
            return expected<std::unique_ptr<faiss::Index>>::Err(
                Status::invalid_args, fmt::format("invalid refine type ({})", refine_type.value()));
        }

        // create an sq
        auto sq_refine = std::make_unique<faiss::IndexScalarQuantizer>(
            local_hnsw_index->storage->d, refine_sq_type.value(), local_hnsw_index->storage->metric_type);

        auto refine_index = std::make_unique<faiss::IndexRefine>(local_hnsw_index.get(), sq_refine.get());

        // let refine_index to own everything
        refine_index->own_refine_index = true;
        refine_index->own_fields = true;
        local_hnsw_index.release();
        sq_refine.release();

        // reassign
        return refine_index;
    }
}

}  // namespace

//
class BaseFaissRegularIndexHNSWSQNode : public BaseFaissRegularIndexHNSWNode {
 public:
    BaseFaissRegularIndexHNSWSQNode(const int32_t& version, const Object& object, DataFormatEnum data_format)
        : BaseFaissRegularIndexHNSWNode(version, object, data_format) {
    }
    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswSqConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FAISS_HNSW_SQ;
    }

 protected:
    Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) override {
        // number of rows
        auto rows = dataset->GetRows();
        // dimensionality of the data
        auto dim = dataset->GetDim();

        // config
        auto hnsw_cfg = static_cast<const FaissHnswSqConfig&>(cfg);

        auto metric = Str2FaissMetricType(hnsw_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }

        // parse a ScalarQuantizer type
        auto sq_type = get_sq_quantizer_type(hnsw_cfg.sq_type.value());
        if (!sq_type.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid scalar quantizer type: " << hnsw_cfg.sq_type.value();
            return Status::invalid_args;
        }

        // create an index
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE);

        std::unique_ptr<faiss::IndexHNSW> hnsw_index;
        if (is_cosine) {
            hnsw_index = std::make_unique<faiss::IndexHNSWSQCosine>(dim, sq_type.value(), hnsw_cfg.M.value());
        } else {
            hnsw_index = std::make_unique<faiss::IndexHNSWSQ>(dim, sq_type.value(), hnsw_cfg.M.value(), metric.value());
        }

        hnsw_index->hnsw.efConstruction = hnsw_cfg.efConstruction.value();

        // should refine be used?
        std::unique_ptr<faiss::Index> final_index;
        if (hnsw_cfg.refine.value_or(false) && hnsw_cfg.refine_type.has_value()) {
            // yes
            auto final_index_cnd = pick_refine_index(data_format, hnsw_cfg.refine_type, std::move(hnsw_index));
            if (!final_index_cnd.has_value()) {
                return Status::invalid_args;
            }

            // assign
            final_index = std::move(final_index_cnd.value());
        } else {
            // no refine

            // assign
            final_index = std::move(hnsw_index);
        }

        // we have to convert the data to float, unfortunately, which costs extra RAM
        auto float_ds_ptr = convert_ds_to_float(dataset, data_format);
        if (float_ds_ptr == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Unsupported data format";
            return Status::invalid_args;
        }

        // train
        LOG_KNOWHERE_INFO_ << "Training HNSW Index";

        final_index->train(rows, reinterpret_cast<const float*>(float_ds_ptr->GetTensor()));

        // done
        index = std::move(final_index);

        return Status::success;
    }
};

template <typename DataType>
class BaseFaissRegularIndexHNSWSQNodeTemplate : public BaseFaissRegularIndexHNSWSQNode {
 public:
    BaseFaissRegularIndexHNSWSQNodeTemplate(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWSQNode(version, object, datatype_v<DataType>) {
    }
};

// this index trains PQ and HNSW+FLAT separately, then constructs HNSW+PQ
class BaseFaissRegularIndexHNSWPQNode : public BaseFaissRegularIndexHNSWNode {
 public:
    BaseFaissRegularIndexHNSWPQNode(const int32_t& version, const Object& object, DataFormatEnum data_format)
        : BaseFaissRegularIndexHNSWNode(version, object, data_format) {
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswPqConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FAISS_HNSW_PQ;
    }

 protected:
    std::unique_ptr<faiss::IndexPQ> tmp_index_pq;

    Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) override {
        // number of rows
        auto rows = dataset->GetRows();
        // dimensionality of the data
        auto dim = dataset->GetDim();

        // config
        auto hnsw_cfg = static_cast<const FaissHnswPqConfig&>(cfg);

        auto metric = Str2FaissMetricType(hnsw_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }

        // create an index
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE);

        // HNSW + PQ index yields BAD recall somewhy.
        // Let's build HNSW+FLAT index, then replace FLAT with PQ

        std::unique_ptr<faiss::IndexHNSW> hnsw_index;
        if (is_cosine) {
            hnsw_index = std::make_unique<faiss::IndexHNSWFlatCosine>(dim, hnsw_cfg.M.value());
        } else {
            hnsw_index = std::make_unique<faiss::IndexHNSWFlat>(dim, hnsw_cfg.M.value(), metric.value());
        }

        hnsw_index->hnsw.efConstruction = hnsw_cfg.efConstruction.value();

        // pq
        std::unique_ptr<faiss::IndexPQ> pq_index;
        if (is_cosine) {
            pq_index = std::make_unique<faiss::IndexPQCosine>(dim, hnsw_cfg.m.value(), hnsw_cfg.nbits.value());
        } else {
            pq_index =
                std::make_unique<faiss::IndexPQ>(dim, hnsw_cfg.m.value(), hnsw_cfg.nbits.value(), metric.value());
        }

        // should refine be used?
        std::unique_ptr<faiss::Index> final_index;
        if (hnsw_cfg.refine.value_or(false) && hnsw_cfg.refine_type.has_value()) {
            // yes
            auto final_index_cnd = pick_refine_index(data_format, hnsw_cfg.refine_type, std::move(hnsw_index));
            if (!final_index_cnd.has_value()) {
                return Status::invalid_args;
            }

            // assign
            final_index = std::move(final_index_cnd.value());
        } else {
            // no refine

            // assign
            final_index = std::move(hnsw_index);
        }

        // we have to convert the data to float, unfortunately, which costs extra RAM
        auto float_ds_ptr = convert_ds_to_float(dataset, data_format);
        if (float_ds_ptr == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Unsupported data format";
            return Status::invalid_args;
        }

        // train hnswflat
        LOG_KNOWHERE_INFO_ << "Training HNSW Index";

        final_index->train(rows, reinterpret_cast<const float*>(float_ds_ptr->GetTensor()));

        // train pq
        LOG_KNOWHERE_INFO_ << "Training PQ Index";

        pq_index->train(rows, reinterpret_cast<const float*>(float_ds_ptr->GetTensor()));
        pq_index->pq.compute_sdc_table();

        // done
        index = std::move(final_index);
        tmp_index_pq = std::move(pq_index);

        return Status::success;
    }

    Status
    AddInternal(const DataSetPtr dataset, const Config&) override {
        if (this->index == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to an empty index.";
            return Status::empty_index;
        }

        auto rows = dataset->GetRows();
        try {
            // hnsw
            LOG_KNOWHERE_INFO_ << "Adding " << rows << " to HNSW Index";

            auto status_reg = add_to_index(index.get(), dataset, data_format);
            if (status_reg != Status::success) {
                return status_reg;
            }

            // pq
            LOG_KNOWHERE_INFO_ << "Adding " << rows << " to PQ Index";

            auto status_pq = add_to_index(tmp_index_pq.get(), dataset, data_format);
            if (status_pq != Status::success) {
                return status_pq;
            }

            // we're done.
            // throw away flat and replace it with pq

            // check if we have a refine available.
            faiss::IndexHNSW* index_hnsw = nullptr;

            faiss::IndexRefine* const index_refine = dynamic_cast<faiss::IndexRefine*>(index.get());

            if (index_refine != nullptr) {
                index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index_refine->base_index);
            } else {
                index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index.get());
            }

            // recreate hnswpq
            std::unique_ptr<faiss::IndexHNSW> index_hnsw_pq;

            if (index_hnsw->storage->is_cosine) {
                index_hnsw_pq = std::make_unique<faiss::IndexHNSWPQCosine>();
            } else {
                index_hnsw_pq = std::make_unique<faiss::IndexHNSWPQ>();
            }

            // C++ slicing.
            // we can't use move, because faiss::IndexHNSW overrides a destructor.
            static_cast<faiss::IndexHNSW&>(*index_hnsw_pq) = static_cast<faiss::IndexHNSW&>(*index_hnsw);

            // clear out the storage
            delete index_hnsw->storage;
            index_hnsw->storage = nullptr;
            index_hnsw_pq->storage = nullptr;

            // replace storage
            index_hnsw_pq->storage = tmp_index_pq.release();

            // replace if refine
            if (index_refine != nullptr) {
                delete index_refine->base_index;
                index_refine->base_index = index_hnsw_pq.release();
            } else {
                index = std::move(index_hnsw_pq);
            }

        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }
};

template <typename DataType>
class BaseFaissRegularIndexHNSWPQNodeTemplate : public BaseFaissRegularIndexHNSWPQNode {
 public:
    BaseFaissRegularIndexHNSWPQNodeTemplate(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWPQNode(version, object, datatype_v<DataType>) {
    }
};

// this index trains PRQ and HNSW+FLAT separately, then constructs HNSW+PRQ
class BaseFaissRegularIndexHNSWPRQNode : public BaseFaissRegularIndexHNSWNode {
 public:
    BaseFaissRegularIndexHNSWPRQNode(const int32_t& version, const Object& object, DataFormatEnum data_format)
        : BaseFaissRegularIndexHNSWNode(version, object, data_format) {
    }

    static std::unique_ptr<BaseConfig>
    StaticCreateConfig() {
        return std::make_unique<FaissHnswPrqConfig>();
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return StaticCreateConfig();
    }

    std::string
    Type() const override {
        return knowhere::IndexEnum::INDEX_FAISS_HNSW_PRQ;
    }

 protected:
    std::unique_ptr<faiss::IndexProductResidualQuantizer> tmp_index_prq;

    Status
    TrainInternal(const DataSetPtr dataset, const Config& cfg) override {
        // number of rows
        auto rows = dataset->GetRows();
        // dimensionality of the data
        auto dim = dataset->GetDim();

        // config
        auto hnsw_cfg = static_cast<const FaissHnswPrqConfig&>(cfg);

        auto metric = Str2FaissMetricType(hnsw_cfg.metric_type.value());
        if (!metric.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Invalid metric type: " << hnsw_cfg.metric_type.value();
            return Status::invalid_metric_type;
        }

        // create an index
        const bool is_cosine = IsMetricType(hnsw_cfg.metric_type.value(), metric::COSINE);

        // HNSW + PRQ index yields BAD recall somewhy.
        // Let's build HNSW+FLAT index, then replace FLAT with PRQ

        std::unique_ptr<faiss::IndexHNSW> hnsw_index;
        if (is_cosine) {
            hnsw_index = std::make_unique<faiss::IndexHNSWFlatCosine>(dim, hnsw_cfg.M.value());
        } else {
            hnsw_index = std::make_unique<faiss::IndexHNSWFlat>(dim, hnsw_cfg.M.value(), metric.value());
        }

        hnsw_index->hnsw.efConstruction = hnsw_cfg.efConstruction.value();

        // prq
        faiss::AdditiveQuantizer::Search_type_t prq_search_type =
            (metric.value() == faiss::MetricType::METRIC_INNER_PRODUCT)
                ? faiss::AdditiveQuantizer::Search_type_t::ST_LUT_nonorm
                : faiss::AdditiveQuantizer::Search_type_t::ST_norm_float;

        std::unique_ptr<faiss::IndexProductResidualQuantizer> prq_index;
        if (is_cosine) {
            prq_index = std::make_unique<faiss::IndexProductResidualQuantizerCosine>(
                dim, hnsw_cfg.m.value(), hnsw_cfg.nrq.value(), hnsw_cfg.nbits.value(), prq_search_type);
        } else {
            prq_index = std::make_unique<faiss::IndexProductResidualQuantizer>(
                dim, hnsw_cfg.m.value(), hnsw_cfg.nrq.value(), hnsw_cfg.nbits.value(), metric.value(), prq_search_type);
        }

        // should refine be used?
        std::unique_ptr<faiss::Index> final_index;
        if (hnsw_cfg.refine.value_or(false) && hnsw_cfg.refine_type.has_value()) {
            // yes
            auto final_index_cnd = pick_refine_index(data_format, hnsw_cfg.refine_type, std::move(hnsw_index));
            if (!final_index_cnd.has_value()) {
                return Status::invalid_args;
            }

            // assign
            final_index = std::move(final_index_cnd.value());
        } else {
            // no refine

            // assign
            final_index = std::move(hnsw_index);
        }

        // we have to convert the data to float, unfortunately, which costs extra RAM
        auto float_ds_ptr = convert_ds_to_float(dataset, data_format);
        if (float_ds_ptr == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Unsupported data format";
            return Status::invalid_args;
        }

        // train hnswflat
        LOG_KNOWHERE_INFO_ << "Training HNSW Index";

        final_index->train(rows, reinterpret_cast<const float*>(float_ds_ptr->GetTensor()));

        // train prq
        LOG_KNOWHERE_INFO_ << "Training ProductResidualQuantizer Index";

        prq_index->train(rows, reinterpret_cast<const float*>(float_ds_ptr->GetTensor()));

        // done
        index = std::move(final_index);
        tmp_index_prq = std::move(prq_index);

        return Status::success;
    }

    Status
    AddInternal(const DataSetPtr dataset, const Config&) override {
        if (this->index == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Can not add data to an empty index.";
            return Status::empty_index;
        }

        auto rows = dataset->GetRows();
        try {
            // hnsw
            LOG_KNOWHERE_INFO_ << "Adding " << rows << " to HNSW Index";

            auto status_reg = add_to_index(index.get(), dataset, data_format);
            if (status_reg != Status::success) {
                return status_reg;
            }

            // prq
            LOG_KNOWHERE_INFO_ << "Adding " << rows << " to ProductResidualQuantizer Index";

            auto status_prq = add_to_index(tmp_index_prq.get(), dataset, data_format);
            if (status_prq != Status::success) {
                return status_prq;
            }

            // we're done.
            // throw away flat and replace it with prq

            // check if we have a refine available.
            faiss::IndexHNSW* index_hnsw = nullptr;

            faiss::IndexRefine* const index_refine = dynamic_cast<faiss::IndexRefine*>(index.get());

            if (index_refine != nullptr) {
                index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index_refine->base_index);
            } else {
                index_hnsw = dynamic_cast<faiss::IndexHNSW*>(index.get());
            }

            // recreate hnswprq
            std::unique_ptr<faiss::IndexHNSW> index_hnsw_prq;

            if (index_hnsw->storage->is_cosine) {
                index_hnsw_prq = std::make_unique<faiss::IndexHNSWProductResidualQuantizerCosine>();
            } else {
                index_hnsw_prq = std::make_unique<faiss::IndexHNSWProductResidualQuantizer>();
            }

            // C++ slicing
            // we can't use move, because faiss::IndexHNSW overrides a destructor.
            static_cast<faiss::IndexHNSW&>(*index_hnsw_prq) = static_cast<faiss::IndexHNSW&>(*index_hnsw);

            // clear out the storage
            delete index_hnsw->storage;
            index_hnsw->storage = nullptr;
            index_hnsw_prq->storage = nullptr;

            // replace storage
            index_hnsw_prq->storage = tmp_index_prq.release();

            // replace if refine
            if (index_refine != nullptr) {
                delete index_refine->base_index;
                index_refine->base_index = index_hnsw_prq.release();
            } else {
                index = std::move(index_hnsw_prq);
            }

        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "faiss inner error: " << e.what();
            return Status::faiss_inner_error;
        }

        return Status::success;
    }
};

template <typename DataType>
class BaseFaissRegularIndexHNSWPRQNodeTemplate : public BaseFaissRegularIndexHNSWPRQNode {
 public:
    BaseFaissRegularIndexHNSWPRQNodeTemplate(const int32_t& version, const Object& object)
        : BaseFaissRegularIndexHNSWPRQNode(version, object, datatype_v<DataType>) {
    }
};

//
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_FLAT, BaseFaissRegularIndexHNSWFlatNodeTemplate, fp32);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_FLAT, BaseFaissRegularIndexHNSWFlatNodeTemplate, fp16);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_FLAT, BaseFaissRegularIndexHNSWFlatNodeTemplate, bf16);

KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_SQ, BaseFaissRegularIndexHNSWSQNodeTemplate, fp32);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_SQ, BaseFaissRegularIndexHNSWSQNodeTemplate, fp16);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_SQ, BaseFaissRegularIndexHNSWSQNodeTemplate, bf16);

KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_PQ, BaseFaissRegularIndexHNSWPQNodeTemplate, fp32);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_PQ, BaseFaissRegularIndexHNSWPQNodeTemplate, fp16);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_PQ, BaseFaissRegularIndexHNSWPQNodeTemplate, bf16);

KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_PRQ, BaseFaissRegularIndexHNSWPRQNodeTemplate, fp32);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_PRQ, BaseFaissRegularIndexHNSWPRQNodeTemplate, fp16);
KNOWHERE_SIMPLE_REGISTER_GLOBAL(FAISS_HNSW_PRQ, BaseFaissRegularIndexHNSWPRQNodeTemplate, bf16);

}  // namespace knowhere
