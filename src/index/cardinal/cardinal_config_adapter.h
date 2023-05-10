// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#pragma once

#include <filesystem>

#include "index/diskann/diskann_config.h"
#include "index/hnsw/hnsw_config.h"
#include "index/vector_index/VecIndexConfig.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/dataset.h"
#include "knowhere/log.h"

namespace knowhere {
class CardinalIndexAdapter {
 public:
    expected<cardinal::VecIndexConfig, Status>
    convertCfg(const Config& cfg) {
        if (isHNSWConfig(cfg)) {
            return hnswConfig(static_cast<const HnswConfig&>(cfg));
        } else if (isDiskANNConfig(cfg)) {
            uint64_t local_row, local_dim;
            std::unique_ptr<float[]> tensor;
            auto build_conf = static_cast<const DiskANNConfig&>(cfg);
            get_bin_metadata(build_conf.data_path, local_row, local_dim, tensor, false);
            LOG_KNOWHERE_ERROR_ << " data_pth: " << build_conf.data_path
                                << " local_dim : " << std::to_string(local_dim);
            return diskANNConfig(static_cast<const DiskANNConfig&>(cfg), local_row, local_dim);
        }
        return unexpected(Status::invalid_args);
    }

    expected<cardinal::VecIndexConfig, Status>
    convertCfg(const Config& cfg, size_t row, size_t dim) {
        if (isHNSWConfig(cfg)) {
            return hnswConfig(static_cast<const HnswConfig&>(cfg));
        } else if (isDiskANNConfig(cfg)) {
            std::unique_ptr<float[]> tensor;
            auto build_conf = static_cast<const DiskANNConfig&>(cfg);
            return diskANNConfig(static_cast<const DiskANNConfig&>(cfg), row, dim);
        }
        return unexpected(Status::invalid_args);
    }

    expected<std::shared_ptr<cardinal::VecDataSet>, Status>
    convertDataSet(bool build, const DataSet& dataset, const Config& cfg) {
        if (build && isDiskANNConfig(cfg)) {
            auto build_conf = static_cast<const DiskANNConfig&>(cfg);
            uint64_t row, dim;
            std::unique_ptr<float[]> tensor;
            if (!std::filesystem::exists(build_conf.data_path)) {
                return unexpected(Status::diskann_file_error);
            }
            get_bin_metadata(build_conf.data_path, row, dim, tensor, true);

            cardinal::VecDataSetProperty property(row, dim, cardinal::Schema::floatVecsSchema(dim));
            auto vecDataSet = std::make_shared<cardinal::VecDataSet>(property);
            cardinal::VecTuple tuple(vecDataSet->schema(), false);
            for (int64_t i = 0; (unsigned)i < row; i++) {
                tuple.deserialize((uint8_t*)(tensor.get() + i * dim));
                vecDataSet->addTuple(tuple, i);
            }
            return vecDataSet;
        } else {
            uint64_t row, dim;
            row = dataset.GetRows();
            dim = dataset.GetDim();
            cardinal::VecDataSetProperty property(row, dim, cardinal::Schema::floatVecsSchema(dataset.GetDim()));
            auto vecDataSet = std::make_shared<cardinal::VecDataSet>(property);
            cardinal::VecTuple tuple(vecDataSet->schema(), false);

            for (int64_t i = 0; i < row; i++) {
                tuple.deserialize((uint8_t*)((float*)dataset.GetTensor() + i * dim));
                vecDataSet->addTuple(tuple, i);
            }
            return vecDataSet;
        }
    }

    expected<DataSetPtr, Status>
    convertResult(cardinal::KnnSearchResult& result) {
        uint64_t k = result.topK;
        uint64_t nq = result.query_size;
        auto p_id = new int64_t[k * nq];
        auto p_dist = new float[k * nq];

        for (uint64_t i = 0; i < result.query_size; i++) {
            for (uint64_t j = 0; j < result.sizes[i]; j++) {
                p_id[i * k + j] = result.labels[i * k + j];
                p_dist[i * k + j] = result.distances[i * k + j];
            }
            for (uint64_t j = result.sizes[i]; j < k; j++) {
                p_id[i * k + j] = -1;
                p_dist[i * k + j] = float(1.0 / 0.0);
            }
        }

        return GenResultDataSet(result.query_size, result.topK, p_id, p_dist);
    }

 private:
    inline void
    get_bin_metadata_impl(std::basic_istream<char>& reader, uint64_t& nrows, uint64_t& ncols,
                          std::unique_ptr<float[]>& tensor, bool needTensor) {
        int nrows_32, ncols_32;
        reader.read((char*)&nrows_32, sizeof(int));
        reader.read((char*)&ncols_32, sizeof(int));
        nrows = nrows_32;
        ncols = ncols_32;
        if (needTensor) {
            auto tmp = std::make_unique<float[]>(nrows * ncols);
            reader.read((char*)tmp.get(), nrows * ncols * sizeof(float));
            tmp.swap(tensor);
        }
    }

    inline void
    get_bin_metadata(const std::string& bin_file, uint64_t& nrows, uint64_t& ncols, std::unique_ptr<float[]>& tensor,
                     bool needTensor) {
        std::ifstream reader(bin_file.c_str(), std::ios::binary);
        get_bin_metadata_impl(reader, nrows, ncols, tensor, needTensor);
    }

    template <typename Base, typename T>
    inline bool instanceof (const T* t) {
        return dynamic_cast<const Base*>(t) != nullptr;
    }

    bool
    isHNSWConfig(const Config& cfg) {
        return instanceof <HnswConfig>(&cfg);
    }

    bool
    isDiskANNConfig(const Config& cfg) {
        return instanceof <DiskANNConfig>(&cfg);
    }

    expected<cardinal::MetricType, Status>
    getCardinalMetricType(std::string metric) {
        static const std::unordered_map<std::string, cardinal::MetricType> metric_map = {
            {metric::L2, cardinal::MetricType::L2}, {metric::IP, cardinal::MetricType::IP}};

        std::transform(metric.begin(), metric.end(), metric.begin(), toupper);
        auto it = metric_map.find(metric);
        if (it == metric_map.end())
            return unexpected(Status::invalid_metric_type);
        return it->second;
    }

    cardinal::VecIndexConfig
    hnswConfig(const HnswConfig& hnswConfig) {
        cardinal::VecIndexConfig config;

        config.topK = hnswConfig.k;
        config.metricType = getCardinalMetricType(hnswConfig.metric_type).value();

        config.lower_bound = hnswConfig.range_filter;
        config.upper_bound = hnswConfig.radius;
        config.l_min_k = 100;
        config.l_max_k = 10000;
        config.l_k_ratio = 2.0;

        config.builder_type = "hnsw";
        config.searcher_type = "hnsw";
        config.index_type = "HNSW";
        config.num_threads = 12;
        config.base_limit = hnswConfig.M * 2;
        config.edge_limit = hnswConfig.M;
        config.build_candidates_limit = hnswConfig.efConstruction;
        config.search_candidates_limit = hnswConfig.ef;

        return config;
    }

    expected<cardinal::VecIndexConfig, Status>
    diskANNConfig(const DiskANNConfig& diskConfig, size_t row, size_t dim) {
        cardinal::VecIndexConfig config;

        config.topK = diskConfig.k;
        auto metric = getCardinalMetricType(diskConfig.metric_type);
        if (metric.has_value()) {
            config.metricType = metric.value();
        } else {
            return unexpected(metric.error());
        }

        config.M = (size_t)(std::floor)((double)diskConfig.pq_code_budget_gb * 1024 * 1024 * 1024 / row);
        config.edge_limit = diskConfig.max_degree;
        config.edge_capacity = diskConfig.max_degree;
        config.build_candidates_limit = diskConfig.search_list_size;
        config.search_candidates_limit = diskConfig.search_list_size;
        config.beamwidth = diskConfig.beamwidth;
        config.num_rnds = 2;

        config.lower_bound = diskConfig.range_filter;
        config.upper_bound = diskConfig.radius;
        config.l_min_k = diskConfig.min_k;
        config.l_max_k = diskConfig.max_k;
        config.l_k_ratio = diskConfig.search_list_and_k_ratio;

        if (config.l_min_k > config.l_max_k || config.search_candidates_limit < config.topK) {
            return unexpected(Status::invalid_args);
        }

        config.maxc = 500;
        config.alpha = 1.0;
        config.alpha_factor = 1.2f;
        config.slack_factor = 1.3f;
        config.last_round_alpha = 1.2f;
        const uint64_t MAX_SAMPLE_POINTS_FOR_WARMUP = 100000;
        config.warm_point_num =
            std::ceil(row * 0.1) > MAX_SAMPLE_POINTS_FOR_WARMUP ? MAX_SAMPLE_POINTS_FOR_WARMUP : std::ceil(row * 0.1);

        uint32_t one_cached_node_budget = (config.edge_limit + 1) * sizeof(unsigned) + sizeof(float) * dim;
        float kCacheExpansionRate = 1.2;
        config.cache_point_num = (size_t)(std::floor)((double)diskConfig.search_cache_budget_gb * 1024 * 1024 * 1024 /
                                                      (one_cached_node_budget * kCacheExpansionRate));
        config.index_type = "DiskANN";
        config.builder_type = "vamana";
        config.searcher_type = "vamana";
        config.dataset_path = "vamana.index.bin";
        config.quantizerType = cardinal::QuantizerType::Original;
        config.inmem_quantizer_type = cardinal::QuantizerType::PQQuantizer;
        return config;
    }
};
}  // namespace knowhere
