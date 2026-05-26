// Copyright (C) 2019-2023 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.

#ifdef KNOWHERE_WITH_SVS

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "knowhere/comp/task.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/index/index_factory.h"
#include "knowhere/log.h"
#include "utils.h"

namespace {

struct FBinData {
    int64_t rows = 0;
    int64_t dim = 0;
    std::vector<float> data;
};

int64_t
GetEnvInt64(const char* name, int64_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    return std::stoll(value);
}

float
GetEnvFloat(const char* name, float default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    return std::stof(value);
}

bool
GetEnvBool(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    const std::string normalized(value);
    return normalized == "1" || normalized == "true" || normalized == "TRUE" || normalized == "yes" ||
           normalized == "YES";
}

std::string
GetEnvString(const char* name, const std::string& default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return default_value;
    }
    return value;
}

FBinData
ReadFBin(const std::filesystem::path& path, int64_t max_rows) {
    std::ifstream reader(path, std::ios::binary);
    REQUIRE(reader.good());

    int32_t rows = 0;
    int32_t dim = 0;
    reader.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    reader.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    REQUIRE(rows > 0);
    REQUIRE(dim > 0);

    const int64_t rows_to_read = max_rows > 0 ? std::min<int64_t>(rows, max_rows) : rows;
    FBinData result;
    result.rows = rows_to_read;
    result.dim = dim;
    result.data.resize(rows_to_read * dim);
    reader.read(reinterpret_cast<char*>(result.data.data()), result.data.size() * sizeof(float));
    REQUIRE(reader.good());
    return result;
}

knowhere::DataSetPtr
ReadTruth(const std::filesystem::path& path, int64_t topk, int64_t max_queries) {
    std::ifstream reader(path, std::ios::binary);
    REQUIRE(reader.good());

    int32_t nq = 0;
    int32_t truth_topk = 0;
    reader.read(reinterpret_cast<char*>(&nq), sizeof(nq));
    reader.read(reinterpret_cast<char*>(&truth_topk), sizeof(truth_topk));
    REQUIRE(nq > 0);
    REQUIRE(truth_topk >= topk);

    const int64_t nq_to_read = max_queries > 0 ? std::min<int64_t>(nq, max_queries) : nq;
    auto ids = std::make_unique<int64_t[]>(nq_to_read * topk);
    auto distances = std::make_unique<float[]>(nq_to_read * topk);

    std::vector<int64_t> truth_ids(truth_topk);
    for (int64_t i = 0; i < nq_to_read; ++i) {
        reader.read(reinterpret_cast<char*>(truth_ids.data()), truth_ids.size() * sizeof(int64_t));
        REQUIRE(reader.good());
        std::copy_n(truth_ids.data(), topk, ids.get() + i * topk);
    }

    const auto distances_offset = static_cast<std::streamoff>(sizeof(int32_t) * 2 + sizeof(int64_t) * nq * truth_topk);
    reader.clear();
    reader.seekg(distances_offset, std::ios::beg);
    if (reader.good()) {
        std::vector<float> truth_distances(truth_topk);
        for (int64_t i = 0; i < nq_to_read; ++i) {
            reader.read(reinterpret_cast<char*>(truth_distances.data()), truth_distances.size() * sizeof(float));
            if (!reader.good()) {
                std::fill(distances.get(), distances.get() + nq_to_read * topk, 0.0f);
                break;
            }
            std::copy_n(truth_distances.data(), topk, distances.get() + i * topk);
        }
    } else {
        std::fill(distances.get(), distances.get() + nq_to_read * topk, 0.0f);
    }

    return knowhere::GenResultDataSet(nq_to_read, topk, std::move(ids), std::move(distances));
}

void
SetIntParam(knowhere::Json& conf, const char* key, int64_t value, bool as_string) {
    if (as_string) {
        conf[key] = std::to_string(value);
    } else {
        conf[key] = value;
    }
}

}  // namespace

TEST_CASE("Reproduce OpenAI500K SVS Vamana LeanVec recall", "[svs][leanvec][reproduce][.]") {
    const auto data_dir_env = std::getenv("KNOWHERE_SVS_REPRO_DATA_DIR");
    if (data_dir_env == nullptr || *data_dir_env == '\0') {
        WARN(
            "Set KNOWHERE_SVS_REPRO_DATA_DIR to the prepared OpenAI500K fbin directory. "
            "See docs/svs_reproduce_openai500k.md.");
        return;
    }

    const std::filesystem::path data_dir(data_dir_env);
    const auto base_path = data_dir / GetEnvString("KNOWHERE_SVS_REPRO_BASE_FILE", "openai.fbin");
    const auto query_path = data_dir / GetEnvString("KNOWHERE_SVS_REPRO_QUERY_FILE", "openai_query.fbin");
    const auto truth_path =
        data_dir / GetEnvString("KNOWHERE_SVS_REPRO_TRUTH_FILE", "openai_query_COSINE_0.99_100.truth");

    const int64_t topk = GetEnvInt64("KNOWHERE_SVS_REPRO_TOPK", 10);
    const int64_t max_base_rows = GetEnvInt64("KNOWHERE_SVS_REPRO_MAX_BASE_ROWS", 0);
    const int64_t max_queries = GetEnvInt64("KNOWHERE_SVS_REPRO_MAX_QUERIES", 0);
    const float min_recall = GetEnvFloat("KNOWHERE_SVS_REPRO_MIN_RECALL", 0.95f);
    const int32_t index_version = static_cast<int32_t>(GetEnvInt64("KNOWHERE_SVS_REPRO_INDEX_VERSION", 9));
    const bool string_params = GetEnvBool("KNOWHERE_SVS_REPRO_STRING_PARAMS", true);
    const bool roundtrip = GetEnvBool("KNOWHERE_SVS_REPRO_ROUNDTRIP", true);
    const bool include_data_path = GetEnvBool("KNOWHERE_SVS_REPRO_INCLUDE_DATA_PATH", true);
    const bool exec_over_build_pool = GetEnvBool("KNOWHERE_SVS_REPRO_EXEC_OVER_BUILD_POOL", true);
    const bool use_zero_bitset = GetEnvBool("KNOWHERE_SVS_REPRO_ZERO_BITSET", false);
    const int64_t search_window_size = GetEnvInt64("KNOWHERE_SVS_REPRO_SEARCH_WINDOW_SIZE", 1000);
    const int64_t search_buffer_capacity = GetEnvInt64("KNOWHERE_SVS_REPRO_SEARCH_BUFFER_CAPACITY", 1000);
    const std::string load_index_file = GetEnvString("KNOWHERE_SVS_REPRO_LOAD_INDEX_FILE", "");

    const auto load_start = std::chrono::steady_clock::now();
    auto base = ReadFBin(base_path, max_base_rows);
    auto query = ReadFBin(query_path, max_queries);
    auto truth = ReadTruth(truth_path, topk, query.rows);
    const auto load_end = std::chrono::steady_clock::now();

    REQUIRE(base.dim == 1536);
    REQUIRE(query.dim == base.dim);
    REQUIRE(truth->GetRows() == query.rows);

    auto train_ds = knowhere::GenDataSet(base.rows, base.dim, base.data.data());
    auto query_ds = knowhere::GenDataSet(query.rows, query.dim, query.data.data());

    knowhere::Json build_conf;
    SetIntParam(build_conf, knowhere::meta::DIM, base.dim, string_params);
    build_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
    SetIntParam(build_conf, knowhere::indexparam::SVS_GRAPH_MAX_DEGREE, 64, string_params);
    SetIntParam(build_conf, knowhere::indexparam::SVS_CONSTRUCTION_WINDOW_SIZE, 200, string_params);
    build_conf[knowhere::indexparam::SVS_STORAGE_KIND] = std::string("leanvec4x8");
    SetIntParam(build_conf, knowhere::indexparam::SVS_LEANVEC_DIM, 768, string_params);
    if (include_data_path) {
        build_conf["data_path"] = base_path.string();
        build_conf["vec_field_size_gb"] =
            static_cast<float>(base.data.size() * sizeof(float)) / 1024.0f / 1024.0f / 1024.0f;
    }

    knowhere::Json search_conf;
    search_conf[knowhere::meta::METRIC_TYPE] = knowhere::metric::COSINE;
    SetIntParam(search_conf, knowhere::meta::TOPK, topk, string_params);
    SetIntParam(search_conf, knowhere::indexparam::SVS_SEARCH_WINDOW_SIZE, search_window_size, string_params);
    SetIntParam(search_conf, knowhere::indexparam::SVS_SEARCH_BUFFER_CAPACITY, search_buffer_capacity, string_params);

    auto idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC,
                                                                         index_version);
    REQUIRE(idx.has_value());
    auto index = idx.value();

    const auto build_start = std::chrono::steady_clock::now();
    if (!load_index_file.empty()) {
        std::ifstream reader(load_index_file, std::ios::binary);
        REQUIRE(reader.good());
        reader.seekg(0, std::ios::end);
        const auto index_size = static_cast<int64_t>(reader.tellg());
        reader.seekg(0, std::ios::beg);
        auto data = std::shared_ptr<uint8_t[]>(new uint8_t[index_size]);
        reader.read(reinterpret_cast<char*>(data.get()), index_size);
        REQUIRE(reader.good());

        knowhere::BinarySet binary_set;
        binary_set.Append(index.Type(), data, index_size);
        knowhere::Json load_conf = build_conf;
        load_conf["vec_field_size_gb"] = 0.125f;
        load_conf["search_cache_budget_gb"] = 1.0f;
        REQUIRE(index.Deserialize(binary_set, load_conf) == knowhere::Status::success);
    } else {
        REQUIRE(index.Build(train_ds, build_conf) == knowhere::Status::success);
    }
    const auto build_end = std::chrono::steady_clock::now();

    double roundtrip_seconds = 0.0;
    if (roundtrip && load_index_file.empty()) {
        const auto roundtrip_start = std::chrono::steady_clock::now();
        knowhere::BinarySet binary_set;
        REQUIRE(index.Serialize(binary_set) == knowhere::Status::success);

        auto loaded_idx = knowhere::IndexFactory::Instance().Create<knowhere::fp32>(
            knowhere::IndexEnum::INDEX_SVS_VAMANA_LEANVEC, index_version);
        REQUIRE(loaded_idx.has_value());
        index = loaded_idx.value();

        knowhere::Json load_conf = build_conf;
        load_conf["vec_field_size_gb"] = 0.125f;
        load_conf["search_cache_budget_gb"] = 1.0f;
        REQUIRE(index.Deserialize(binary_set, load_conf) == knowhere::Status::success);
        const auto roundtrip_end = std::chrono::steady_clock::now();
        roundtrip_seconds = std::chrono::duration<double>(roundtrip_end - roundtrip_start).count();
    }

    knowhere::expected<knowhere::DataSetPtr> result =
        knowhere::expected<knowhere::DataSetPtr>::Err(knowhere::Status::empty_index, "search did not run");
    std::vector<uint8_t> zero_bitset;
    knowhere::BitsetView bitset_view;
    if (use_zero_bitset) {
        zero_bitset.resize((base.rows + 7) / 8, 0);
        bitset_view = knowhere::BitsetView(zero_bitset.data(), base.rows, 0);
    }
    if (exec_over_build_pool) {
        std::vector<std::function<void()>> tasks;
        tasks.emplace_back([&]() { result = index.Search(query_ds, search_conf, bitset_view); });
        knowhere::ExecOverBuildThreadPool(tasks);
    } else {
        result = index.Search(query_ds, search_conf, bitset_view);
    }
    const auto search_end = std::chrono::steady_clock::now();
    REQUIRE(result.has_value());

    const float recall = GetKNNRecall(*truth, *result.value());
    const auto load_seconds = std::chrono::duration<double>(load_end - load_start).count();
    const auto build_seconds = std::chrono::duration<double>(build_end - build_start).count();
    const auto search_seconds = std::chrono::duration<double>(search_end - build_end).count();

    std::cout << "SVS OpenAI500K reproduce: rows=" << base.rows << ", dim=" << base.dim << ", nq=" << query.rows
              << ", topk=" << topk << ", recall=" << recall << ", load_s=" << load_seconds
              << ", build_s=" << build_seconds << ", roundtrip_s=" << roundtrip_seconds
              << ", search_s=" << search_seconds << ", string_params=" << string_params
              << ", roundtrip=" << roundtrip << ", include_data_path=" << include_data_path
              << ", exec_over_build_pool=" << exec_over_build_pool << ", zero_bitset=" << use_zero_bitset
              << ", search_window_size=" << search_window_size
              << ", search_buffer_capacity=" << search_buffer_capacity
              << ", load_index_file=" << load_index_file << std::endl;

    REQUIRE(recall >= min_recall);
}

#endif  // KNOWHERE_WITH_SVS
