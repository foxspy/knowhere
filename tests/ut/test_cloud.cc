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

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "index/diskann/diskann_config.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/comp/local_file_manager.h"
#include "knowhere/factory.h"
#include "utils.h"
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
error "Missing the <filesystem> header."
#endif
#include <fstream>
std::string dataset = "sift";
std::string kDir = "/home/fox/data/" + dataset + "/";
std::string kRawDataPath;
std::string kL2IndexDir = kDir + "l2_index/";
std::string kIPIndexDir = kDir + "ip_index/";
std::string kL2IndexPrefix = kL2IndexDir + "l2";
std::string kIPIndexPrefix = kIPIndexDir + "ip";

#define METRIC_RECALL
#define CLEAR_ONCE

class TimeProfile {
public:
    TimeProfile(std::string name) {
        entry_name_ = name;
        enter_time_ = std::chrono::steady_clock::now();
        LOG_KNOWHERE_INFO_ << "-------------------------------------------------------------Start Processing " << name;
    }

    long long
    nanotime() {
        auto now = std::chrono::steady_clock::now();
        long long all_cost = std::chrono::duration_cast<std::chrono::nanoseconds>(now - enter_time_).count();
        return all_cost;
    }

    std::string
    cost() {
        auto now = std::chrono::steady_clock::now();
        long long all_cost = std::chrono::duration_cast<std::chrono::nanoseconds>(now - enter_time_).count();
        return "-------------------------------------------------------------Processing " + entry_name_ +
               " done, cost " + std::to_string(all_cost / 1000000.0) + "[ms]";
    }

    std::string
    cost(long long time) {
        return "-------------------------------------------------------------Processing " + entry_name_ +
               " done, cost " + std::to_string(time / 1000000.0) + "[ms]";
    }

private:
    std::string entry_name_;
    std::chrono::steady_clock::time_point enter_time_;
};

std::unique_ptr<float[]>
get_bin_data(const std::string& bin_file, int32_t& nrows, int32_t& ncols) {
    std::ifstream reader(bin_file.c_str(), std::ios::binary);
    int nrows_32, ncols_32;
    reader.read((char*)&nrows_32, sizeof(int));
    reader.read((char*)&ncols_32, sizeof(int));
    nrows = nrows_32;
    ncols = ncols_32;

    auto tensor = std::make_unique<float[]>(nrows * ncols);
    reader.read((char*)tensor.get(), nrows * ncols * sizeof(float));
    return tensor;
}

TEST_CASE("Test CLOUD Performance", "[CLOUD]") {
    int32_t row, dim;
    auto metric = GENERATE(as<std::string>{}, knowhere::metric::IP);
    auto version = 1;

    auto search_list = {30, 60,90,120,150,180,210,240,270,300};

    auto base_datapath = kDir + dataset + ".fbin";
    kRawDataPath = base_datapath;

    if (!std::filesystem::exists(kRawDataPath)) {
        return;
    }
    REQUIRE_NOTHROW(fs::create_directory(kDir));

#ifdef CLEAR_ONCE
    REQUIRE_NOTHROW(fs::remove_all(metric == knowhere::metric::L2 ? kL2IndexDir : kIPIndexDir));
#endif

    REQUIRE_NOTHROW(fs::create_directory(metric == knowhere::metric::L2 ? kL2IndexDir : kIPIndexDir));

    auto base_tensor = get_bin_data(base_datapath, row, dim);
    auto base_dataset = knowhere::GenDataSet(row, dim, base_tensor.get());

    auto query_datapath = kDir + dataset + "_query.fbin";
    int32_t query_row, query_dim;
    auto query_tensor = get_bin_data(query_datapath, query_row, query_dim);
    auto query_dataset = knowhere::GenDataSet(query_row, query_dim, query_tensor.get());

    auto base_gen = [&row, &dim, &metric, &version] {
        knowhere::Json json;
        json["dim"] = dim;
        json["metric_type"] = metric;
        json["k"] = 100;
        if (metric == knowhere::metric::L2) {
            json["radius"] = CFG_FLOAT::value_type(200000);
            json["range_filter"] = CFG_FLOAT::value_type(0);
        } else {
            json["radius"] = CFG_FLOAT::value_type(50000);
            json["range_filter"] = std::numeric_limits<CFG_FLOAT::value_type>::max();
        }
        return json;
    };

    auto ivf_build_config = [&base_gen, &row, &dim, &metric]() {
        knowhere::Json json = base_gen();

        json["nlist"] = 1024;

        return json;
    };

    auto ivf_search_config = [&base_gen, &row, &dim, &metric]() {
        knowhere::Json json = base_gen();

        json["nprobe"] = 128;

        return json;
    };

    std::shared_ptr<knowhere::FileManager> file_manager = std::make_shared<knowhere::LocalFileManager>();
    auto index_pack = knowhere::Pack(file_manager);

#ifdef METRIC_RECALL
    TimeProfile brute_search("BruteForce.Search");
    auto result_knn = knowhere::BruteForce::Search(base_dataset, query_dataset, base_gen(), nullptr);
    LOG_KNOWHERE_INFO_ << brute_search.cost();
#endif
    // build process
    SECTION("Test Cloud Performance") {
        using std::make_tuple;
        auto disk_dataset = std::make_shared<knowhere::DataSet>();

        auto index_list = std::vector<std::string>{knowhere::IndexEnum::INDEX_FAISS_IVFFLAT};

        for (auto& index_type : index_list) {
            auto index = knowhere::IndexFactory::Instance().Create(index_type, version, index_pack);
            std::string index_prefix = (metric == knowhere::metric::L2 ? kL2IndexPrefix : kIPIndexPrefix);
            std::string bin_file = index_prefix + "_ivf.index.bin";
            auto build_config = ivf_build_config();
            if (!std::filesystem::exists(bin_file)) {
                TimeProfile index_build(index_type + " building ");
                auto status = index.Build(*base_dataset, build_config);
                knowhere::BinarySet binarySet;
                index.Serialize(binarySet);
                if (binarySet.Contains(index_type)) {
                    std::fstream file;
                    file.open(bin_file, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
                    const char* data = reinterpret_cast<const char*>(binarySet.GetByName(index_type)->data.get());
                    int64_t data_size = binarySet.GetByName(index_type)->size;
                    file.write(data, data_size);
                    file.flush();
                    file.close();
                }
                REQUIRE(status == knowhere::Status::success);
                LOG_KNOWHERE_INFO_ << index_build.cost();
            }

            auto search_index = knowhere::IndexFactory::Instance().Create(index_type, version, index_pack);
            {
                TimeProfile index_load(index_type + " loading ");
                std::fstream index_file;
                index_file.open(bin_file, std::ios::binary | std::ios::in);
                index_file.seekg(0, std::ios::end);
                auto index_size = index_file.tellg();
                std::shared_ptr<std::uint8_t[]> buffer;
                std::uint8_t* ptr = new std::uint8_t[index_size];
                buffer.reset(ptr);
                index_file.seekg(0, std::ios::beg);
                index_file.read(reinterpret_cast<char*>(buffer.get()), index_size);
                knowhere::BinarySet binaryIndex;
                binaryIndex.Append(index_type, buffer, index_size);
                search_index.Deserialize(binaryIndex);
                LOG_KNOWHERE_INFO_ << index_load.cost();
            }

            auto search_config = ivf_search_config();
            for (int i = 0; i < 3; i++) {
                for (auto& search_size : search_list) {
                    TimeProfile ivf_search(index_type + " knn search " + std::to_string(search_size));
                    search_config["nprobe"] = std::to_string(search_size);
                    auto knn_res = search_index.Search(*query_dataset, search_config, nullptr);
                    auto time_consume = ivf_search.nanotime();
                    REQUIRE(knn_res.has_value());
                    LOG_KNOWHERE_INFO_ << ivf_search.cost(time_consume);
#ifdef METRIC_RECALL
                    auto knn_recall = GetKNNRecall(*result_knn.value(), *knn_res.value());
                    LOG_KNOWHERE_INFO_ << ivf_search.cost(time_consume) << ", recall = " << knn_recall;
#endif
                }
            }
        }
    }
}
