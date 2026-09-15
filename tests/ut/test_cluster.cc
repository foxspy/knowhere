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

#include <sys/stat.h>

#include <cmath>
#include <unordered_set>

#include "catch2/catch_approx.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "faiss/Clustering.h"
#include "faiss/IndexFlat.h"
#include "faiss/cppcontrib/knowhere/utils/binary_distances.h"
#include "hnswlib/hnswalg.h"
#include "knowhere/bitsetview.h"
#include "knowhere/cluster/cluster_factory.h"
#include "knowhere/comp/brute_force.h"
#include "knowhere/dataset.h"
#include "knowhere/log.h"
#include "utils.h"

namespace {
constexpr float kKnnRecallThreshold = 0.1f;
constexpr float kKnnMinClusterSizeRatio = 0.1f;
constexpr float kKnnMaxClusterSizeRatio = 10.0f;
}  // namespace

// use kNN search to test the correctness of kmeans
TEST_CASE("Test Kmeans With Float Vector", "[float metrics]") {
    using Catch::Approx;

    size_t nb = 100000, nq = 100, nt, ns;
    const int64_t dim = 128;
    const int64_t num_clusters = 10;
    const std::string NUM_CLUSTERS = "num_clusters";
    const std::string NUM_ITER = "num_iter";
    auto topk = 1;
    auto maxTopK = 10;
    auto nprobes = 3;
    auto num_trains = 10;
    auto base_gen = [=]() {
        knowhere::Json json;
        json[NUM_CLUSTERS] = num_clusters;
        json[NUM_ITER] = 7;
        json[knowhere::meta::NUM_BUILD_THREAD] = 1;
        return json;
    };

    std::unique_ptr<float[]> sampled_data;
    auto base_ds = GenDataSet(nb, dim);
    auto query_ds = GenDataSet(nq, dim);

    const knowhere::Json conf = {
        {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
        {knowhere::meta::TOPK, maxTopK},
    };
    knowhere::BitsetView bitset;
    auto gt = knowhere::BruteForce::Search<knowhere::fp32>(base_ds, query_ds, conf, bitset);

    SECTION("Test Kmeans result") {
        using std::make_tuple;
        auto [name, gen] = GENERATE_REF(table<std::string, std::function<knowhere::Json()>>(
            {make_tuple(knowhere::ClusterEnum::CLUSTER_KMEANS, base_gen)}));
        auto cluster = knowhere::ClusterFactory::Instance().Create<knowhere::fp32>(name).value();
        auto cfg_json = gen().dump();
        CAPTURE(name, cfg_json);
        knowhere::Json json = knowhere::Json::parse(cfg_json);
        REQUIRE(cluster.Type() == name);

        for (int i = 0; i < num_trains; i++) {
            GenRandomSlice<float>((const float*)base_ds->GetTensor(), nb, dim, 0.01, sampled_data, ns);
            auto sample_ds = knowhere::GenDataSet(ns, dim, sampled_data.get());
            auto res = cluster.Train(*sample_ds, json);
            REQUIRE(res.has_value());
        }

        std::vector<std::vector<int64_t>> ids(num_clusters);
        size_t min_cluster_size = nb, max_cluster_size = 0, avg_cluster_size = 0;
        auto res_assign = cluster.Assign(*base_ds);
        REQUIRE(res_assign.has_value());
        for (int64_t i = 0; i < nb; ++i) {
            auto centroid_id = reinterpret_cast<const uint32_t*>(res_assign.value()->GetTensor())[i];
            REQUIRE(centroid_id < num_clusters);
            ids[centroid_id].push_back(i);
        }
        for (size_t i = 0; i < num_clusters; i++) {
            size_t cs = ids[i].size();
            if (cs < min_cluster_size) {
                min_cluster_size = cs;
            }
            if (cs > max_cluster_size) {
                max_cluster_size = cs;
            }
            avg_cluster_size += cs;
        }
        avg_cluster_size /= num_clusters;
        REQUIRE((float(min_cluster_size) / float(avg_cluster_size) > kKnnMinClusterSizeRatio));
        REQUIRE((float(max_cluster_size) / float(avg_cluster_size) < kKnnMaxClusterSizeRatio));
        faiss::IndexFlatL2 index(dim);
        res_assign = cluster.Assign(*query_ds);
        REQUIRE(res_assign.has_value());
        auto res_assign_with_distance = cluster.AssignWithDistance(*query_ds, json);
        REQUIRE(res_assign_with_distance.has_value());
        REQUIRE(res_assign_with_distance.value()->GetIds() != nullptr);
        REQUIRE(res_assign_with_distance.value()->GetDistance() != nullptr);

        // each query select its nearest cluster as the result
        // like ivfflat choose nprobe=1
        std::vector<std::vector<int64_t>> result_assign(nq);
        for (int64_t i = 0; i < nq; ++i) {
            auto centroid_id = reinterpret_cast<const uint32_t*>(res_assign.value()->GetTensor())[i];
            result_assign[i] = ids[centroid_id];
        }
        res_assign.value().reset();
        float recall_top1 = GetKNNRecall(*gt.value(), result_assign, 1);
        LOG_KNOWHERE_INFO_ << "recall_top1: " << recall_top1;
        float recall_top10 = GetKNNRecall(*gt.value(), result_assign, maxTopK);
        LOG_KNOWHERE_INFO_ << "recall_top10: " << recall_top10;

        result_assign.clear();
        auto res_centroids = cluster.GetCentroids();
        REQUIRE(res_centroids.has_value());
        const float* centroids = static_cast<const float*>(res_centroids.value()->GetTensor());
        const float* queries = static_cast<const float*>(query_ds->GetTensor());
        index.add(num_clusters, centroids);
        nq = 100;
        std::vector<std::vector<int64_t>> result(nq);
        for (int64_t i = 0; i < nq; ++i) {
            std::vector<float> dis(nprobes);
            std::vector<faiss::idx_t> idx(nprobes);
            index.search(1, &queries[i * dim], nprobes, dis.data(), idx.data());
            CHECK(res_assign_with_distance.value()->GetIds()[i] == idx[0]);
            CHECK(res_assign_with_distance.value()->GetDistance()[i] == Approx(dis[0]));
            for (int64_t j = 0; j < nprobes; j++) {
                auto centroid_id = idx[j];
                result[i].insert(result[i].end(), ids[centroid_id].begin(), ids[centroid_id].end());
            }
        }
        float recall = GetKNNRecall(*gt.value(), result, nprobes);
        LOG_KNOWHERE_INFO_ << "recall: " << recall;
        REQUIRE(recall > kKnnRecallThreshold);
    }
}
template <typename DataType>
void
CheckLowPrecisionKmeans() {
    constexpr int64_t rows = 100;
    constexpr int64_t dim = 4;
    std::vector<DataType> vectors(rows * dim);
    for (int64_t row = 0; row < rows; ++row) {
        const auto center = row < rows / 2 ? -10.0f : 10.0f;
        for (int64_t d = 0; d < dim; ++d) {
            vectors[row * dim + d] = DataType(center + float(row % 5) * 0.1f);
        }
    }

    auto dataset = knowhere::GenDataSet(rows, dim, vectors.data());
    auto cluster = knowhere::ClusterFactory::Instance().Create<DataType>(knowhere::ClusterEnum::CLUSTER_KMEANS).value();
    const knowhere::Json config = {{"num_clusters", 2},
                                   {"num_iter", 5},
                                   {knowhere::meta::METRIC_TYPE, knowhere::metric::L2},
                                   {knowhere::meta::NUM_BUILD_THREAD, 1}};

    auto train = cluster.Train(*dataset, config);
    REQUIRE(train.has_value());

    auto centroids = cluster.GetCentroids();
    REQUIRE(centroids.has_value());
    REQUIRE(centroids.value()->GetRows() == 2);
    REQUIRE(centroids.value()->GetDim() == dim);
    const auto* centroid_data = static_cast<const DataType*>(centroids.value()->GetTensor());
    for (int64_t i = 0; i < 2 * dim; ++i) {
        CHECK(std::isfinite(static_cast<float>(centroid_data[i])));
    }

    auto assignment = cluster.AssignWithDistance(*dataset, config);
    REQUIRE(assignment.has_value());
    REQUIRE(assignment.value()->GetIds() != nullptr);
    REQUIRE(assignment.value()->GetDistance() != nullptr);
    for (int64_t row = 0; row < rows; ++row) {
        CHECK(assignment.value()->GetIds()[row] >= 0);
        CHECK(assignment.value()->GetIds()[row] < 2);
        CHECK(std::isfinite(assignment.value()->GetDistance()[row]));
        CHECK(assignment.value()->GetDistance()[row] >= 0.0f);
    }
}

TEST_CASE("Kmeans preserves low precision centroid types", "[cluster low precision]") {
    SECTION("fp16") {
        CheckLowPrecisionKmeans<knowhere::fp16>();
    }
    SECTION("bf16") {
        CheckLowPrecisionKmeans<knowhere::bf16>();
    }
}

TEST_CASE("Native Kmeans builds deterministic IVF compaction groups", "[cluster compaction]") {
    auto cluster =
        knowhere::ClusterFactory::Instance().Create<knowhere::fp32>(knowhere::ClusterEnum::CLUSTER_KMEANS).value();
    auto result = cluster.BuildCompactionPlan({40, 0, 70, 20, 150}, {{"planner", "ivf"}, {"compaction_max_rows", 100}});
    REQUIRE(result.has_value());
    CHECK(result.value().row_count == 280);
    CHECK(result.value().centroid_count == 5);
    const std::vector<uint64_t> expected_counts{40, 0, 70, 20, 150};
    CHECK(result.value().centroid_counts == expected_counts);
    REQUIRE(result.value().centroid_groups.size() == 3);
    CHECK(result.value().centroid_groups[0].centroids == std::vector<uint32_t>{0});
    CHECK(result.value().centroid_groups[0].rows == 40);
    const std::vector<uint32_t> expected_middle_group{2, 3};
    CHECK(result.value().centroid_groups[1].centroids == expected_middle_group);
    CHECK(result.value().centroid_groups[1].rows == 90);
    CHECK(result.value().centroid_groups[2].centroids == std::vector<uint32_t>{4});
    CHECK(result.value().centroid_groups[2].rows == 150);

    CHECK(cluster.BuildCompactionPlan({1}, {{"planner", "metis"}, {"compaction_max_rows", 100}}).error() ==
          knowhere::Status::invalid_args);
    CHECK(cluster.BuildCompactionPlan({1}, {{"planner", "ivf"}}).error() == knowhere::Status::invalid_args);
}
