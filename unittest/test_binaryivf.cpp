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

#include <gtest/gtest.h>
#include <thread>

#include "knowhere/common/Exception.h"
#include "knowhere/index/VecIndexFactory.h"
#include "knowhere/index/vector_index/ConfAdapterMgr.h"
#include "knowhere/index/vector_index/IndexBinaryIVF.h"
#include "knowhere/index/vector_index/adapter/VectorAdapter.h"
#include "unittest/Helper.h"
#include "unittest/range_utils.h"
#include "unittest/utils.h"

using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

class BinaryIVFTest : public DataGen,
                      public TestWithParam<knowhere::IndexMode> {
 protected:
    void
    SetUp() override {
        Init_with_default(true);
        index_mode_ = GetParam();
        index_type_ = knowhere::IndexEnum::INDEX_FAISS_BIN_IVFFLAT;
        index_ = knowhere::VecIndexFactory::GetInstance().CreateVecIndex(index_type_, index_mode_);
        conf_ = ParamGenerator::GetInstance().Gen(index_type_);
    }

    void
    TearDown() override {
    }

 protected:
    knowhere::Config conf_;
    knowhere::IndexMode index_mode_;
    knowhere::IndexType index_type_;
    knowhere::VecIndexPtr index_ = nullptr;
};

INSTANTIATE_TEST_CASE_P(
    METRICParameters,
    BinaryIVFTest,
    Values(knowhere::IndexMode::MODE_CPU));

TEST_P(BinaryIVFTest, binaryivf_basic) {
    assert(!xb_bin.empty());

    // null faiss index
    {
        ASSERT_ANY_THROW(index_->Serialize(conf_));
        ASSERT_ANY_THROW(index_->Query(query_dataset, conf_, nullptr));
        ASSERT_ANY_THROW(index_->AddWithoutIds(nullptr, conf_));
    }

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    ASSERT_GT(index_->Size(), 0);

    ASSERT_TRUE(index_->HasRawData(knowhere::GetMetaMetricType(conf_)));
    auto result = index_->GetVectorById(id_dataset, conf_);
    AssertBinVec(result, base_dataset, id_dataset, nq, dim);

    std::vector<int64_t> ids_invalid(nq, nb);
    auto id_dataset_invalid = knowhere::GenDatasetWithIds(nq, dim, ids_invalid.data());
    ASSERT_ANY_THROW(index_->GetVectorById(id_dataset_invalid, conf_));

    auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
    ASSERT_TRUE(adapter->CheckSearch(conf_, index_type_, index_mode_));

    auto result1 = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result1, nq, knowhere::GetMetaTopk(conf_));
    // PrintResult(result, nq, k);

    auto result2 = index_->Query(query_dataset, conf_, *bitset);
    AssertAnns(result2, nq, k, CheckMode::CHECK_NOT_EQUAL);
}

TEST_P(BinaryIVFTest, binaryivf_serialize) {
    auto serialize = [](const std::string& filename, knowhere::BinaryPtr& bin, uint8_t* ret) {
        {
            FileIOWriter writer(filename);
            writer(static_cast<void*>(bin->data.get()), bin->size);
        }
        FileIOReader reader(filename);
        reader(ret, bin->size);
    };

    // serialize index
    index_->BuildAll(base_dataset, conf_);
    auto binaryset = index_->Serialize(conf_);
    auto bin = binaryset.GetByName("BinaryIVF");

    std::string filename = temp_path("/tmp/binaryivf_test_serialize.bin");
    auto load_data = new uint8_t[bin->size];
    serialize(filename, bin, load_data);

    binaryset.clear();
    std::shared_ptr<uint8_t[]> data(load_data);
    binaryset.Append("BinaryIVF", data, bin->size);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, knowhere::GetMetaTopk(conf_));
    // PrintResult(result, nq, k);
}

TEST_P(BinaryIVFTest, binaryivf_slice) {
    // serialize index
    index_->BuildAll(base_dataset, conf_);
    auto binaryset = index_->Serialize(conf_);

    index_->Load(binaryset);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);
    auto result = index_->Query(query_dataset, conf_, nullptr);
    AssertAnns(result, nq, knowhere::GetMetaTopk(conf_));
    // PrintResult(result, nq, k);
}

TEST_P(BinaryIVFTest, binaryivf_range_search_hamming) {
    knowhere::MetricType metric_type = knowhere::metric::HAMMING;
    knowhere::SetMetaMetricType(conf_, knowhere::metric::HAMMING);

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    auto test_range_search_hamming = [&](const float range_filter, const float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunBinaryRangeSearchBF(golden_labels, golden_distances, golden_lims, knowhere::metric::HAMMING,
                               xb_bin.data(), nb, xq_bin.data(), nq, dim, radius, range_filter, bitset);

        auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
        ASSERT_TRUE(adapter->CheckRangeSearch(conf_, index_type_, index_mode_));

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult(result, metric_type, nq, radius, range_filter,
                               golden_labels.data(), golden_lims.data(), false, bitset);
    };

    for (std::pair<float, float> range: {
        std::make_pair<float, float>(0.0f, 45.0f),
        std::make_pair<float, float>(45.0f, 48.0f),
        std::make_pair<float, float>(48.0f, 50.0f)}) {
        knowhere::SetMetaRangeFilter(conf_, range.first);
        knowhere::SetMetaRadius(conf_, range.second);
        test_range_search_hamming(range.first, range.second, nullptr);
        test_range_search_hamming(range.first, range.second, *bitset);
    }
}

TEST_P(BinaryIVFTest, binaryivf_range_search_jaccard) {
    knowhere::MetricType metric_type = knowhere::metric::JACCARD;
    knowhere::SetMetaMetricType(conf_, knowhere::metric::JACCARD);

    // serialize index
    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    auto test_range_search_jaccard = [&](const float range_filter, const float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunBinaryRangeSearchBF(golden_labels, golden_distances, golden_lims, knowhere::metric::JACCARD,
                               xb_bin.data(), nb, xq_bin.data(), nq, dim, radius, range_filter, bitset);

        auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
        ASSERT_TRUE(adapter->CheckRangeSearch(conf_, index_type_, index_mode_));

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult(result, metric_type, nq, radius, range_filter,
                               golden_labels.data(), golden_lims.data(), false, bitset);
    };

    for (std::pair<float, float> range: {
        std::make_pair<float, float>(0.0f, 0.5f),
        std::make_pair<float, float>(0.5f, 0.55f),
        std::make_pair<float, float>(0.55f, 0.6f)}) {
        knowhere::SetMetaRangeFilter(conf_, range.first);
        knowhere::SetMetaRadius(conf_, range.second);
        test_range_search_jaccard(range.first, range.second, nullptr);
        test_range_search_jaccard(range.first, range.second, *bitset);
    }
}

TEST_P(BinaryIVFTest, binaryivf_range_search_tanimoto) {
    knowhere::MetricType metric_type = knowhere::metric::TANIMOTO;
    knowhere::SetMetaMetricType(conf_, knowhere::metric::TANIMOTO);

    index_->BuildAll(base_dataset, conf_);
    EXPECT_EQ(index_->Count(), nb);
    EXPECT_EQ(index_->Dim(), dim);

    auto qd = knowhere::GenDataset(nq, dim, xq_bin.data());

    auto test_range_search_tanimoto = [&](const float range_filter, const float radius, const faiss::BitsetView bitset) {
        std::vector<int64_t> golden_labels;
        std::vector<float> golden_distances;
        std::vector<size_t> golden_lims;
        RunBinaryRangeSearchBF(golden_labels, golden_distances, golden_lims, knowhere::metric::TANIMOTO,
                               xb_bin.data(), nb, xq_bin.data(), nq, dim, radius, range_filter, bitset);

        auto adapter = knowhere::AdapterMgr::GetInstance().GetAdapter(index_type_);
        ASSERT_TRUE(adapter->CheckRangeSearch(conf_, index_type_, index_mode_));

        auto result = index_->QueryByRange(qd, conf_, bitset);
        CheckRangeSearchResult(result, metric_type, nq, radius, range_filter,
                               golden_labels.data(), golden_lims.data(), false, bitset);
    };

    for (std::pair<float, float> range: {
        std::make_pair<float, float>(0.0f, 1.0f),
        std::make_pair<float, float>(1.0f, 1.2f),
        std::make_pair<float, float>(1.2f, 1.5f)}) {
        knowhere::SetMetaRangeFilter(conf_, range.first);
        knowhere::SetMetaRadius(conf_, range.second);
        test_range_search_tanimoto(range.first, range.second, nullptr);
        test_range_search_tanimoto(range.first, range.second, *bitset);
    }
}

TEST_P(BinaryIVFTest, binaryivf_range_search_superstructure) {
    knowhere::SetMetaMetricType(conf_, knowhere::metric::SUPERSTRUCTURE);
    ASSERT_ANY_THROW(index_->Train(base_dataset, conf_));
}

TEST_P(BinaryIVFTest, binaryivf_range_search_substructure) {
    knowhere::SetMetaMetricType(conf_, knowhere::metric::SUBSTRUCTURE);
    ASSERT_ANY_THROW(index_->Train(base_dataset, conf_));
}
