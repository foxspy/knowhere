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

#include <index/IndexProxy.h>

#include "cardinal_config_adapter.h"
#include "common/range_util.h"
#include "index/diskann/diskann_config.h"
#include "knowhere/factory.h"
#include "knowhere/index_node.h"
#include "knowhere/log.h"

namespace knowhere {

enum class CardinalIndexType { IndexHNSW, IndexDiskANN };

static std::map<cardinal::CardinalStatus, Status> statsMapper = {
    std::make_pair(cardinal::CardinalStatus::success, Status::success),
    std::make_pair(cardinal::CardinalStatus::invalid_build_args, Status::invalid_args),
    std::make_pair(cardinal::CardinalStatus::invalid_search_args, Status::invalid_args),
    std::make_pair(cardinal::CardinalStatus::prepare_failed, Status::empty_index)};

Status
convertStatus(cardinal::CardinalStatus status) {
    if (statsMapper.find(status) != statsMapper.end()) {
        return statsMapper[status];
    }
    return Status::invalid_args;
}

template <CardinalIndexType IndexType>
class CardinalIndexNode : public IndexNode {
 public:
    CardinalIndexNode(const Object& object) : index_proxy_(nullptr) {
    }

    Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Status::success) {
            return err;
        }
        err = Add(dataset, cfg);
        return err;
    }
    Status
    Train(const DataSet& dataset, const Config& cfg) override {
        CardinalIndexAdapter adapter;
        if (!index_proxy_) {
            auto config = adapter.convertCfg(cfg);
            if (!config.has_value()) {
                return config.error();
            }
            index_proxy_ = std::make_unique<cardinal::IndexProxy>();
            index_proxy_->create(config.value());
        }
        auto proxy_dataset = adapter.convertDataSet(true, dataset, cfg);
        if (!proxy_dataset.has_value()) {
            return proxy_dataset.error();
        }
        index_proxy_->train(proxy_dataset.value());
        return Status::success;
    }
    Status
    Add(const DataSet& dataset, const Config& cfg) override {
        CardinalIndexAdapter adapter;
        if (!index_proxy_) {
            auto config = adapter.convertCfg(cfg);
            if (!config.has_value()) {
                return config.error();
            }
            index_proxy_ = std::make_unique<cardinal::IndexProxy>();
            index_proxy_->create(config.value());
        }
        auto proxy_dataset = adapter.convertDataSet(true, dataset, cfg);
        if (!proxy_dataset.has_value()) {
            return proxy_dataset.error();
        }
        index_proxy_->add(proxy_dataset.value());
        return Status::success;
    }

    expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_proxy_) {
            LOG_KNOWHERE_WARNING_ << "search on empty index";
            return unexpected(Status::empty_index);
        }
        CardinalIndexAdapter adapter;
        auto proxy_dataset = adapter.convertDataSet(false, dataset, cfg);
        if (!proxy_dataset.has_value()) {
            return unexpected(proxy_dataset.error());
        }
        auto config = adapter.convertCfg(cfg, index_proxy_->count(), index_proxy_->dim());
        if (!config.has_value()) {
            return unexpected(config.error());
        }
        cardinal::Bitset bit_set(bitset.data(), bitset.count());
        auto result = index_proxy_->search(proxy_dataset.value(), config.value(), bit_set);
        if (!result.success()) {
            return unexpected(Status::empty_index);
        }
        return adapter.convertResult(result.getResult());
    }

    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_proxy_) {
            LOG_KNOWHERE_WARNING_ << "search on empty index";
            return unexpected(Status::empty_index);
        }
        CardinalIndexAdapter adapter;
        auto proxy_dataset = adapter.convertDataSet(false, dataset, cfg);
        if (!proxy_dataset.has_value()) {
            return unexpected(proxy_dataset.error());
        }
        auto config = adapter.convertCfg(cfg, index_proxy_->count(), index_proxy_->dim());
        if (!config.has_value()) {
            return unexpected(config.error());
        }
        cardinal::Bitset bit_set(bitset.data(), bitset.count());
        auto result = index_proxy_->range_search(proxy_dataset.value(), config.value(), bit_set);

        int64_t* ids = nullptr;
        float* distances = nullptr;
        size_t* lims = nullptr;

        GetRangeSearchResult(result.getResult().distances, result.getResult().labels, false, dataset.GetRows(),
                             config.value().upper_bound, config.value().lower_bound, distances, ids, lims);

        return GenResultDataSet(dataset.GetRows(), ids, distances, lims);
    }

    expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        float* data = nullptr;
        auto dim = Dim();
        auto rows = dataset.GetRows();
        auto ids = dataset.GetIds();
        try {
            data = new float[dim * rows];
            std::vector<cardinal::id_t> v_ids(rows);
            for (int64_t i = 0; i < rows; i++) {
                v_ids[i] = ids[i];
            }
            index_proxy_->get_vectors(v_ids, data);
            return GenResultDataSet(rows, dim, data);
        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "cardinal inner error: " << e.what();
            std::unique_ptr<float> auto_del(data);
            return unexpected(Status::hnsw_inner_error);
        }
    }
    expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }
    Status
    Serialize(BinarySet& binset) const override {
        if (!index_proxy_) {
            return Status::empty_index;
        }
        try {
            auto writer = std::make_shared<cardinal::MemoryWriter>();
            size_t size = index_proxy_->save(cardinal::BinaryWriter(writer));
            binset.Append("Cardinal", writer->bin(), size);
        } catch (const std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "error inner cardinal, " << e.what();
            return Status::faiss_inner_error;
        }
        return Status::success;
    }

    bool
    HasRawData(const std::string& metric_type) const override {
        return false;
    }

    Status
    DeserializeFromFile(const std::string& filename, const LoadConfig& config) override {
        return Status::success;
    }

    Status
    Deserialize(const BinarySet& binset) override {
        auto binary_index = binset.GetByName("Cardinal");
        auto reader = std::make_shared<cardinal::MemoryReader>(binary_index->data.get(), binary_index->size);
        if (!index_proxy_) {
            index_proxy_ = std::make_unique<cardinal::IndexProxy>();
        }
        reader->seek(0);
        index_proxy_->load(cardinal::BinaryReader(reader));
        return Status::success;
    }
    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        if constexpr (IndexType == CardinalIndexType::IndexHNSW) {
            return std::make_unique<HnswConfig>();
        } else if constexpr (IndexType == CardinalIndexType::IndexDiskANN) {
            return std::make_unique<DiskANNConfig>();
        }
        return std::make_unique<BaseConfig>();
    }
    int64_t
    Dim() const override {
        return index_proxy_->dim();
    }
    int64_t
    Size() const override {
        return 0;
    }
    int64_t
    Count() const override {
        return 0;
    }
    std::string
    Type() const override {
        if constexpr (IndexType == CardinalIndexType::IndexHNSW) {
            return IndexEnum::INDEX_HNSW_CLOUD;
        } else if constexpr (IndexType == CardinalIndexType::IndexDiskANN) {
            return IndexEnum::INDEX_DISKANN_CLOUD;
        }
        return IndexEnum::INVALID;
    }

    ~CardinalIndexNode() override {
    }

 private:
    std::unique_ptr<cardinal::IndexProxy> index_proxy_;
};

#ifdef ENABLE_CARDINAL_DISKANN
KNOWHERE_REGISTER_GLOBAL(DISKANN, [](const Object& object) {
    return Index<CardinalIndexNode<CardinalIndexType::IndexDiskANN>>::Create(object);
});
#endif

KNOWHERE_REGISTER_GLOBAL(HNSW, [](const Object& object) {
    return Index<CardinalIndexNode<CardinalIndexType::IndexHNSW>>::Create(object);
});
}  // namespace knowhere
