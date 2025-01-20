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

#pragma once

#include <cstring>
#include "knowhere/io_stream.h"


namespace knowhere {
/**
 * @brief LocalFileManager is used for placeholder purpose. It will not do anything to the file on disk.
 *
 * This class is not thread-safe.
 */
class MemoryStream : public InputStream {

    MemoryStream() = delete;

    explicit MemoryStream(uint8_t* data, size_t size) : data_(data), rp_(0), total_(size) {
    }

    size_t
    Size() const {
        return total_;
    }

    size_t
    Tell() const {
        return rp_;
    }

    bool
    Eof() const {
        return rp_ >= total_;
    }

    size_t
    Read(void* ptr, size_t size) {
        size_t rest = total_ - rp_;
        size_t read_size = std::min(rest, size);

        memcpy(ptr, data_ + rp_, read_size);
        return read_size;
    }

private:
    uint8_t* data_;
    size_t rp_ = 0;
    size_t total_ = 0;
};

}  // namespace knowhere
