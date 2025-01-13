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

#ifndef IO_STREAM_H
#define IO_STREAM_H

#include <string>

namespace knowhere {
class InputStream {
    /**
     * @brief get the total size of the stream
     *
     * @return
     */
    virtual size_t
    Size() const = 0;

    /**
     * @brief get the current position in the stream
     *
     * @return
     */
    virtual size_t
    Tell() const = 0;

    /**
     * @brief check if the end of the stream has been reached
     *
     * @return
     */
    virtual bool
    Eof() const = 0;

    /**
     * @brief reads a specified number of bytes from the stream into ptr
     *
     * @param ptr
     * @param size
     * @return
     */

    virtual size_t
    Read(void* ptr, size_t size) = 0;

    /**
     * @brief read data from the stream to a object with given type
     *
     * @tparam T
     * @return
     */
    template <typename T>
    T
    Read() {
        T value;
        Read(&value, sizeof(T));
        return value;
    };
};
}  // namespace knowhere

#endif
