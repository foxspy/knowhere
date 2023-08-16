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

#include <strings.h>

#include <regex>

#include "log.h"

namespace knowhere {
namespace {
static constexpr const char* version_regex = "^[vV]\\d+\\.\\d+\\.\\d+$";
static constexpr const char* minimal_vesion = "v0.0.0";
static constexpr const char* current_version = "v0.0.0";
static constexpr size_t version_group_size = 3;
static constexpr char delimiter = '.';

static bool
version_check(std::string version) {
    return std::regex_match(version.c_str(), std::regex(version_regex));
}

static std::vector<int32_t>
version_split(std::string version_str) {
    if (!version_check(version_str)) {
        return std::vector<int32_t>();
    }
    try {
        auto num_str = version_str.substr(1);
        auto right = num_str.find(delimiter);
        std::vector<int32_t> version_nums;
        while (right != std::string::npos) {
            auto version_num = num_str.substr(0, right);
            version_nums.emplace_back(atoi(version_num.c_str()));
            num_str = num_str.substr(right + 1);
            right = num_str.find(delimiter);
        }
        version_nums.emplace_back(atoi(num_str.c_str()));
        return version_nums;
    } catch (std::exception& e) {
        LOG_KNOWHERE_ERROR_ << "unexpected version code : " << version_str;
        return std::vector<int32_t>();
    }
}

}  // namespace

class Version {
 public:
    explicit Version(const char* version_code_) {
        version_code = version_code_;
        auto codes = version_split(version_code_);
        if (codes.size() == version_group_size) {
            major_version = codes[0];
            middle_version = codes[1];
            mini_version = codes[2];
        } else {
            major_version = unexpected_version_num;
            middle_version = unexpected_version_num;
            mini_version = unexpected_version_num;
        }
    }

    int32_t
    GetMajorVersion() {
        return major_version;
    }

    int32_t
    GetMiddleVersion() {
        return middle_version;
    }

    int32_t
    GetMiniVersion() {
        return mini_version;
    }

    bool
    CloudVersion() {
        return major_version > 0;
    }

    bool
    Valid() {
        return major_version != unexpected_version_num && middle_version != unexpected_version_num &&
               mini_version != unexpected_version_num;
    };

    std::string
    ToString() {
        return version_code;
    }

    static inline Version
    GetCurrentVersion() {
        return Version(current_version);
    }

    static inline Version
    GetMinimalSupport() {
        return Version(minimal_vesion);
    }

    static inline bool
    VersionSupport(Version version) {
        return GetMinimalSupport() < version;
    }

    friend bool
    operator<(const Version& lhs, const Version& rhs) {
        if (lhs.major_version != rhs.major_version) {
            return lhs.major_version < rhs.major_version;
        } else if (lhs.middle_version != rhs.middle_version) {
            return lhs.middle_version < rhs.middle_version;
        } else if (lhs.mini_version != rhs.mini_version) {
            return lhs.mini_version < rhs.mini_version;
        }
        return true;
    }

    friend bool
    operator>(const Version& lhs, const Version& rhs) {
        return operator<(rhs, lhs);
    }

 private:
    static constexpr int32_t unexpected_version_num = -1;
    std::string version_code;
    int32_t major_version;
    int32_t middle_version;
    int32_t mini_version;
};

}  // namespace knowhere
