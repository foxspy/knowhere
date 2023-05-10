knowhere_file_glob(
  GLOB CARDINAL_SRCS
        thirdparty/cardinal/src/index/*.cpp
        thirdparty/cardinal/src/index/vector_index/*.cpp
        thirdparty/cardinal/src/module/*/*.cpp
        thirdparty/cardinal/src/module/serialize/*.cpp
        thirdparty/cardinal/src/utils/*/*.cpp)

if(WITH_DISKANN)
    add_definitions(-DENABLE_CARDINAL_DISKANN)
endif()

add_library(cardinal STATIC ${CARDINAL_SRCS})

include_directories(thirdparty/cardinal/third_party)

if (APPLE)
    include_directories(thirdparty/cardinal/src/apple_patch/*/*.h)
    target_link_libraries(cardinal PUBLIC tbb nlohmann_json::nlohmann_json)
else()
    find_package(aio REQUIRED)
    include_directories(${AIO_INCLUDE})
    target_link_libraries(cardinal PUBLIC ${AIO_LIBRARIES} tbb nlohmann_json::nlohmann_json)
endif()
