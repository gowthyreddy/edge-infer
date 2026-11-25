cmake_minimum_required(VERSION 3.20)
project(edge_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clang-tidy

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# Sanitizer builds: cmake -DSANITIZE=address,undefined  or  -DSANITIZE=thread
set(SANITIZE "" CACHE STRING "sanitizers to enable, e.g. address,undefined or thread")
if(SANITIZE)
    add_compile_options(-fsanitize=${SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
    add_link_options(-fsanitize=${SANITIZE})
endif()

# ---------------- dependencies ----------------
# Don't look for packages via PATH: an active conda env would otherwise supply its own
# GTest/libcurl and break linking against the system OpenCV.
set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH OFF)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(Threads REQUIRED)

# ONNX Runtime: official prebuilt package from NuGet (it has both linux-x64 and linux-arm64
# libraries plus the C++ headers), instead of building it from source.
include(FetchContent)
set(ORT_VERSION 1.22.0)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(ORT_ARCH arm64)
else()
    set(ORT_ARCH x64)
endif()
FetchContent_Declare(onnxruntime
    URL https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${ORT_VERSION}
    DOWNLOAD_NAME onnxruntime-${ORT_VERSION}.zip  # .nupkg is a zip; the name tells CMake how to extract it
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(onnxruntime)

set(ORT_LIB_DIR ${onnxruntime_SOURCE_DIR}/runtimes/linux-${ORT_ARCH}/native)
# the library's soname is libonnxruntime.so.1, but the package only ships libonnxruntime.so
file(CREATE_LINK libonnxruntime.so ${ORT_LIB_DIR}/libonnxruntime.so.1 SYMBOLIC)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ORT_LIB_DIR}/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_SOURCE_DIR}/build/native/include)

# ---------------- library ----------------
add_library(edge_infer src/models.cpp src/thread_pool.cpp src/pipeline.cpp)
target_include_directories(edge_infer PUBLIC include)
target_link_libraries(edge_infer PUBLIC ${OpenCV_LIBS} onnxruntime Threads::Threads)

# ---------------- apps ----------------
add_executable(infer_image apps/infer_image.cpp)
target_link_libraries(infer_image PRIVATE edge_infer)

add_executable(run_pipeline apps/run_pipeline.cpp)
target_link_libraries(run_pipeline PRIVATE edge_infer)

# ---------------- tests ----------------
enable_testing()
find_package(GTest REQUIRED)
add_executable(unit_tests tests/test_models.cpp tests/test_golden.cpp tests/test_concurrency.cpp
                          tests/test_pipeline.cpp)
target_link_libraries(unit_tests PRIVATE edge_infer GTest::gtest_main)
target_compile_definitions(unit_tests PRIVATE EDGE_INFER_ROOT="${CMAKE_SOURCE_DIR}")
include(GoogleTest)
gtest_discover_tests(unit_tests DISCOVERY_TIMEOUT 60
    PROPERTIES ENVIRONMENT "TSAN_OPTIONS=suppressions=${CMAKE_SOURCE_DIR}/tests/tsan.supp")
