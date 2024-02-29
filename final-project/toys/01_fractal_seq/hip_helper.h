// #ifndef __HIP_HELPER_H__
// #define __HIP_HELPER_H__
#pragma once

#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK_WITH_LOCATION(hip_return, file, line)             \
{                                                                   \
    hipError_t err = hip_return;                                    \
    if (err != hipSuccess) {                                        \
        const char *error_name = hipGetErrorName(err);              \
        const char *error_message = hipGetErrorString(err);         \
        std::cerr << "HIP Error at " << file << ":" << line << ": " \
                  << error_name << " (" << err << ")" << std::endl; \
        std::cerr << "  Message: " << error_message << std::endl;   \
        throw std::runtime_error("HIP error occurred");             \
    }                                                               \
}

#define HIP_ERRCHECK(hip_return) HIP_CHECK_WITH_LOCATION(hip_return, __FILE__, __LINE__)
// #endif //__HIP_HELPER_H__
