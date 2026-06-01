#include <cstddef>
#include <cstdint>

#if __has_include(<hip/hip_runtime.h>) && __has_include(<rocprim/rocprim.hpp>) &&                  \
    __has_include(<hipcub/hipcub.hpp>)
#define ROCM_OXIDE_HAS_ROCPRIM 1
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <rocprim/rocprim.hpp>
#else
#define ROCM_OXIDE_HAS_ROCPRIM 0
#endif

namespace
{
constexpr int unavailable_status = 1000001;

#if ROCM_OXIDE_HAS_ROCPRIM
template<typename T>
int reduce_sum(void*         temporary_storage,
               std::size_t* storage_size,
               const T*     input,
               T*           output,
               std::size_t  size,
               void*        stream)
{
    if(storage_size == nullptr || input == nullptr || output == nullptr)
    {
        return hipErrorInvalidValue;
    }
    return static_cast<int>(rocprim::reduce(temporary_storage,
                                            *storage_size,
                                            input,
                                            output,
                                            size,
                                            rocprim::plus<T>(),
                                            reinterpret_cast<hipStream_t>(stream)));
}

template<typename T>
int inclusive_sum(void*         temporary_storage,
                  std::size_t* storage_size,
                  const T*     input,
                  T*           output,
                  std::size_t  size,
                  void*        stream)
{
    if(storage_size == nullptr || input == nullptr || output == nullptr)
    {
        return hipErrorInvalidValue;
    }
    return static_cast<int>(rocprim::inclusive_scan(temporary_storage,
                                                   *storage_size,
                                                   input,
                                                   output,
                                                   size,
                                                   rocprim::plus<T>(),
                                                   reinterpret_cast<hipStream_t>(stream)));
}

template<typename T>
int exclusive_sum(void*         temporary_storage,
                  std::size_t* storage_size,
                  const T*     input,
                  T*           output,
                  T            initial_value,
                  std::size_t  size,
                  void*        stream)
{
    if(storage_size == nullptr || input == nullptr || output == nullptr)
    {
        return hipErrorInvalidValue;
    }
    return static_cast<int>(rocprim::exclusive_scan(temporary_storage,
                                                   *storage_size,
                                                   input,
                                                   output,
                                                   initial_value,
                                                   size,
                                                   rocprim::plus<T>(),
                                                   reinterpret_cast<hipStream_t>(stream)));
}

struct add_u32
{
    std::uint32_t value;

    __host__ __device__ std::uint32_t operator()(std::uint32_t input) const
    {
        return input + value;
    }
};
#endif
}

extern "C" int rocm_oxide_rocprim_available()
{
    return ROCM_OXIDE_HAS_ROCPRIM;
}

#if ROCM_OXIDE_HAS_ROCPRIM
extern "C" int rocm_oxide_rocprim_reduce_sum_u32(void*         temporary_storage,
                                                  std::size_t* storage_size,
                                                  const std::uint32_t* input,
                                                  std::uint32_t*       output,
                                                  std::size_t          size,
                                                  void*                stream)
{
    return reduce_sum(temporary_storage, storage_size, input, output, size, stream);
}

extern "C" int rocm_oxide_rocprim_reduce_sum_i32(void*               temporary_storage,
                                                  std::size_t*        storage_size,
                                                  const std::int32_t* input,
                                                  std::int32_t*       output,
                                                  std::size_t         size,
                                                  void*               stream)
{
    return reduce_sum(temporary_storage, storage_size, input, output, size, stream);
}

extern "C" int rocm_oxide_rocprim_reduce_sum_f32(void*        temporary_storage,
                                                  std::size_t* storage_size,
                                                  const float* input,
                                                  float*       output,
                                                  std::size_t  size,
                                                  void*        stream)
{
    return reduce_sum(temporary_storage, storage_size, input, output, size, stream);
}

extern "C" int rocm_oxide_rocprim_inclusive_sum_u32(void*         temporary_storage,
                                                     std::size_t* storage_size,
                                                     const std::uint32_t* input,
                                                     std::uint32_t*       output,
                                                     std::size_t          size,
                                                     void*                stream)
{
    return inclusive_sum(temporary_storage, storage_size, input, output, size, stream);
}

extern "C" int rocm_oxide_rocprim_inclusive_sum_i32(void*               temporary_storage,
                                                     std::size_t*        storage_size,
                                                     const std::int32_t* input,
                                                     std::int32_t*       output,
                                                     std::size_t         size,
                                                     void*               stream)
{
    return inclusive_sum(temporary_storage, storage_size, input, output, size, stream);
}

extern "C" int rocm_oxide_rocprim_inclusive_sum_f32(void*        temporary_storage,
                                                     std::size_t* storage_size,
                                                     const float* input,
                                                     float*       output,
                                                     std::size_t  size,
                                                     void*        stream)
{
    return inclusive_sum(temporary_storage, storage_size, input, output, size, stream);
}

extern "C" int rocm_oxide_rocprim_exclusive_sum_u32(void*         temporary_storage,
                                                     std::size_t* storage_size,
                                                     const std::uint32_t* input,
                                                     std::uint32_t*       output,
                                                     std::uint32_t        initial_value,
                                                     std::size_t          size,
                                                     void*                stream)
{
    return exclusive_sum(temporary_storage, storage_size, input, output, initial_value, size, stream);
}

extern "C" int rocm_oxide_rocprim_exclusive_sum_i32(void*               temporary_storage,
                                                     std::size_t*        storage_size,
                                                     const std::int32_t* input,
                                                     std::int32_t*       output,
                                                     std::int32_t        initial_value,
                                                     std::size_t         size,
                                                     void*               stream)
{
    return exclusive_sum(temporary_storage, storage_size, input, output, initial_value, size, stream);
}

extern "C" int rocm_oxide_rocprim_exclusive_sum_f32(void*        temporary_storage,
                                                     std::size_t* storage_size,
                                                     const float* input,
                                                     float*       output,
                                                     float        initial_value,
                                                     std::size_t  size,
                                                     void*        stream)
{
    return exclusive_sum(temporary_storage, storage_size, input, output, initial_value, size, stream);
}

extern "C" int rocm_oxide_rocprim_sort_keys_u32(void*                temporary_storage,
                                                 std::size_t*         storage_size,
                                                 const std::uint32_t* input,
                                                 std::uint32_t*       output,
                                                 std::size_t          size,
                                                 void*                stream)
{
    if(storage_size == nullptr || input == nullptr || output == nullptr)
    {
        return hipErrorInvalidValue;
    }
    return static_cast<int>(rocprim::radix_sort_keys(temporary_storage,
                                                    *storage_size,
                                                    input,
                                                    output,
                                                    size,
                                                    0,
                                                    8 * sizeof(std::uint32_t),
                                                    reinterpret_cast<hipStream_t>(stream)));
}

extern "C" int rocm_oxide_rocprim_select_flagged_u32(void*                temporary_storage,
                                                      std::size_t*         storage_size,
                                                      const std::uint32_t* input,
                                                      const std::uint8_t*  flags,
                                                      std::uint32_t*       output,
                                                      std::uint32_t*       selected_count,
                                                      std::size_t          size,
                                                      void*                stream)
{
    if(storage_size == nullptr || input == nullptr || flags == nullptr || output == nullptr
       || selected_count == nullptr)
    {
        return hipErrorInvalidValue;
    }
    return static_cast<int>(rocprim::select(temporary_storage,
                                           *storage_size,
                                           input,
                                           flags,
                                           output,
                                           selected_count,
                                           size,
                                           reinterpret_cast<hipStream_t>(stream)));
}

extern "C" int rocm_oxide_rocprim_transform_add_u32(const std::uint32_t* input,
                                                     std::uint32_t*       output,
                                                     std::uint32_t        addend,
                                                     std::size_t          size,
                                                     void*                stream)
{
    if(input == nullptr || output == nullptr)
    {
        return hipErrorInvalidValue;
    }
    return static_cast<int>(rocprim::transform(input,
                                              output,
                                              size,
                                              add_u32{addend},
                                              reinterpret_cast<hipStream_t>(stream)));
}
#else
extern "C" int rocm_oxide_rocprim_reduce_sum_u32(void*,
                                                  std::size_t*,
                                                  const std::uint32_t*,
                                                  std::uint32_t*,
                                                  std::size_t,
                                                  void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_reduce_sum_i32(void*,
                                                  std::size_t*,
                                                  const std::int32_t*,
                                                  std::int32_t*,
                                                  std::size_t,
                                                  void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_reduce_sum_f32(void*,
                                                  std::size_t*,
                                                  const float*,
                                                  float*,
                                                  std::size_t,
                                                  void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_inclusive_sum_u32(void*,
                                                     std::size_t*,
                                                     const std::uint32_t*,
                                                     std::uint32_t*,
                                                     std::size_t,
                                                     void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_inclusive_sum_i32(void*,
                                                     std::size_t*,
                                                     const std::int32_t*,
                                                     std::int32_t*,
                                                     std::size_t,
                                                     void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_inclusive_sum_f32(void*,
                                                     std::size_t*,
                                                     const float*,
                                                     float*,
                                                     std::size_t,
                                                     void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_exclusive_sum_u32(void*,
                                                     std::size_t*,
                                                     const std::uint32_t*,
                                                     std::uint32_t*,
                                                     std::uint32_t,
                                                     std::size_t,
                                                     void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_exclusive_sum_i32(void*,
                                                     std::size_t*,
                                                     const std::int32_t*,
                                                     std::int32_t*,
                                                     std::int32_t,
                                                     std::size_t,
                                                     void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_exclusive_sum_f32(void*,
                                                     std::size_t*,
                                                     const float*,
                                                     float*,
                                                     float,
                                                     std::size_t,
                                                     void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_sort_keys_u32(void*,
                                                 std::size_t*,
                                                 const std::uint32_t*,
                                                 std::uint32_t*,
                                                 std::size_t,
                                                 void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_select_flagged_u32(void*,
                                                      std::size_t*,
                                                      const std::uint32_t*,
                                                      const std::uint8_t*,
                                                      std::uint32_t*,
                                                      std::uint32_t*,
                                                      std::size_t,
                                                      void*)
{
    return unavailable_status;
}

extern "C" int rocm_oxide_rocprim_transform_add_u32(const std::uint32_t*,
                                                     std::uint32_t*,
                                                     std::uint32_t,
                                                     std::size_t,
                                                     void*)
{
    return unavailable_status;
}
#endif
