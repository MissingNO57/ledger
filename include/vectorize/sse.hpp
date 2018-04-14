#ifndef VECTORIZE_SSE_HPPP
#define VECTORIZE_SSE_HPPP
#include "vectorize/vectorize_constants.hpp"
#include "vectorize/register.hpp"

#include <emmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>
#include <cstdint>
#include <cstddef>


namespace fetch {
namespace vectorize {
  
  namespace details {
    template< typename T, std::size_t N >
    struct UnrollSet {
      static void Set(T *ptr, T const &c) {
        (*ptr)  = c;
        UnrollSet<T, N - 1>::Set(ptr+1, c);
      }
    };
    
    template<typename T >
    struct UnrollSet<T,0> {
      static void Set(T *ptr, T const &c) { }
    };
  };
  
// SSE integers
template <typename T>
class VectorRegister<T, 128, __m128i> {
 public:
  typedef T type;
  typedef __m128i mm_register_type;
  
  enum {
    E_REGISTER_SIZE = sizeof(mm_register_type),
    E_BLOCK_COUNT = E_REGISTER_SIZE / sizeof(type)
  };
  
  static_assert((E_BLOCK_COUNT * sizeof(type)) == E_REGISTER_SIZE,
                "type cannot be contained in the given register size.");

  VectorRegister() {}
  VectorRegister(type const *d) { data_ = _mm_load_si128((mm_register_type *)d); }
  VectorRegister(type const &c)  {
    alignas(16) type constant[E_BLOCK_COUNT];
    details::UnrollSet< type, E_BLOCK_COUNT >::Set(constant, c);
    data_ = _mm_load_si128((mm_register_type *)constant);
    
  }
  VectorRegister(mm_register_type const &d) : data_(d) {}
  VectorRegister(mm_register_type &&d) : data_(d) {}

  explicit operator mm_register_type() { return data_; }

  void Store(type *ptr) const { _mm_store_si128((mm_register_type *)ptr, data_); }
  void Stream(type *ptr) const { _mm_stream_si128((mm_register_type *)ptr, data_); }
  
  mm_register_type const &data() const { return data_; }
  mm_register_type &data()  { return data_; }  
 private:
  mm_register_type data_;

};


template <>
class VectorRegister<float, 128, __m128> {
 public:
  typedef float type;
  typedef __m128 mm_register_type;
  
  enum {
    E_REGISTER_SIZE = sizeof(mm_register_type),
    E_BLOCK_COUNT = E_REGISTER_SIZE / sizeof(type)
  };
  
  static_assert((E_BLOCK_COUNT * sizeof(type)) == E_REGISTER_SIZE,
                "type cannot be contained in the given register size.");

  VectorRegister() {}
  VectorRegister(type const *d) { data_ = _mm_load_ps(d); }
  VectorRegister(mm_register_type const &d) : data_(d) {}
  VectorRegister(mm_register_type &&d) : data_(d) {}
  VectorRegister(type const &c)  {
    alignas(16) type constant[E_BLOCK_COUNT];
    details::UnrollSet< type, E_BLOCK_COUNT >::Set(constant, c);
    data_ = _mm_load_ps(constant);
    
  }
  
  explicit operator mm_register_type() { return data_; }

  void Store(type *ptr) const { _mm_store_ps(ptr, data_); }
  void Stream(type *ptr) const { _mm_stream_ps(ptr, data_); }

  mm_register_type const &data() const { return data_; }
  mm_register_type &data()  { return data_; }  
 private:
  mm_register_type data_;
};
  
template <>
class VectorRegister<double, 128, __m128d> {
 public:
  typedef double type;
  typedef __m128d mm_register_type;
  
  enum {
    E_REGISTER_SIZE = sizeof(mm_register_type),
    E_BLOCK_COUNT = E_REGISTER_SIZE / sizeof(type)
  };
  
  static_assert((E_BLOCK_COUNT * sizeof(type)) == E_REGISTER_SIZE,
                "type cannot be contained in the given register size.");

  VectorRegister() {}
  VectorRegister(type const *d) { data_ = _mm_load_pd(d); }
  VectorRegister(mm_register_type const &d) : data_(d) {}
  VectorRegister(mm_register_type &&d) : data_(d) {}
  VectorRegister(type const &c)  {
    alignas(16) type constant[E_BLOCK_COUNT];
    details::UnrollSet< type, E_BLOCK_COUNT >::Set(constant, c);
    data_ = _mm_load_pd(constant);
    
  }
  
  explicit operator mm_register_type() { return data_; }

  void Store(type *ptr) const { _mm_stream_pd(ptr, data_); }
  void Stream(type *ptr) const { _mm_stream_pd(ptr, data_); }
  
  mm_register_type const &data() const { return data_; }
  mm_register_type &data()  { return data_; }  
 private:
  mm_register_type data_;
};


  template <typename T, std::size_t N, typename S = typename VectorInfo<T,N>::register_type  >
  class VectorRegisterIterator {
  public:
    typedef T type;
    typedef VectorRegister< T, N, S > vector_register_type;
    typedef typename vector_register_type::mm_register_type mm_register_type;

    VectorRegisterIterator(type const *d) { ptr_ = (mm_register_type*)d; }
    
    void Next(vector_register_type &m) {
      m.data() = *ptr_;
      ++ptr_;
    }
  private:
    mm_register_type *ptr_;
  };
  
#define AILIB_ADD_OPERATOR(op, type, L, fnc)                            \
  inline VectorRegister<type, 128, L>                                   \
  operator op( VectorRegister<type,128, L> const &a,                    \
               VectorRegister<type,128, L> const &b) {                  \
    L ret = fnc(a.data(), b.data());                                    \
    return VectorRegister<type, 128, L>(ret);                           \
  }

AILIB_ADD_OPERATOR(*, int, __m128i, _mm_mullo_epi32);
AILIB_ADD_OPERATOR(-, int, __m128i, _mm_sub_epi32);
//AILIB_ADD_OPERATOR(/, int, __m128i, _mm_div_epi32);
AILIB_ADD_OPERATOR(+, int, __m128i, _mm_add_epi32);  

AILIB_ADD_OPERATOR(*, float, __m128, _mm_mul_ps);
AILIB_ADD_OPERATOR(-, float, __m128, _mm_sub_ps);  
AILIB_ADD_OPERATOR(/, float, __m128, _mm_div_ps);
AILIB_ADD_OPERATOR(+, float, __m128, _mm_add_ps);  

AILIB_ADD_OPERATOR(*, double, __m128d, _mm_mul_pd);
AILIB_ADD_OPERATOR(-, double, __m128d, _mm_sub_pd);  
AILIB_ADD_OPERATOR(/, double, __m128d, _mm_div_pd);
AILIB_ADD_OPERATOR(+, double, __m128d, _mm_add_pd);  
  
  
#undef AILIB_ADD_OPERATOR

#undef REQUIRED_SSE
};
};
#endif
