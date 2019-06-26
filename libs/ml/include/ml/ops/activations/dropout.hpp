#pragma once
//------------------------------------------------------------------------------
//
//   Copyright 2018-2019 Fetch.AI Limited
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//
//------------------------------------------------------------------------------

#include "core/macros.hpp"
#include "core/random/lfg.hpp"
#include "math/fundamental_operators.hpp"
#include "math/matrix_operations.hpp"
#include "ml/ops/ops.hpp"

namespace fetch {
namespace ml {
namespace ops {

template <class T>
class Dropout : public fetch::ml::Ops<T>
{
public:
  using ArrayType     = T;
  using DataType      = typename ArrayType::Type;
  using SizeType      = typename ArrayType::SizeType;
  using RNG           = fetch::random::LaggedFibonacciGenerator<>;
  using VecTensorType = typename Ops<T>::VecTensorType;

  Dropout(DataType const probability, SizeType const &random_seed = 25102015)
    : probability_(probability)
  {
    assert(probability >= 0.0 && probability <= 1.0);
    rng_.Seed(random_seed);
    drop_values_ = ArrayType{0};
  }

  virtual ~Dropout() = default;

  void Forward(VecTensorType const &inputs, ArrayType &output)
  {
    assert(inputs.size() == 1);
    assert(output.shape() == this->ComputeOutputShape(inputs));

    if (!this->is_training_)
    {
      output.Copy(inputs.front().get());
      return;
    }

    if (drop_values_.shape() != output.shape())
    {
      drop_values_ = ArrayType(inputs.front().get().shape());
    }
    UpdateRandomValues();

    fetch::math::Multiply(inputs.front().get(), drop_values_, output);
  }

  std::vector<ArrayType> Backward(VecTensorType const &inputs, ArrayType const &error_signal)
  {
    FETCH_UNUSED(inputs);
    assert(inputs.size() == 1);
    assert(error_signal.shape() == inputs.front(static_cast<DataType>(1)).get().shape());
    assert(drop_values_.shape() == inputs.front().get().shape());

    ArrayType return_signal{error_signal.shape()};

    // gradient of dropout is 1.0 for enabled neurons and 0.0 for disabled
    // multiply by error_signal (chain rule)
    if (this->is_training_)
    {
      fetch::math::Multiply(error_signal, drop_values_, return_signal);
    }
    else
    {
      return_signal.Copy(error_signal);
    }

    return {return_signal};
  }

  std::vector<SizeType> ComputeOutputShape(VecTensorType const &inputs) const
  {
    return inputs.front().get().shape();
  }

  static constexpr char const *DESCRIPTOR = "Dropout";

private:
  void UpdateRandomValues()
  {
    DataType zero{0};
    DataType one{1};

    double d_probability = static_cast<double>(probability_);
    auto   it            = drop_values_.begin();
    while (it.is_valid())
    {
      if (rng_.AsDouble() <= d_probability)
      {
        *it = one;
      }
      else
      {
        *it = zero;
      }
      ++it;
    }
  }

  ArrayType drop_values_;
  DataType  probability_;
  RNG       rng_;
};

}  // namespace ops
}  // namespace ml
}  // namespace fetch
