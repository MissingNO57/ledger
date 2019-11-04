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

#include "math/tensor.hpp"
#include "math/top_k.hpp"
#include "ml/exceptions/exceptions.hpp"
#include "ml/ops/ops.hpp"

#include <cassert>
#include <utility>
#include <vector>

namespace fetch {
namespace ml {
namespace ops {

template <class T>
class TopK : public fetch::ml::ops::Ops<T>
{
public:
  using TensorType     = T;
  using SizeType       = typename TensorType::SizeType;
  using TensorSizeType = fetch::math::Tensor<SizeType>;
  using DataType       = typename TensorType::Type;
  using ArrayPtrType   = std::shared_ptr<TensorType>;
  using VecTensorType  = typename Ops<T>::VecTensorType;
  using SPType         = OpTopKSaveableParams<T>;
  using MyType         = TopK<TensorType>;

  /**
   * One hot function based on tf.top_k
   * @param depth number of classes
   * @param axis
   * @param on_value TRUE value
   * @param off_value FALSE value
   */
  explicit TopK(SizeType k, bool sorted = true)
    : k_(k)
    , sorted_(sorted)
  {}

  explicit TopK(SPType const &sp)
    : Ops<T>(sp)
  {
    k_      = sp.k;
    sorted_ = sp.sorted;
  }

  ~TopK() override = default;

  std::shared_ptr<OpsSaveableParams> GetOpSaveableParams() override
  {
    SPType sp{};
    sp.k      = k_;
    sp.sorted = sorted_;

    return std::make_shared<SPType>(sp);
  }

  std::shared_ptr<fetch::ml::ops::Ops<TensorType>> MakeSharedCopy(
      std::shared_ptr<fetch::ml::ops::Ops<TensorType>> me) override
  {
    FETCH_UNUSED(me);
    assert(me.get() == this);

    auto copyshare = std::make_shared<MyType>(*this);  // calls default copy constructor of MyType

    return copyshare;
  }
  void Forward(VecTensorType const &inputs, TensorType &output) override
  {
    assert(inputs.size() == 1);
    assert(output.shape() == this->ComputeOutputShape(inputs));

    UpdateIndices(inputs);

    fetch::math::TopK<TensorType, TensorSizeType>(output, indices_, *(inputs.at(0)), k_, sorted_);
  }

  std::vector<TensorType> Backward(VecTensorType const &inputs,
                                   TensorType const &   error_signal) override
  {
    FETCH_UNUSED(error_signal);
    assert(inputs.size() == 1);

    // Forward needs to be run first
    assert(indices_.size() != 0);

    assert(error_signal.shape() == this->ComputeOutputShape(inputs));

    TensorType ret_signal(inputs.at(0)->shape());

    switch (error_signal.shape().size())
    {

    // 1D
    case 1:
    {
      for (SizeType i{0}; i < error_signal.shape().at(0); i++)
      {
        ret_signal.At(indices_.At(i)) = error_signal.At(i);
      }
    }
    break;

    // 2D
    case 2:
    {
      for (SizeType i{0}; i < error_signal.shape().at(0); i++)
      {
        for (SizeType j{0}; j < error_signal.shape().at(1); j++)
        {
          ret_signal.At(i, indices_.At(i, j)) = error_signal.At(i, j);
        }
      }
    }
    break;

    default:
    {
      throw exceptions::InvalidMode("Backward pass for more than 2D array not supported yet.");
    }
    break;
    }

    return {ret_signal};
  }

  std::vector<SizeType> ComputeOutputShape(VecTensorType const &inputs) const override
  {
    assert(inputs.size() == 1);

    std::vector<SizeType> ret_shape    = inputs.at(0)->shape();
    ret_shape.at(ret_shape.size() - 1) = k_;

    return ret_shape;
  }

  static constexpr OpType OpCode()
  {
    return OpType::OP_TOP_K;
  }
  static constexpr char const *DESCRIPTOR = "TopK";

private:
  SizeType       k_;
  bool           sorted_;
  TensorSizeType indices_;

  void UpdateIndices(VecTensorType const &inputs)
  {
    std::vector<SizeType> ret_shape = ComputeOutputShape(inputs);
    if (indices_.shape() != ret_shape)
    {
      indices_ = TensorSizeType(ret_shape);
    }
  }
};

}  // namespace ops
}  // namespace ml
}  // namespace fetch
