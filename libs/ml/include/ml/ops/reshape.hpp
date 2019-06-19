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

#include "math/matrix_operations.hpp"
#include "ml/ops/ops.hpp"

namespace fetch {
namespace ml {
namespace ops {

template <class T>
class Reshape : public fetch::ml::BatchOps<T>
{
public:
  using ArrayType     = T;
  using SizeType      = typename ArrayType::SizeType;
  using ArrayPtrType  = std::shared_ptr<ArrayType>;
  using VecTensorType = typename BatchOps<T>::VecTensorType;

  Reshape(std::vector<SizeType> new_shape)
    : new_shape_(new_shape)
  {}
  ~Reshape() = default;

  void Forward(VecTensorType const &inputs, ArrayType &output)
  {
    ASSERT(inputs.size() == 1);
    ASSERT(output.shape() == ComputeOutputShape(inputs));
    ASSERT(inputs.front().get().size() == output.size());

    output.Assign(inputs.front().get());
  }

  std::vector<ArrayType> Backward(VecTensorType const &inputs, ArrayType const &error_signal)
  {
    ASSERT(inputs.size() == 1);
    ArrayType ret(inputs.front().get().shape());
    ret.Assign(error_signal);
    return {ret};
  }

  std::vector<SizeType> ComputeOutputShape(VecTensorType const &inputs) const
  {
    std::vector<SizeType> output_size;
    for (SizeType i{0}; i < new_shape_.size(); i++)
    {
      output_size.push_back(inputs.front().get().shape().at(new_shape_.at(i)));
    }

    return output_size;
  }

  static constexpr char const *DESCRIPTOR = "Reshape";

private:
  std::vector<SizeType> new_shape_;
};

}  // namespace ops
}  // namespace ml
}  // namespace fetch
