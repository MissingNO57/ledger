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
class BatchwiseFlattenOp : public fetch::ml::BatchOps<T>
{
public:
  using ArrayType     = T;
  using SizeType      = typename ArrayType::SizeType;
  using ArrayPtrType  = std::shared_ptr<ArrayType>;
  using VecTensorType = typename BatchOps<T>::VecTensorType;

  BatchwiseFlattenOp()          = default;
  virtual ~BatchwiseFlattenOp() = default;

  virtual void Forward(VecTensorType const &inputs, ArrayType &output)
  {
    ASSERT(inputs.size() == 1);
    ASSERT(output.shape() == ComputeOutputShape(inputs));
    input_shape_ = inputs.front().get().shape();

    BatchwiseFlatten(inputs.front().get(), output);
  }

  virtual std::vector<ArrayType> Backward(VecTensorType const &inputs,
                                          ArrayType const &    error_signal)
  {
    ASSERT(inputs.size() == 1);
    ArrayType ret(input_shape_);
    BatchwiseFlatten(error_signal, ret);

    return {ret};
  }

  virtual std::vector<SizeType> ComputeOutputShape(VecTensorType const &inputs) const
  {
    SizeType batch_size =
        inputs.at(0).get().shape().at(inputs.at(0).get().shape().size() - SizeType{1});
    SizeType data_size = 1;
    for (SizeType i{0}; i < inputs.at(0).get().shape().size() - SizeType{1}; i++)
    {
      data_size *= inputs.at(0).get().shape().at(i);
    }

    return {data_size, batch_size};
  }

  static constexpr char const *DESCRIPTOR = "BatchwiseFlatten";

private:
  std::vector<std::uint64_t> input_shape_;
};

}  // namespace ops
}  // namespace ml
}  // namespace fetch
