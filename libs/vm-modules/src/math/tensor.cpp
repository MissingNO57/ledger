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
#include "vm/array.hpp"
#include "vm/module.hpp"
#include "vm/object.hpp"
#include "vm_modules/math/tensor.hpp"
#include "vm_modules/math/type.hpp"
#include "vm_modules/utilities.hpp"

#include <cstdint>
#include <vector>

using namespace fetch::vm;

namespace fetch {
namespace vm_modules {
namespace math {

using ArrayType  = fetch::math::Tensor<VMTensor::DataType>;
using SizeType   = ArrayType::SizeType;
using SizeVector = ArrayType::SizeVector;

VMTensor::VMTensor(VM *vm, TypeId type_id, std::vector<uint64_t> const &shape)
  : Object(vm, type_id)
  , tensor_(shape)
  , estimator_(*this)
{}

VMTensor::VMTensor(VM *vm, TypeId type_id, ArrayType tensor)
  : Object(vm, type_id)
  , tensor_(std::move(tensor))
  , estimator_(*this)
{}

VMTensor::VMTensor(VM *vm, TypeId type_id)
  : Object(vm, type_id)
  , estimator_(*this)
{}

Ptr<VMTensor> VMTensor::Constructor(VM *vm, TypeId type_id, Ptr<Array<SizeType>> const &shape)
{
  return Ptr<VMTensor>{new VMTensor(vm, type_id, shape->elements)};
}

/////////////////////////////////////////////////////////
// Bind member function and their estimators to module //
/////////////////////////////////////////////////////////

void VMTensor::Bind(Module &module)
{
  module.CreateClassType<VMTensor>("Tensor")
      .CreateConstructor(&VMTensor::Constructor)
      .CreateSerializeDefaultConstructor([](VM *vm, TypeId type_id) -> Ptr<VMTensor> {
        return Ptr<VMTensor>{new VMTensor(vm, type_id)};
      })
      .CreateMemberFunction("at", &VMTensor::AtOne,
                            estimator_use(&VMTensor::TensorEstimator::AtOne))
      .CreateMemberFunction("at", &VMTensor::AtTwo,
                            estimator_use(&VMTensor::TensorEstimator::AtTwo))
      .CreateMemberFunction("at", &VMTensor::AtThree,
                            estimator_use(&VMTensor::TensorEstimator::AtThree))
      .CreateMemberFunction("at", &VMTensor::AtFour,
                            estimator_use(&VMTensor::TensorEstimator::AtFour))
      .CreateMemberFunction("setAt", &VMTensor::SetAtOne,
                            estimator_use(&VMTensor::TensorEstimator::SetAtOne))
      .CreateMemberFunction("setAt", &VMTensor::SetAtTwo,
                            estimator_use(&VMTensor::TensorEstimator::SetAtTwo))
      .CreateMemberFunction("setAt", &VMTensor::SetAtThree,
                            estimator_use(&VMTensor::TensorEstimator::SetAtThree))
      .CreateMemberFunction("setAt", &VMTensor::SetAtFour,
                            estimator_use(&VMTensor::TensorEstimator::SetAtFour))
      .CreateMemberFunction("size", &VMTensor::size,
                            estimator_use(&VMTensor::TensorEstimator::size))
      .CreateMemberFunction("fill", &VMTensor::Fill,
                            estimator_use(&VMTensor::TensorEstimator::Fill))
      .CreateMemberFunction("fillRandom", &VMTensor::FillRandom,
                            estimator_use(&VMTensor::TensorEstimator::FillRandom))
      .CreateMemberFunction("reshape", &VMTensor::Reshape,
                            estimator_use(&VMTensor::TensorEstimator::Reshape))
      .CreateMemberFunction("squeeze", &VMTensor::Squeeze,
                            estimator_use(&VMTensor::TensorEstimator::Squeeze))
      .CreateMemberFunction("transpose", &VMTensor::Transpose,
                            estimator_use(&VMTensor::TensorEstimator::Transpose))
      .CreateMemberFunction("unsqueeze", &VMTensor::Unsqueeze,
                            estimator_use(&VMTensor::TensorEstimator::Unsqueeze))
      .CreateMemberFunction("fromString", &VMTensor::FromString,
                            estimator_use(&VMTensor::TensorEstimator::FromString))
      .CreateMemberFunction("toString", &VMTensor::ToString,
                            estimator_use(&VMTensor::TensorEstimator::ToString));

  // Add support for Array of Tensors
  module.GetClassInterface<IArray>().CreateInstantiationType<Array<Ptr<VMTensor>>>();
}

SizeVector VMTensor::shape() const
{
  return tensor_.shape();
}

SizeType VMTensor::size() const
{
  return tensor_.size();
}

////////////////////////////////////
/// ACCESSING AND SETTING VALUES ///
////////////////////////////////////

DataType VMTensor::AtOne(SizeType idx1) const
{
  return tensor_.At(idx1);
}

DataType VMTensor::AtTwo(uint64_t idx1, uint64_t idx2) const
{
  return tensor_.At(idx1, idx2);
}

DataType VMTensor::AtThree(uint64_t idx1, uint64_t idx2, uint64_t idx3) const
{
  return tensor_.At(idx1, idx2, idx3);
}

DataType VMTensor::AtFour(uint64_t idx1, uint64_t idx2, uint64_t idx3, uint64_t idx4) const
{
  return tensor_.At(idx1, idx2, idx3, idx4);
}

void VMTensor::SetAtOne(uint64_t idx1, DataType const &value)
{
  tensor_.At(idx1) = value;
}

void VMTensor::SetAtTwo(uint64_t idx1, uint64_t idx2, DataType const &value)
{
  tensor_.At(idx1, idx2) = value;
}

void VMTensor::SetAtThree(uint64_t idx1, uint64_t idx2, uint64_t idx3, DataType const &value)
{
  tensor_.At(idx1, idx2, idx3) = value;
}

void VMTensor::SetAtFour(uint64_t idx1, uint64_t idx2, uint64_t idx3, uint64_t idx4,
                         DataType const &value)
{
  tensor_.At(idx1, idx2, idx3, idx4) = value;
}

void VMTensor::Copy(ArrayType const &other)
{
  tensor_.Copy(other);
}

void VMTensor::Fill(DataType const &value)
{
  tensor_.Fill(value);
}

void VMTensor::FillRandom()
{
  tensor_.FillUniformRandom();
}

Ptr<VMTensor> VMTensor::Squeeze()
{
  auto squeezed_tensor = tensor_.Copy();
  squeezed_tensor.Squeeze();
  return fetch::vm::Ptr<VMTensor>(new VMTensor(vm_, type_id_, squeezed_tensor));
}

Ptr<VMTensor> VMTensor::Unsqueeze()
{
  auto unsqueezed_tensor = tensor_.Copy();
  unsqueezed_tensor.Unsqueeze();
  return fetch::vm::Ptr<VMTensor>(new VMTensor(vm_, type_id_, unsqueezed_tensor));
}

bool VMTensor::Reshape(Ptr<Array<SizeType>> const &new_shape)
{
  return tensor_.Reshape(new_shape->elements);
}

void VMTensor::Transpose()
{
  tensor_.Transpose();
}

//////////////////////////////
/// PRINTING AND EXPORTING ///
//////////////////////////////

void VMTensor::FromString(fetch::vm::Ptr<fetch::vm::String> const &string)
{
  tensor_.Assign(fetch::math::Tensor<DataType>::FromString(string->string()));
}

Ptr<String> VMTensor::ToString() const
{
  return Ptr<String>{new String(vm_, tensor_.ToString())};
}

ArrayType &VMTensor::GetTensor()
{
  return tensor_;
}

ArrayType const &VMTensor::GetConstTensor()
{
  return tensor_;
}

bool VMTensor::SerializeTo(serializers::MsgPackSerializer &buffer)
{
  buffer << tensor_;
  return true;
}

bool VMTensor::DeserializeFrom(serializers::MsgPackSerializer &buffer)
{
  buffer >> tensor_;
  return true;
}

VMTensor::TensorEstimator &VMTensor::Estimator()
{
  return estimator_;
}

//////////////////////////
/// CHARGE ESTIMATIONS ///
//////////////////////////

VMTensor::TensorEstimator::TensorEstimator(VMTensor &tensor)
  : tensor_{tensor}
{}

ChargeAmount VMTensor::TensorEstimator::size()
{
  return low_charge;
}

ChargeAmount VMTensor::TensorEstimator::AtOne(TensorType::SizeType /*idx1*/)
{
  return low_charge;
}

ChargeAmount VMTensor::TensorEstimator::AtTwo(uint64_t /*idx1*/, uint64_t /*idx2*/)
{
  return low_charge;
}

ChargeAmount VMTensor::TensorEstimator::AtThree(uint64_t /*idx1*/, uint64_t /*idx2*/,
                                                uint64_t /*idx3*/)
{
  return low_charge;
}

ChargeAmount VMTensor::TensorEstimator::AtFour(uint64_t /*idx1*/, uint64_t /*idx2*/,
                                               uint64_t /*idx3*/, uint64_t /*idx4*/)
{
  return low_charge;
}

ChargeAmount VMTensor::TensorEstimator::SetAtOne(uint64_t /*idx1*/, DataType const & /*value*/)
{
  return low_charge;
}

ChargeAmount VMTensor::TensorEstimator::SetAtTwo(uint64_t /*idx1*/, uint64_t /*idx2*/,
                                                 DataType const & /*value*/)
{
  return low_charge;
}

ChargeAmount VMTensor::TensorEstimator::SetAtThree(uint64_t /*idx1*/, uint64_t /*idx2*/,
                                                   uint64_t /*idx3*/, DataType const & /*value*/)
{
  return low_charge;
}

ChargeAmount VMTensor::TensorEstimator::SetAtFour(uint64_t /*idx1*/, uint64_t /*idx2*/,
                                                  uint64_t /*idx3*/, uint64_t /*idx4*/,
                                                  DataType const & /*value*/)
{
  return low_charge;
}

ChargeAmount VMTensor::TensorEstimator::Fill(DataType const & /*value*/)
{
  return charge_func_of_tensor_size();
}

ChargeAmount VMTensor::TensorEstimator::FillRandom()
{
  return charge_func_of_tensor_size();
}

ChargeAmount VMTensor::TensorEstimator::Squeeze()
{
  return charge_func_of_tensor_size();
}

ChargeAmount VMTensor::TensorEstimator::Unsqueeze()
{
  return charge_func_of_tensor_size();
}

ChargeAmount VMTensor::TensorEstimator::Reshape(
    fetch::vm::Ptr<fetch::vm::Array<TensorType::SizeType>> const &new_shape)
{
  FETCH_UNUSED(new_shape);
  return charge_func_of_tensor_size();
}

ChargeAmount VMTensor::TensorEstimator::Transpose()
{
  return charge_func_of_tensor_size();
}

ChargeAmount VMTensor::TensorEstimator::FromString(fetch::vm::Ptr<fetch::vm::String> const &string)
{
  size_t val_size = 2;
  return static_cast<ChargeAmount>(static_cast<size_t>(string->Length()) / val_size);
}

ChargeAmount VMTensor::TensorEstimator::ToString()
{
  return charge_func_of_tensor_size();
}

ChargeAmount VMTensor::TensorEstimator::charge_func_of_tensor_size(size_t factor)
{
  return static_cast<ChargeAmount>(vm::CHARGE_UNIT * factor * tensor_.size());
}

}  // namespace math
}  // namespace vm_modules
}  // namespace fetch
