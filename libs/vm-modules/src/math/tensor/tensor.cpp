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
#include "vm_modules/math/tensor/tensor.hpp"
#include "vm_modules/math/tensor/tensor_estimator.hpp"
#include "vm_modules/math/type.hpp"
#include "vm_modules/use_estimator.hpp"

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

void VMTensor::Bind(Module &module)
{
  using Index = fetch::math::SizeType;
  module.CreateClassType<VMTensor>("Tensor")
      .CreateConstructor(&VMTensor::Constructor)
      .CreateSerializeDefaultConstructor([](VM *vm, TypeId type_id) -> Ptr<VMTensor> {
        return Ptr<VMTensor>{new VMTensor(vm, type_id)};
      })
      .CreateMemberFunction("at", &VMTensor::At<Index>, use_estimator(&TensorEstimator::AtOne))
      .CreateMemberFunction("at", &VMTensor::At<Index, Index>,
                            use_estimator(&TensorEstimator::AtTwo))
      .CreateMemberFunction("at", &VMTensor::At<Index, Index, Index>,
                            use_estimator(&TensorEstimator::AtThree))
      .CreateMemberFunction("at", &VMTensor::At<Index, Index, Index, Index>,
                            use_estimator(&TensorEstimator::AtFour))
      .CreateMemberFunction("setAt", &VMTensor::SetAt<Index, DataType>,
                            use_estimator(&TensorEstimator::SetAtOne))
      .CreateMemberFunction("setAt", &VMTensor::SetAt<Index, Index, DataType>,
                            use_estimator(&TensorEstimator::SetAtTwo))
      .CreateMemberFunction("setAt", &VMTensor::SetAt<Index, Index, Index, DataType>,
                            use_estimator(&TensorEstimator::SetAtThree))
      .CreateMemberFunction("setAt", &VMTensor::SetAt<Index, Index, Index, Index, DataType>,
                            use_estimator(&TensorEstimator::SetAtFour))
      .CreateMemberFunction("size", &VMTensor::size, use_estimator(&TensorEstimator::size))
      .CreateMemberFunction("fill", &VMTensor::Fill, use_estimator(&TensorEstimator::Fill))
      .CreateMemberFunction("fillRandom", &VMTensor::FillRandom,
                            use_estimator(&TensorEstimator::FillRandom))
      .CreateMemberFunction("min", &VMTensor::Min, use_estimator(&TensorEstimator::Min))
      .CreateMemberFunction("max", &VMTensor::Max, use_estimator(&TensorEstimator::Max))
      .CreateMemberFunction("reshape", &VMTensor::Reshape, use_estimator(&TensorEstimator::Reshape))
      .CreateMemberFunction("squeeze", &VMTensor::Squeeze, use_estimator(&TensorEstimator::Squeeze))
      .CreateMemberFunction("sum", &VMTensor::Sum, use_estimator(&TensorEstimator::Sum))
      // TODO - need to add the estimators, but enableOperator can't handle estimators yet
      .EnableOperator(Operator::Negate)
      .EnableOperator(Operator::Equal)
      .EnableOperator(Operator::NotEqual)
      .EnableOperator(Operator::Add)
      .EnableOperator(Operator::Subtract)
      .EnableOperator(Operator::InplaceAdd)
      .EnableOperator(Operator::InplaceSubtract)
      .EnableOperator(Operator::Multiply)
      .EnableOperator(Operator::Divide)
      //      .EnableOperator(Operator::InplaceMultiply)
      //      .EnableOperator(Operator::InplaceDivide)
      //      .EnableOperator(Operator::GreaterThan)
      .CreateMemberFunction("transpose", &VMTensor::Transpose,
                            use_estimator(&TensorEstimator::Transpose))
      .CreateMemberFunction("unsqueeze", &VMTensor::Unsqueeze,
                            use_estimator(&TensorEstimator::Unsqueeze))
      .CreateMemberFunction("fromString", &VMTensor::FromString,
                            use_estimator(&TensorEstimator::FromString))
      .CreateMemberFunction("toString", &VMTensor::ToString,
                            use_estimator(&TensorEstimator::ToString));

  // Add support for Array of Tensors
  module.GetClassInterface<IArray>().CreateInstantiationType<Array<Ptr<VMTensor>>>();
}

// void RightAdd(Variant &objectv, Variant &rhsv) override;
// void RightSubtract(Variant &objectv, Variant &rhsv) override;
// void InplaceRightSubtract(Ptr<Object> const &lhso, Variant const &rhsv) override;
// void LeftMultiply(Variant &lhsv, Variant &objectv) override;
// void RightMultiply(Variant &objectv, Variant &rhsv) override;
// void InplaceRightMultiply(Ptr<Object> const &lhso, Variant const &rhsv) override;
// void RightDivide(Variant &objectv, Variant &rhsv) override;
// void InplaceRightDivide(Ptr<Object> const &lhso, Variant const &rhsv) override;

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

template <typename... Indices>
VMTensor::DataType VMTensor::At(Indices... indices) const
{
  return tensor_.At(indices...);
}

template <typename... Args>
void VMTensor::SetAt(Args... args)
{
  tensor_.Set(args...);
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

/////////////////////////
/// BASIC ARITHMETIC  ///
/////////////////////////

bool VMTensor::IsEqual(vm::Ptr<Object> const &lhso, vm::Ptr<Object> const &rhso)
{
  Ptr<VMTensor> left   = lhso;
  Ptr<VMTensor> right  = rhso;
  bool          result = (left->GetTensor() == right->GetTensor());
  return result;
}

bool VMTensor::IsNotEqual(vm::Ptr<Object> const &lhso, vm::Ptr<Object> const &rhso)
{
  Ptr<VMTensor> left   = lhso;
  Ptr<VMTensor> right  = rhso;
  bool          result = (left->GetTensor() != right->GetTensor());
  return result;
}

void VMTensor::Add(vm::Ptr<Object> &lhso, vm::Ptr<Object> &rhso)
{
  Ptr<VMTensor> left  = lhso;
  Ptr<VMTensor> right = rhso;
  this->GetTensor()   = (left->GetTensor() + right->GetTensor());
}

void VMTensor::Subtract(vm::Ptr<Object> &lhso, vm::Ptr<Object> &rhso)
{
  Ptr<VMTensor> left  = lhso;
  Ptr<VMTensor> right = rhso;
  this->GetTensor()   = (left->GetTensor() - right->GetTensor());
}

void VMTensor::InplaceAdd(vm::Ptr<Object> const &lhso, vm::Ptr<Object> const &rhso)
{
  Ptr<VMTensor> left  = lhso;
  Ptr<VMTensor> right = rhso;
  left->GetTensor().InlineAdd(right->GetTensor());
}

void VMTensor::InplaceSubtract(vm::Ptr<Object> const &lhso, vm::Ptr<Object> const &rhso)
{
  Ptr<VMTensor> left  = lhso;
  Ptr<VMTensor> right = rhso;
  left->GetTensor().InlineSubtract(right->GetTensor());
}

void VMTensor::Multiply(vm::Ptr<Object> &lhso, vm::Ptr<Object> &rhso)
{
  Ptr<VMTensor> left  = lhso;
  Ptr<VMTensor> right = rhso;
  this->GetTensor()   = (left->GetTensor() * right->GetTensor());
}

void VMTensor::Divide(vm::Ptr<Object> &lhso, vm::Ptr<Object> &rhso)
{
  Ptr<VMTensor> left  = lhso;
  Ptr<VMTensor> right = rhso;
  this->GetTensor()   = (left->GetTensor() / right->GetTensor());
}

void VMTensor::Negate(fetch::vm::Ptr<Object> &object)
{
  Ptr<VMTensor> operand = object;
  Ptr<VMTensor> t       = Ptr<VMTensor>{new VMTensor(this->vm_, this->type_id_, shape())};
  fetch::math::Multiply(operand->GetTensor(), DataType(-1), t->GetTensor());
  object = std::move(t);
}

//  virtual void        InplaceAdd(Ptr<Object> const &lhso, Ptr<Object> const &rhso);
//  virtual void        InplaceRightAdd(Ptr<Object> const &lhso, Variant const &rhsv);
//  virtual void        Subtract(Ptr<Object> &lhso, Ptr<Object> &rhso);
//  virtual void        LeftSubtract(Variant &lhsv, Variant &objectv);
//  virtual void        RightSubtract(Variant &objectv, Variant &rhsv);
//  virtual void        InplaceSubtract(Ptr<Object> const &lhso, Ptr<Object> const &rhso);
//  virtual void        InplaceRightSubtract(Ptr<Object> const &lhso, Variant const &rhsv);
//  virtual void        Multiply(Ptr<Object> &lhso, Ptr<Object> &rhso);
//  virtual void        LeftMultiply(Variant &lhsv, Variant &objectv);
//  virtual void        RightMultiply(Variant &objectv, Variant &rhsv);
//  virtual void        InplaceMultiply(Ptr<Object> const &lhso, Ptr<Object> const &rhso);
//  virtual void        InplaceRightMultiply(Ptr<Object> const &lhso, Variant const &rhsv);
//  virtual void        Divide(Ptr<Object> &lhso, Ptr<Object> &rhso);
//  virtual void        LeftDivide(Variant &lhsv, Variant &objectv);
//  virtual void        RightDivide(Variant &objectv, Variant &rhsv);
//  virtual void        InplaceDivide(Ptr<Object> const &lhso, Ptr<Object> const &rhso);
//  virtual void        InplaceRightDivide(Ptr<Object> const &lhso, Variant const &rhsv);

//      .EnableOperator(Operator::Equal)
//      .EnableOperator(Operator::NotEqual)
//      .EnableOperator(Operator::LessThan)
//      .EnableOperator(Operator::Add)
//      .EnableOperator(Operator::Subtract)
//      .EnableOperator(Operator::InplaceAdd)
//      .EnableOperator(Operator::InplaceSubtract)
//      .EnableOperator(Operator::Multiply)
//      .EnableOperator(Operator::Divide)
//      .EnableOperator(Operator::InplaceMultiply)
//      .EnableOperator(Operator::InplaceDivide)
//      .EnableOperator(Operator::GreaterThan)
// void Add(Ptr<Object> &lhso, Ptr<Object> &rhso) override;
//
// void RightAdd(Variant &objectv, Variant &rhsv) override;
//
// void InplaceAdd(Ptr<Object> const &lhso, Ptr<Object> const &rhso) override;
//
// void InplaceRightAdd(Ptr<Object> const &lhso, Variant const &rhsv) override;
//
// void Subtract(Ptr<Object> &lhso, Ptr<Object> &rhso) override;
//
// void RightSubtract(Variant &objectv, Variant &rhsv) override;
//
// void InplaceSubtract(Ptr<Object> const &lhso, Ptr<Object> const &rhso) override;
//
// void InplaceRightSubtract(Ptr<Object> const &lhso, Variant const &rhsv) override;
//
// void Multiply(Ptr<Object> &lhso, Ptr<Object> &rhso) override;
//
// void LeftMultiply(Variant &lhsv, Variant &objectv) override;
//
// void RightMultiply(Variant &objectv, Variant &rhsv) override;
//
// void InplaceRightMultiply(Ptr<Object> const &lhso, Variant const &rhsv) override;
//
// void RightDivide(Variant &objectv, Variant &rhsv) override;
//
// void InplaceRightDivide(Ptr<Object> const &lhso, Variant const &rhsv) override;

/////////////////////////
/// MATRIX OPERATIONS ///
/////////////////////////

DataType VMTensor::Min()
{
  return fetch::math::Min(tensor_);
}

DataType VMTensor::Max()
{
  return fetch::math::Max(tensor_);
}

DataType VMTensor::Sum()
{
  return fetch::math::Sum(tensor_);
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

TensorEstimator &VMTensor::Estimator()
{
  return estimator_;
}

}  // namespace math
}  // namespace vm_modules
}  // namespace fetch
