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

#include "math/standard_functions/abs.hpp"
#include "vm_modules/math/math.hpp"
#include "vm_modules/math/tensor/tensor.hpp"
#include "vm_modules/math/type.hpp"
#include "vm_test_toolkit.hpp"

#include "gmock/gmock.h"

#include <sstream>

using namespace fetch::vm;

namespace math_tensor_tests {

using ::testing::Between;

using DataType = fetch::vm_modules::math::DataType;

class MathTensorTests : public ::testing::Test
{
public:
  std::stringstream stdout;
  VmTestToolkit     toolkit{&stdout};
};

TEST_F(MathTensorTests, tensor_squeeze_test)
{
  static char const *tensor_serialiase_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(3);
      tensor_shape[0] = 4u64;
      tensor_shape[1] = 1u64;
      tensor_shape[2] = 4u64;
      var x = Tensor(tensor_shape);
      var squeezed_x = x.squeeze();
      return squeezed_x;
    endfunction
  )";

  Variant res;
  ASSERT_TRUE(toolkit.Compile(tensor_serialiase_src));
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  fetch::math::Tensor<DataType> gt({4, 4});

  EXPECT_TRUE(tensor->GetTensor().shape() == gt.shape());
}

/// GETTER AND SETTER TESTS ///

TEST_F(MathTensorTests, tensor_set_and_at_one_test)
{
  static char const *tensor_serialiase_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(1);
      tensor_shape[0] = 2u64;

      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(2.0fp64);

      y.setAt(0u64,x.at(0u64));
      y.setAt(1u64,x.at(1u64));

     return y;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_serialiase_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  fetch::math::Tensor<DataType> gt({2});
  gt.Fill(static_cast<DataType>(2.0));

  EXPECT_TRUE(gt.AllClose(tensor->GetTensor()));
}

TEST_F(MathTensorTests, tensor_set_and_at_two_test)
{
  static char const *tensor_serialiase_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 2u64;
      tensor_shape[1] = 2u64;

      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(2.0fp64);

      y.setAt(0u64,0u64,x.at(0u64,0u64));
      y.setAt(0u64,1u64,x.at(0u64,1u64));
      y.setAt(1u64,0u64,x.at(1u64,0u64));
      y.setAt(1u64,1u64,x.at(1u64,1u64));

     return y;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_serialiase_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  fetch::math::Tensor<DataType> gt({2, 2});
  gt.Fill(static_cast<DataType>(2.0));

  EXPECT_TRUE(gt.AllClose(tensor->GetTensor()));
}

TEST_F(MathTensorTests, tensor_set_and_at_three_test)
{
  static char const *tensor_serialiase_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(3);
      tensor_shape[0] = 2u64;
      tensor_shape[1] = 2u64;
      tensor_shape[2] = 2u64;

      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(2.0fp64);

      y.setAt(0u64,0u64,0u64,x.at(0u64,0u64,0u64));
      y.setAt(0u64,1u64,0u64,x.at(0u64,1u64,0u64));
      y.setAt(1u64,0u64,0u64,x.at(1u64,0u64,0u64));
      y.setAt(1u64,1u64,0u64,x.at(1u64,1u64,0u64));
      y.setAt(0u64,0u64,1u64,x.at(0u64,0u64,1u64));
      y.setAt(0u64,1u64,1u64,x.at(0u64,1u64,1u64));
      y.setAt(1u64,0u64,1u64,x.at(1u64,0u64,1u64));
      y.setAt(1u64,1u64,1u64,x.at(1u64,1u64,1u64));

     return y;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_serialiase_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  fetch::math::Tensor<DataType> gt({2, 2, 2});
  gt.Fill(static_cast<DataType>(2.0));

  EXPECT_TRUE(gt.AllClose(tensor->GetTensor()));
}

TEST_F(MathTensorTests, tensor_set_and_at_four_test)
{
  static char const *tensor_serialiase_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(4);
      tensor_shape[0] = 2u64;
      tensor_shape[1] = 2u64;
      tensor_shape[2] = 2u64;
      tensor_shape[3] = 2u64;

      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(2.0fp64);

      y.setAt(0u64,0u64,0u64,0u64,x.at(0u64,0u64,0u64,0u64));
      y.setAt(0u64,1u64,0u64,0u64,x.at(0u64,1u64,0u64,0u64));
      y.setAt(1u64,0u64,0u64,0u64,x.at(1u64,0u64,0u64,0u64));
      y.setAt(1u64,1u64,0u64,0u64,x.at(1u64,1u64,0u64,0u64));
      y.setAt(0u64,0u64,1u64,0u64,x.at(0u64,0u64,1u64,0u64));
      y.setAt(0u64,1u64,1u64,0u64,x.at(0u64,1u64,1u64,0u64));
      y.setAt(1u64,0u64,1u64,0u64,x.at(1u64,0u64,1u64,0u64));
      y.setAt(1u64,1u64,1u64,0u64,x.at(1u64,1u64,1u64,0u64));
      y.setAt(0u64,0u64,0u64,1u64,x.at(0u64,0u64,0u64,1u64));
      y.setAt(0u64,1u64,0u64,1u64,x.at(0u64,1u64,0u64,1u64));
      y.setAt(1u64,0u64,0u64,1u64,x.at(1u64,0u64,0u64,1u64));
      y.setAt(1u64,1u64,0u64,1u64,x.at(1u64,1u64,0u64,1u64));
      y.setAt(0u64,0u64,1u64,1u64,x.at(0u64,0u64,1u64,1u64));
      y.setAt(0u64,1u64,1u64,1u64,x.at(0u64,1u64,1u64,1u64));
      y.setAt(1u64,0u64,1u64,1u64,x.at(1u64,0u64,1u64,1u64));
      y.setAt(1u64,1u64,1u64,1u64,x.at(1u64,1u64,1u64,1u64));

     return y;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_serialiase_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  fetch::math::Tensor<DataType> gt({2, 2, 2, 2});
  gt.Fill(static_cast<DataType>(2.0));

  EXPECT_TRUE(gt.AllClose(tensor->GetTensor()));
}

TEST_F(MathTensorTests, tensor_set_from_string)
{
  static char const *tensor_from_string_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(3);
      tensor_shape[0] = 4u64;
      tensor_shape[1] = 1u64;
      tensor_shape[2] = 1u64;

      var x = Tensor(tensor_shape);
      x.fill(2.0fp64);

      var string_vals = "1.0, 1.0, 1.0, 1.0";
      x.fromString(string_vals);

      return x;

    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_from_string_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  fetch::math::Tensor<DataType> gt({4, 1, 1});
  gt.Fill(static_cast<DataType>(1.0));

  EXPECT_TRUE(gt.AllClose(tensor->GetTensor()));
}

/// TENSOR ARITHMETIC TESTS ///

TEST_F(MathTensorTests, tensor_equal_etch_test)
{
  static char const *tensor_equal_true_src = R"(
    function main() : Bool
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(7.0fp64);
      var result : Bool = (x == y);
      return result;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_equal_true_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const result = res.Get<bool>();
  EXPECT_TRUE(result == true);

  // test again for when not equal
  static char const *tensor_equal_false_src = R"(
    function main() : Bool
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(7.0fp64);
      y.setAt(0u64, 0u64, 1.0fp64);
      var result : Bool = (x == y);
      return result;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_equal_false_src));
  ASSERT_TRUE(toolkit.Run(&res));

  auto const result2 = res.Get<bool>();
  EXPECT_TRUE(result2 == false);
}

TEST_F(MathTensorTests, tensor_not_equal_etch_test)
{
  static char const *tensor_not_equal_true_src = R"(
    function main() : Bool
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(7.0fp64);
      var result : Bool = (x != y);
      return result;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_not_equal_true_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const result = res.Get<bool>();
  EXPECT_TRUE(result == false);

  // test again for when not equal
  static char const *tensor_not_equal_false_src = R"(
    function main() : Bool
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(7.0fp64);
      y.setAt(0u64, 0u64, 1.0fp64);
      var result : Bool = (x != y);
      return result;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_not_equal_false_src));
  ASSERT_TRUE(toolkit.Run(&res));

  auto const result2 = res.Get<bool>();
  EXPECT_TRUE(result2 == true);
}

TEST_F(MathTensorTests, tensor_add_test)
{
  static char const *tensor_add_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(7.0fp64);
      var result = x + y;
      return result;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_add_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor_ptr = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  auto                          tensor     = tensor_ptr->GetTensor();
  fetch::math::Tensor<DataType> gt({3, 3});
  gt.Fill(DataType(14.0));

  EXPECT_TRUE(gt.AllClose(tensor));
}

TEST_F(MathTensorTests, tensor_subtract_test)
{
  static char const *tensor_add_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(9.0fp64);
      var result = x - y;
      return result;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_add_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor_ptr = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  auto                          tensor     = tensor_ptr->GetTensor();
  fetch::math::Tensor<DataType> gt({3, 3});
  gt.Fill(DataType(-2.0));

  EXPECT_TRUE(gt.AllClose(tensor));
}

TEST_F(MathTensorTests, tensor_multiply_test)
{
  static char const *tensor_mul_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(7.0fp64);
      var result = x * y;
      return result;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_mul_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor_ptr = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  auto                          tensor     = tensor_ptr->GetTensor();
  fetch::math::Tensor<DataType> gt({3, 3});
  gt.Fill(DataType(49.0));

  EXPECT_TRUE(gt.AllClose(tensor));
}

TEST_F(MathTensorTests, tensor_divide_test)
{
  static char const *tensor_div_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(14.0fp64);
      var result = x / y;
      return result;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_div_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor_ptr = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  auto                          tensor     = tensor_ptr->GetTensor();
  fetch::math::Tensor<DataType> gt({3, 3});
  gt.Fill(DataType(0.5));

  EXPECT_TRUE(gt.AllClose(tensor));
}

TEST_F(MathTensorTests, tensor_inplace_multiply_test)
{
  static char const *tensor_inplace_mul_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(7.0fp64);
      x *= y;
      return x;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_inplace_mul_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor_ptr = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  auto                          tensor     = tensor_ptr->GetTensor();
  fetch::math::Tensor<DataType> gt({3, 3});
  gt.Fill(DataType(49.0));

  std::cout << "gt.ToString(): " << gt.ToString() << std::endl;
  std::cout << "tensor.ToString(): " << tensor.ToString() << std::endl;

  EXPECT_TRUE(gt.AllClose(tensor));
}

TEST_F(MathTensorTests, tensor_inplace_divide_test)
{
  static char const *tensor_inplace_div_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(14.0fp64);
      x /= y;
      return x;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_inplace_div_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor_ptr = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  auto                          tensor     = tensor_ptr->GetTensor();
  fetch::math::Tensor<DataType> gt({3, 3});
  gt.Fill(DataType(0.5));

  std::cout << "gt.ToString(): " << gt.ToString() << std::endl;
  std::cout << "tensor.ToString(): " << tensor.ToString() << std::endl;

  EXPECT_TRUE(gt.AllClose(tensor));
}

TEST_F(MathTensorTests, tensor_inplace_add_test)
{
  static char const *tensor_add_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(7.0fp64);
      x += y;
      return x;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_add_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor_ptr = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  auto                          tensor     = tensor_ptr->GetTensor();
  fetch::math::Tensor<DataType> gt({3, 3});
  gt.Fill(DataType(14.0));

  EXPECT_TRUE(gt.AllClose(tensor));
}

TEST_F(MathTensorTests, tensor_inplace_subtract_test)
{
  static char const *tensor_add_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      var y = Tensor(tensor_shape);
      x.fill(7.0fp64);
      y.fill(9.0fp64);
      x -= y;
      return x;
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_add_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor_ptr = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  auto                          tensor     = tensor_ptr->GetTensor();
  fetch::math::Tensor<DataType> gt({3, 3});
  gt.Fill(DataType(-2.0));

  EXPECT_TRUE(gt.AllClose(tensor));
}

TEST_F(MathTensorTests, tensor_negate_etch_test)
{
  static char const *tensor_negate_src = R"(
    function main() : Tensor
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      x.fill(7.0fp64);
      x = -x;
      return x;
    endfunction
  )";

  std::string const state_name{"tensor"};

  ASSERT_TRUE(toolkit.Compile(tensor_negate_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const tensor_ptr = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  auto       tensor     = tensor_ptr->GetTensor();
  std::cout << "tensor: " << tensor.ToString() << std::endl;

  fetch::math::Tensor<DataType> gt({3, 3});
  gt.Fill(DataType(-7.0));

  EXPECT_TRUE(gt.AllClose(tensor));
}

/// MATRIX OPERATION TESTS ///

TEST_F(MathTensorTests, tensor_min_test)
{
  fetch::math::Tensor<DataType> tensor = fetch::math::Tensor<DataType>::FromString(
      "0.5, 7.1, 9.1; 6.2, 7.1, 4.; -99.1, 14328.1, 10.0;");
  fetch::vm_modules::math::VMTensor vm_tensor(&toolkit.vm(), 0, tensor);

  DataType result = vm_tensor.Min();
  DataType gt{-99.1};

  EXPECT_TRUE(result == gt);
}

TEST_F(MathTensorTests, tensor_min_etch_test)
{
  static char const *tensor_min_src = R"(
    function main() : Fixed64
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      x.fill(7.0fp64);
      x.setAt(0u64, 1u64, -7.0fp64);
      x.setAt(1u64, 1u64, 23.1fp64);
      var ret = x.min();
      return ret;
    endfunction
  )";

  std::string const state_name{"tensor"};

  ASSERT_TRUE(toolkit.Compile(tensor_min_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const min_val = res.Get<DataType>();
  DataType   gt{-7.0};

  EXPECT_TRUE(gt == min_val);
}

TEST_F(MathTensorTests, tensor_max_test)
{
  fetch::math::Tensor<DataType> tensor = fetch::math::Tensor<DataType>::FromString(
      "0.5, 7.1, 9.1; 6.2, 7.1, 4.; -99.1, 14328.1, 10.0;");
  fetch::vm_modules::math::VMTensor vm_tensor(&toolkit.vm(), 0, tensor);

  DataType result = vm_tensor.Max();
  DataType gt{14328.1};

  EXPECT_TRUE(result == gt);
}

TEST_F(MathTensorTests, tensor_max_etch_test)
{
  static char const *tensor_max_src = R"(
    function main() : Fixed64
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      x.fill(7.0fp64);
      x.setAt(0u64, 1u64, -7.0fp64);
      x.setAt(1u64, 1u64, 23.1fp64);
      var ret = x.max();
      return ret;
    endfunction
  )";

  std::string const state_name{"tensor"};

  ASSERT_TRUE(toolkit.Compile(tensor_max_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const max_val = res.Get<DataType>();
  DataType   gt{23.1};

  std::cout << "gt: " << gt << std::endl;
  std::cout << "max_val: " << max_val << std::endl;

  EXPECT_TRUE(gt == max_val);
}

TEST_F(MathTensorTests, tensor_sum_test)
{
  fetch::math::Tensor<DataType> tensor = fetch::math::Tensor<DataType>::FromString(
      "0.5, 7.1, 9.1; 6.2, 7.1, 4.; -99.1, 14328.1, 10.0;");
  fetch::vm_modules::math::VMTensor vm_tensor(&toolkit.vm(), 0, tensor);

  DataType result = vm_tensor.Sum();
  DataType gt{14273.0};

  EXPECT_TRUE(fetch::math::Abs(gt - result) < DataType::TOLERANCE);
}

TEST_F(MathTensorTests, tensor_sum_etch_test)
{
  static char const *tensor_sum_src = R"(
    function main() : Fixed64
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 3u64;
      tensor_shape[1] = 3u64;
      var x = Tensor(tensor_shape);
      x.fill(7.0fp64);
      x.setAt(0u64, 1u64, -7.0fp64);
      x.setAt(1u64, 1u64, 23.1fp64);
      var ret = x.sum();
      return ret;
    endfunction
  )";

  std::string const state_name{"tensor"};

  ASSERT_TRUE(toolkit.Compile(tensor_sum_src));
  Variant res;
  ASSERT_TRUE(toolkit.Run(&res));

  auto const sum_val = res.Get<DataType>();
  DataType   gt{65.1};

  EXPECT_TRUE(fetch::math::Abs(gt - sum_val) < DataType::TOLERANCE);
}

/// SERIALISATION TESTS ///

TEST_F(MathTensorTests, tensor_state_test)
{
  static char const *tensor_serialiase_src = R"(
    function main()
      var tensor_shape = Array<UInt64>(2);
      tensor_shape[0] = 2u64;
      tensor_shape[1] = 10u64;
      var x = Tensor(tensor_shape);
      x.fill(7.0fp64);
      var state = State<Tensor>("tensor");
      state.set(x);
    endfunction
  )";

  std::string const state_name{"tensor"};

  ASSERT_TRUE(toolkit.Compile(tensor_serialiase_src));

  EXPECT_CALL(toolkit.observer(), Write(state_name, _, _));
  ASSERT_TRUE(toolkit.Run());

  static char const *tensor_deserialiase_src = R"(
    function main() : Tensor
      var state = State<Tensor>("tensor");
      return state.get();
    endfunction
  )";

  ASSERT_TRUE(toolkit.Compile(tensor_deserialiase_src));

  Variant res;
  EXPECT_CALL(toolkit.observer(), Exists(state_name));
  EXPECT_CALL(toolkit.observer(), Read(state_name, _, _)).Times(Between(1, 2));
  ASSERT_TRUE(toolkit.Run(&res));

  auto const                    tensor = res.Get<Ptr<fetch::vm_modules::math::VMTensor>>();
  fetch::math::Tensor<DataType> gt({2, 10});
  gt.Fill(static_cast<DataType>(7.0));

  EXPECT_TRUE(gt.AllClose(tensor->GetTensor()));
}

}  // namespace math_tensor_tests
