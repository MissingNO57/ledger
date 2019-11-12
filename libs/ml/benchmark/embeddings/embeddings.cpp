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

#include "logging/logging.hpp"
#include "math/tensor.hpp"
#include "ml/core/graph.hpp"
#include "ml/layers/fully_connected.hpp"
#include "ml/ops/activations/relu.hpp"
#include "ml/ops/loss_functions/mean_square_error_loss.hpp"
#include "ml/optimisation/adam_optimiser.hpp"
#include "ml/optimisation/lazy_adam_optimiser.hpp"

#include "benchmark/benchmark.h"

#include <memory>
#include <string>

/**
 *  Point of this benchmark is to get an idea of the overall time cost of training a simple
 * embeddings model under different hyperparameters and to compare performance benefits of sparse
 * optimisers with normal optimisers
 */

namespace fetch {
namespace ml {
namespace benchmark {

//////////////////////////
/// reusable functions ///
//////////////////////////

template <typename TypeParam>
std::shared_ptr<fetch::ml::Graph<TypeParam>> PrepareTestGraph(
    typename TypeParam::SizeType embedding_dimensions, typename TypeParam::SizeType n_datapoints,
    std::string &input_name, std::string &label_name, std::string &error_name)
{
  std::shared_ptr<fetch::ml::Graph<TypeParam>> g(std::make_shared<fetch::ml::Graph<TypeParam>>());

  input_name = g->template AddNode<fetch::ml::ops::PlaceHolder<TypeParam>>("", {});

  std::string output_name = g->template AddNode<fetch::ml::ops::Embeddings<TypeParam>>(
      "Embeddings", {input_name}, embedding_dimensions, n_datapoints);

  label_name = g->template AddNode<fetch::ml::ops::PlaceHolder<TypeParam>>("", {});
  error_name = g->template AddNode<fetch::ml::ops::MeanSquareErrorLoss<TypeParam>>(
      "Error", {output_name, label_name});

  return g;
}

template <typename T, fetch::math::SizeType B, fetch::math::SizeType S, fetch::math::SizeType D,
          fetch::math::SizeType E, typename OptimiserType>
void BM_Setup_And_Train_Embeddings(::benchmark::State &state)
{
  using SizeType   = fetch::math::SizeType;
  using DataType   = T;
  using TensorType = fetch::math::Tensor<DataType>;

  fetch::SetGlobalLogLevel(fetch::LogLevel::ERROR);

  SizeType batch_size           = B;
  SizeType embedding_dimensions = S;
  SizeType n_datapoints         = D;
  SizeType n_epochs             = E;

  auto learning_rate = DataType{0.1f};

  // Prepare data and labels
  TensorType data({1, batch_size});
  TensorType gt({embedding_dimensions, 1, batch_size});
  data.FillUniformRandomIntegers(0, static_cast<int64_t>(n_datapoints));
  gt.FillUniformRandom();

  for (auto _ : state)
  {
    // make a graph
    std::string input_name;
    std::string label_name;
    std::string error_name;
    auto        g = PrepareTestGraph<TensorType>(embedding_dimensions, n_datapoints, input_name,
                                          label_name, error_name);

    // Initialise Optimiser
    OptimiserType optimiser(g, {input_name}, label_name, error_name, learning_rate);

    // Do optimisation
    for (SizeType i = 0; i < n_epochs; ++i)
    {
      data.FillUniformRandomIntegers(0, static_cast<int64_t>(n_datapoints));
      optimiser.Run({data}, gt);
    }
  }
}

// Normal Adam tests
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 1, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 10, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 100, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 1000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 10000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 20000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 30000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 40000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 50000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 60000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 70000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 80000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 90000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 100000, 500, 10000, 10,
                   fetch::ml::optimisers::AdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);

// Sparse LazyAdam tests
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 1, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 10, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 100, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 1000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 10000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 20000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 30000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 40000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 50000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 60000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 70000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 80000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 90000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_Setup_And_Train_Embeddings, float, 100000, 500, 10000, 10,
                   fetch::ml::optimisers::LazyAdamOptimiser<fetch::math::Tensor<float>>)
    ->Unit(::benchmark::kMillisecond);
}  // namespace benchmark
}  // namespace ml
}  // namespace fetch

BENCHMARK_MAIN();
