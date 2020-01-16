#pragma once
//------------------------------------------------------------------------------
//
//   Copyright 2018-2020 Fetch.AI Limited
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

#include "core/serializers/main_serializer.hpp"
#include "math/base_types.hpp"
#include "math/statistics/mean.hpp"
#include "ml/dataloaders/dataloader.hpp"

namespace fetch {
namespace ml {
namespace optimisers {
/**
 * Training annealing config
 * @tparam T
 */
struct LearningRateParam
{
  enum class LearningRateDecay : uint8_t
  {
    EXPONENTIAL,
    LINEAR,
    NONE
  };
  LearningRateDecay mode                   = LearningRateDecay::NONE;
    fixed_point::fp32_t          starting_learning_rate = fetch::math::Type<fixed_point::fp32_t>("0.001");
    fixed_point::fp32_t          ending_learning_rate   = starting_learning_rate / fixed_point::fp32_t{10000};
    fixed_point::fp32_t          linear_decay_rate      = starting_learning_rate / fixed_point::fp32_t{10000};
    fixed_point::fp32_t          exponential_decay_rate = fetch::math::Type<fixed_point::fp32_t>("0.999");
};
}  // namespace optimisers
}  // namespace ml

namespace serializers {
/**
 * serializer for Learning Rate Params
 * @tparam TensorType
 */
template <typename D>
struct MapSerializer<ml::optimisers::LearningRateParam, D>
{
  using Type       = ml::optimisers::LearningRateParam;
  using DriverType = D;

  static uint8_t const LEARNING_RATE_DECAY_MODE = 1;
  static uint8_t const STARTING_LEARNING_RATE   = 2;
  static uint8_t const ENDING_LEARNING_RATE     = 3;
  static uint8_t const LINEAR_DECAY_RATE        = 4;
  static uint8_t const EXPONENTIAL_DECAY_RATE   = 5;

  template <typename Constructor>
  static void Serialize(Constructor &map_constructor, Type const &sp)
  {
    auto map = map_constructor(5);

    map.Append(LEARNING_RATE_DECAY_MODE, static_cast<uint8_t>(sp.mode));

    map.Append(STARTING_LEARNING_RATE, sp.starting_learning_rate);
    map.Append(ENDING_LEARNING_RATE, sp.ending_learning_rate);
    map.Append(LINEAR_DECAY_RATE, sp.linear_decay_rate);
    map.Append(EXPONENTIAL_DECAY_RATE, sp.exponential_decay_rate);
  }

  template <typename MapDeserializer>
  static void Deserialize(MapDeserializer &map, Type &sp)
  {
    uint8_t lrdm;
    map.ExpectKeyGetValue(LEARNING_RATE_DECAY_MODE, lrdm);
    sp.mode =
        static_cast<typename fetch::ml::optimisers::LearningRateParam::LearningRateDecay>(lrdm);

    map.ExpectKeyGetValue(STARTING_LEARNING_RATE, sp.starting_learning_rate);
    map.ExpectKeyGetValue(ENDING_LEARNING_RATE, sp.ending_learning_rate);
    map.ExpectKeyGetValue(LINEAR_DECAY_RATE, sp.linear_decay_rate);
    map.ExpectKeyGetValue(EXPONENTIAL_DECAY_RATE, sp.exponential_decay_rate);
  }
};
}  // namespace serializers

}  // namespace fetch
