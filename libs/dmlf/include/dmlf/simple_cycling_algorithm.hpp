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

#include <memory>

#include "dmlf/ishuffle_algorithm.hpp"

namespace fetch {
namespace dmlf {

class SimpleCyclingAlgorithm : public IShuffleAlgorithm
{
public:
  SimpleCyclingAlgorithm(std::size_t count, std::size_t number_of_outputs_per_cycle);
  virtual ~SimpleCyclingAlgorithm()
  {}

  std::vector<std::size_t> getNextOutputs();

protected:
private:
  std::size_t next_output_index;
  std::size_t number_of_outputs_per_cycle;

  SimpleCyclingAlgorithm(const SimpleCyclingAlgorithm &other) = delete;
  SimpleCyclingAlgorithm &operator=(const SimpleCyclingAlgorithm &other)  = delete;
  bool                    operator==(const SimpleCyclingAlgorithm &other) = delete;
  bool                    operator<(const SimpleCyclingAlgorithm &other)  = delete;
};

}  // namespace dmlf
}  // namespace fetch
