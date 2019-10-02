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

#include "core/mutex.hpp"
#include "core/serializers/base_types.hpp"
#include "dmlf/abstract_learner_networker.hpp"
#include <list>
#include <map>
#include <memory>

namespace fetch {
namespace dmlf {

class LocalLearnerNetworker : public AbstractLearnerNetworker
{
public:
  using PeerP = std::shared_ptr<LocalLearnerNetworker>;
  using Peers = std::vector<PeerP>;

  LocalLearnerNetworker();
  ~LocalLearnerNetworker() override;
  void pushUpdate(const std::shared_ptr<UpdateInterface> &update) override;

  std::size_t getPeerCount() const override
  {
    return peers.size();
  }
  void addPeers(Peers new_peers);
  void clearPeers();

  LocalLearnerNetworker(const LocalLearnerNetworker &other) = delete;
  LocalLearnerNetworker &operator=(const LocalLearnerNetworker &other)  = delete;
  bool                   operator==(const LocalLearnerNetworker &other) = delete;
  bool                   operator<(const LocalLearnerNetworker &other)  = delete;

protected:
private:
  using Mutex = fetch::Mutex;
  using Lock  = std::unique_lock<Mutex>;
  using Bytes = AbstractLearnerNetworker::Bytes;

  mutable Mutex mutex;
  Peers         peers;

  void rx(const Bytes &data);
};

}  // namespace dmlf
}  // namespace fetch
