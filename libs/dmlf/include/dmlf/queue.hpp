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
#include <queue>

#include "core/mutex.hpp"

#include "dmlf/iqueue.hpp"
#include "dmlf/iupdate.hpp"
#include "dmlf/update.hpp"

namespace fetch {
namespace dmlf {

template <typename T>
class Queue : public IQueue
{
public:
  using UpdateType   = T;
  using Bytes        = IQueue::Bytes;
  using UpdatePtr   = std::shared_ptr<UpdateType>;
  using QueueUpdates = std::priority_queue<UpdatePtr, std::vector<UpdatePtr>, std::greater<UpdatePtr>>;

  Queue()
  {}
  virtual ~Queue()
  {}

  virtual void PushNewMessage(Bytes msg) override 
  {
    auto update = std::make_shared<UpdateType>();
    update->deserialise(msg);
    std::cout << "[Queue] Got update: " << update->TimeStamp() << std::endl;
    {
      Lock l{updates_m_};
      updates_.push(update);
    }
    
  }

  virtual std::size_t size() const override
  {
    Lock l{updates_m_};
    return updates_.size();
  }

  std::shared_ptr<UpdateType> getUpdate()
  {
    Lock l{updates_m_};
    if(!updates_.empty())
    {
      auto upd = updates_.top();
      updates_.pop();
      return upd;
   }
   throw std::length_error{"Updates queue is empty"};
  }
  
private:
  using Mutex            = fetch::Mutex;
  using Lock             = std::unique_lock<Mutex>;
  
  QueueUpdates  updates_;
  mutable Mutex updates_m_; 
  
  Queue(const Queue &other) = delete;
  Queue &operator=(const Queue &other)  = delete;
  bool               operator==(const Queue &other) = delete;
  bool               operator<(const Queue &other)  = delete;
};

}  // namespace dmlf
}  // namespace fetch


