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

#include "vm_modules/ml/model/model.hpp"

#include "ml/layers/fully_connected.hpp"
#include "ml/model/dnn_classifier.hpp"
#include "ml/model/dnn_regressor.hpp"
#include "ml/model/sequential.hpp"
#include "ml/ops/loss_functions/mean_square_error_loss.hpp"
#include "ml/ops/loss_functions/types.hpp"
#include "vm/module.hpp"
#include "vm_modules/ml/model/model.hpp"
#include "vm_modules/ml/model/model_estimator.hpp"
#include "vm_modules/ml/state_dict.hpp"
#include "vm_modules/use_estimator.hpp"

using namespace fetch::vm;

namespace fetch {
namespace vm_modules {
namespace ml {
namespace model {

using fetch::math::SizeType;
using fetch::ml::ops::LossType;
using fetch::ml::OptimiserType;
using fetch::ml::details::ActivationType;
using VMPtrString = Ptr<String>;

std::map<std::string, SupportedLayerType> const VMModel::layer_types_{
    {"dense", SupportedLayerType::DENSE},
    {"conv1d", SupportedLayerType::CONV1D},
    {"conv2d", SupportedLayerType::CONV2D},
};

std::map<std::string, ActivationType> const VMModel::activations_{
    {"nothing", ActivationType::NOTHING},
    {"leaky_relu", ActivationType::LEAKY_RELU},
    {"log_sigmoid", ActivationType::LOG_SIGMOID},
    {"log_softmax", ActivationType::LOG_SOFTMAX},
    {"relu", ActivationType::RELU},
    {"sigmoid", ActivationType::SIGMOID},
    {"softmax", ActivationType::SOFTMAX},
    {"gelu", ActivationType::GELU},
};

std::map<std::string, LossType> const VMModel::losses_{
    {"mse", LossType::MEAN_SQUARE_ERROR},
    {"cel", LossType::CROSS_ENTROPY},
    {"scel", LossType::SOFTMAX_CROSS_ENTROPY},
};

std::map<std::string, OptimiserType> const VMModel::optimisers_{
    {"adagrad", OptimiserType::ADAGRAD},   {"adam", OptimiserType::ADAM},
    {"momentum", OptimiserType::MOMENTUM}, {"rmsprop", OptimiserType::RMSPROP},
    {"sgd", OptimiserType::SGD},
};

std::map<std::string, uint8_t> const VMModel::model_categories_{
    {"none", static_cast<uint8_t>(ModelCategory::NONE)},
    {"sequential", static_cast<uint8_t>(ModelCategory::SEQUENTIAL)},
    {"regressor", static_cast<uint8_t>(ModelCategory::REGRESSOR)},
    {"classifier", static_cast<uint8_t>(ModelCategory::CLASSIFIER)},
};

/**
 * Converts between user specified string and output type (e.g. activation, layer etc.)
 * invokes VM runtime error if parsing failed.
 * @param name user specified string to convert
 * @param dict dictionary of existing entities
 * @param errmsg preferred display name of expected type, that was not parsed
 */
template <typename T>
inline T VMModel::ParseName(std::string const &name, std::map<std::string, T> const &dict,
                            std::string const &errmsg) const
{
  if (dict.find(name) == dict.end())
  {
    std::string const message{"Unknown " + errmsg + " name : " + name};
    vm_->RuntimeError(message);
    throw std::runtime_error(message);
  }
  return dict.find(name)->second;
}

VMModel::VMModel(VM *vm, TypeId type_id)
  : Object(vm, type_id)
  , estimator_{*this}
{
  Init("none");
}

VMModel::VMModel(VM *vm, TypeId type_id, fetch::vm::Ptr<fetch::vm::String> const &model_category)
  : Object(vm, type_id)
  , estimator_{*this}
{
  Init(model_category->string());
}

VMModel::VMModel(VM *vm, TypeId type_id, std::string const &model_category)
  : Object(vm, type_id)
  , estimator_{*this}
{
  Init(model_category);
}

void VMModel::Init(std::string const &model_category)
{
  uint8_t const parsed_category_num =
      ParseName(model_category, model_categories_, "model category");

  // As far as ParseName succeeded, parsed_category_num is guaranteed to be a valid
  // model category number.
  model_category_ = ModelCategory(parsed_category_num);
  model_config_   = std::make_shared<ModelConfigType>();

  if (model_category_ == ModelCategory::SEQUENTIAL)
  {
    model_ = std::make_shared<fetch::ml::model::Sequential<TensorType>>(*model_config_);
  }
  compiled_ = false;
}

Ptr<VMModel> VMModel::Constructor(VM *vm, TypeId type_id,
                                  fetch::vm::Ptr<fetch::vm::String> const &model_category)
{
  return Ptr<VMModel>{new VMModel(vm, type_id, model_category)};
}

void VMModel::CompileSequential(fetch::vm::Ptr<fetch::vm::String> const &loss,
                                fetch::vm::Ptr<fetch::vm::String> const &optimiser)
{
  LossType const      loss_type      = ParseName(loss->string(), losses_, "loss function");
  OptimiserType const optimiser_type = ParseName(optimiser->string(), optimisers_, "optimiser");

  // Prepare the dataloader
  CompileDataloader();

  try
  {
    model_->Compile(optimiser_type, loss_type);
  }
  catch (std::exception &e)
  {
    vm_->RuntimeError("Compilation of a Sequential model failed : " + std::string(e.what()));
    return;
  }
  compiled_ = true;
}

void VMModel::CompileSimple(fetch::vm::Ptr<fetch::vm::String> const &        optimiser,
                            fetch::vm::Ptr<vm::Array<math::SizeType>> const &in_layers)
{
  OptimiserType const optimiser_type = ParseName(optimiser->string(), optimisers_, "optimiser");
  if (optimiser_type != OptimiserType::ADAM)
  {
    vm_->RuntimeError(R"(Wrong optimiser, a "Simple" model can use only "adam", while given : )" +
                      optimiser->string());
    return;
  }

  size_t const n_elements = in_layers->elements.size();

  std::vector<math::SizeType> layers;
  layers.reserve(n_elements);
  for (size_t i = 0; i < n_elements; ++i)
  {
    layers.emplace_back(in_layers->elements.at(i));
  }

  switch (model_category_)
  {
  case (ModelCategory::REGRESSOR):
    model_ = std::make_shared<fetch::ml::model::DNNRegressor<TensorType>>(*model_config_, layers);
    break;

  case (ModelCategory::CLASSIFIER):
    model_ = std::make_shared<fetch::ml::model::DNNClassifier<TensorType>>(*model_config_, layers);
    break;

  default:
    vm_->RuntimeError("Only REGRESSOR and CLASSIFIER model types take layers on compilation!");
    return;
  }

  // Prepare the dataloader
  CompileDataloader();

  try
  {
    model_->Compile(optimiser_type);
  }
  catch (std::exception &e)
  {
    vm_->RuntimeError("Compilation of a Simple model failed : " + std::string(e.what()));
    return;
  }
  compiled_ = true;
}

void VMModel::Fit(vm::Ptr<VMTensor> const &data, vm::Ptr<VMTensor> const &labels,
                  fetch::math::SizeType const &batch_size)
{
  // prepare dataloader
  auto data_loader = std::make_unique<TensorDataloader>();
  data_loader->SetRandomMode(true);
  data_loader->AddData({data->GetTensor()}, labels->GetTensor());
  model_->SetDataloader(std::move(data_loader));

  // set batch size
  model_config_->batch_size = batch_size;
  model_->UpdateConfig(*model_config_);

  // train for one epoch
  model_->Train();
}

typename VMModel::DataType VMModel::Evaluate()
{
  return (model_->Evaluate(fetch::ml::dataloaders::DataLoaderMode::TRAIN)).at(0);
}

vm::Ptr<VMModel::VMTensor> VMModel::Predict(vm::Ptr<VMTensor> const &data)
{
  vm::Ptr<VMTensor> prediction = this->vm_->CreateNewObject<VMTensor>(data->shape());
  model_->Predict(data->GetTensor(), prediction->GetTensor());
  return prediction;
}

void VMModel::Bind(Module &module)
{
  using StringPtrRef = fetch::vm::Ptr<fetch::vm::String> const &;
  using SizeRef      = math::SizeType const &;
  module.CreateClassType<VMModel>("Model")
      .CreateConstructor(&VMModel::Constructor)
      .CreateSerializeDefaultConstructor([](VM *vm, TypeId type_id) -> Ptr<VMModel> {
        return Ptr<VMModel>{new VMModel(vm, type_id)};
      })
      .CreateMemberFunction("add", &VMModel::AddLayer<SizeRef, SizeRef>,
                            use_estimator(&ModelEstimator::LayerAddDense))
      .CreateMemberFunction("add", &VMModel::AddLayer<SizeRef, SizeRef, SizeRef, SizeRef>,
                            use_estimator(&ModelEstimator::LayerAddConv))
      .CreateMemberFunction("add", &VMModel::AddLayer<SizeRef, SizeRef, StringPtrRef>,
                            use_estimator(&ModelEstimator::LayerAddDenseActivation))
      .CreateMemberFunction("add",
                            &VMModel::AddLayer<SizeRef, SizeRef, SizeRef, SizeRef, StringPtrRef>,
                            use_estimator(&ModelEstimator::LayerAddConvActivation))
      .CreateMemberFunction("compile", &VMModel::CompileSequential,
                            use_estimator(&ModelEstimator::CompileSequential))
      .CreateMemberFunction("compile", &VMModel::CompileSimple,
                            use_estimator(&ModelEstimator::CompileSimple))
      .CreateMemberFunction("fit", &VMModel::Fit, use_estimator(&ModelEstimator::Fit))
      .CreateMemberFunction("evaluate", &VMModel::Evaluate,
                            use_estimator(&ModelEstimator::Evaluate))
      .CreateMemberFunction("predict", &VMModel::Predict, use_estimator(&ModelEstimator::Predict))
      .CreateMemberFunction("evaluate", &VMModel::Evaluate,
                            use_estimator(&ModelEstimator::Evaluate))
      .CreateMemberFunction("predict", &VMModel::Predict, use_estimator(&ModelEstimator::Predict))
      .CreateMemberFunction("serializeToString", &VMModel::SerializeToString,
                            use_estimator(&ModelEstimator::SerializeToString))
      .CreateMemberFunction("deserializeFromString", &VMModel::DeserializeFromString,
                            use_estimator(&ModelEstimator::DeserializeFromString))
      .CreateMemberFunction("add", &VMModel::AddLayer<SizeRef, SizeRef>)
      .CreateMemberFunction("add", &VMModel::AddLayer<SizeRef, SizeRef, SizeRef, SizeRef>)
      .CreateMemberFunction("add", &VMModel::AddLayer<SizeRef, SizeRef, StringPtrRef>)
      .CreateMemberFunction("add",
                            &VMModel::AddLayer<SizeRef, SizeRef, SizeRef, SizeRef, StringPtrRef>)
      .CreateMemberFunction("compile", &VMModel::CompileSequential)
      .CreateMemberFunction("compile", &VMModel::CompileSimple)
      .CreateMemberFunction("fit", &VMModel::Fit)
      .CreateMemberFunction("evaluate", &VMModel::Evaluate)
      .CreateMemberFunction("predict", &VMModel::Predict)
      .CreateMemberFunction("evaluate", &VMModel::Evaluate)
      .CreateMemberFunction("predict", &VMModel::Predict)
      .CreateMemberFunction("serializeToString", &VMModel::SerializeToString)
      .CreateMemberFunction("deserializeFromString", &VMModel::DeserializeFromString);
}

typename VMModel::ModelPtrType &VMModel::GetModel()
{
  return model_;
}

bool VMModel::SerializeTo(serializers::MsgPackSerializer &buffer)
{
  bool success = false;

  // can't serialise uncompiled model
  if (!compiled_)
  {
    vm_->RuntimeError("cannot set state with uncompiled model");
  }
  // can't serialise without a model
  else if (!model_)
  {
    vm_->RuntimeError("cannot set state with model undefined");
  }

  // can't serialise without dataloader ready
  else if (!model_->GetDataloader())
  {
    vm_->RuntimeError("cannot set state with dataloader not set");
  }

  // can't serialise without optimiser ready
  else if (!model_->GetOptimiser())
  {
    vm_->RuntimeError("cannot set state with optimiser not set");
  }

  // should be fine to serialise
  else
  {
    buffer << static_cast<uint8_t>(model_category_);
    buffer << *model_config_;
    buffer << compiled_;
    buffer << *model_;
    estimator_.SerializeTo(buffer);
    success = true;
  }

  return success;
}

bool VMModel::DeserializeFrom(serializers::MsgPackSerializer &buffer)
{
  // deserialise the model category
  uint8_t model_category_int;
  buffer >> model_category_int;

  std::string model_category_name{};
  for (std::pair<std::string, uint8_t> found_category : model_categories_)
  {
    if (found_category.second == model_category_int)
    {
      model_category_name = found_category.first;
    }
  }

  if (model_category_name.empty())
  {
    vm_->RuntimeError("Cannot parse a valid model category from given number : " +
                      std::to_string(model_category_int));
    return false;
  }

  auto const model_category = static_cast<ModelCategory>(model_category_int);

  // deserialise the model config
  ModelConfigType model_config;
  buffer >> model_config;
  model_config_ = std::make_shared<ModelConfigType>(model_config);

  // deserialise the compiled status
  bool compiled = false;
  buffer >> compiled;

  // deserialise the model
  auto model_ptr = std::make_shared<fetch::ml::model::Model<TensorType>>();
  buffer >> (*model_ptr);

  // deserialise the estimator
  estimator_.DeserializeFrom(buffer);

  // assign deserialised model category
  VMModel vm_model(this->vm_, this->type_id_, model_category_name);
  vm_model.model_category_ = model_category;

  // assign deserialised model config
  vm_model.model_config_ = model_config_;

  // assign deserialised model
  vm_model.GetModel() = model_ptr;

  // assign compiled status
  vm_model.compiled_ = compiled;

  // point this object pointer at the deserialised model
  *this = vm_model;

  return true;
}

fetch::vm::Ptr<fetch::vm::String> VMModel::SerializeToString()
{
  serializers::MsgPackSerializer b;
  SerializeTo(b);
  auto byte_array_data = b.data().ToBase64();
  return Ptr<String>{new fetch::vm::String(vm_, static_cast<std::string>(byte_array_data))};
}

fetch::vm::Ptr<VMModel> VMModel::DeserializeFromString(
    fetch::vm::Ptr<fetch::vm::String> const &model_string)
{
  byte_array::ConstByteArray b(model_string->string());
  b = byte_array::FromBase64(b);
  MsgPackSerializer buffer(b);
  DeserializeFrom(buffer);

  auto vm_model        = fetch::vm::Ptr<VMModel>(new VMModel(vm_, type_id_));
  vm_model->GetModel() = model_;

  return vm_model;
}

VMModel::ModelEstimator &VMModel::Estimator()
{
  return estimator_;
}

void VMModel::AssertLayerTypeMatches(SupportedLayerType                layer,
                                     std::vector<SupportedLayerType> &&valids) const
{
  static const std::map<SupportedLayerType, std::string> LAYER_NAMES_{
      {SupportedLayerType::DENSE, "dense"},
      {SupportedLayerType::CONV1D, "conv1d"},
      {SupportedLayerType::CONV2D, "conv2d"},
  };
  if (std::find(valids.begin(), valids.end(), layer) == valids.end())
  {
    std::string const message{"Invalid params specified for \"" + LAYER_NAMES_.at(layer) +
                              "\" layer."};
    vm_->RuntimeError(message);
    throw std::runtime_error(message);
  }
}

VMModel::SequentialModelPtr VMModel::GetMeAsSequentialIfPossible()
{
  if (model_category_ != ModelCategory::SEQUENTIAL)
  {
    std::string const message{"No \"add\" method exists for non-sequential models!"};
    vm_->RuntimeError(message);
    throw std::runtime_error(message);
  }
  return std::dynamic_pointer_cast<fetch::ml::model::Sequential<TensorType>>(model_);
}

template <typename... LayerArgs>
void VMModel::AddLayer(fetch::vm::Ptr<fetch::vm::String> const &layer, LayerArgs... args)
{
  auto const layer_type = ParseName(layer->string(), layer_types_, "layer type");
  AddLayerSpecificImpl(layer_type, args...);
}

void VMModel::AddLayerSpecificImpl(SupportedLayerType layer, math::SizeType const &inputs,
                                   math::SizeType const &hidden_nodes)
{
  AddLayerSpecificImpl(layer, inputs, hidden_nodes, ActivationType::NOTHING);
}

void VMModel::AddLayerSpecificImpl(SupportedLayerType layer, math::SizeType const &inputs,
                                   math::SizeType const &                   hidden_nodes,
                                   fetch::vm::Ptr<fetch::vm::String> const &activation)
{
  AddLayerSpecificImpl(layer, inputs, hidden_nodes,
                       ParseName(activation->string(), activations_, "activation function"));
}

void VMModel::AddLayerSpecificImpl(SupportedLayerType layer, math::SizeType const &inputs,
                                   math::SizeType const &             hidden_nodes,
                                   fetch::ml::details::ActivationType activation)
{
  auto me = GetMeAsSequentialIfPossible();
  AssertLayerTypeMatches(layer, {SupportedLayerType::DENSE});
  compiled_ = false;
  me->Add<fetch::ml::layers::FullyConnected<TensorType>>(inputs, hidden_nodes, activation);
}

void VMModel::AddLayerSpecificImpl(SupportedLayerType layer, math::SizeType const &output_channels,
                                   math::SizeType const &input_channels,
                                   math::SizeType const &kernel_size,
                                   math::SizeType const &stride_size)
{
  AddLayerSpecificImpl(layer, output_channels, input_channels, kernel_size, stride_size,
                       ActivationType::NOTHING);
}

void VMModel::AddLayerSpecificImpl(SupportedLayerType layer, math::SizeType const &output_channels,
                                   math::SizeType const &                   input_channels,
                                   math::SizeType const &                   kernel_size,
                                   math::SizeType const &                   stride_size,
                                   fetch::vm::Ptr<fetch::vm::String> const &activation)
{
  AddLayerSpecificImpl(layer, output_channels, input_channels, kernel_size, stride_size,
                       ParseName(activation->string(), activations_, "activation function"));
}

void VMModel::AddLayerSpecificImpl(SupportedLayerType layer, math::SizeType const &output_channels,
                                   math::SizeType const &             input_channels,
                                   math::SizeType const &             kernel_size,
                                   math::SizeType const &             stride_size,
                                   fetch::ml::details::ActivationType activation)
{
  auto me = GetMeAsSequentialIfPossible();
  AssertLayerTypeMatches(layer, {SupportedLayerType::CONV1D, SupportedLayerType::CONV2D});
  compiled_ = false;
  if (layer == SupportedLayerType::CONV1D)
  {
    me->Add<fetch::ml::layers::Convolution1D<TensorType>>(output_channels, input_channels,
                                                          kernel_size, stride_size, activation);
  }
  else if (layer == SupportedLayerType::CONV2D)
  {
    me->Add<fetch::ml::layers::Convolution2D<TensorType>>(output_channels, input_channels,
                                                          kernel_size, stride_size, activation);
  }
}

/**
 * for regressor and classifier we can't prepare the dataloder until after compile has begun
 * because model_ isn't ready until then.
 */
void VMModel::CompileDataloader()
{
  // set up the dataloader
  auto data_loader = std::make_unique<TensorDataloader>();
  data_loader->SetRandomMode(true);
  model_->SetDataloader(std::move(data_loader));
}
}  // namespace model
}  // namespace ml
}  // namespace vm_modules
}  // namespace fetch
