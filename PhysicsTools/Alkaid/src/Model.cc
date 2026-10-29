#include "PhysicsTools/Alkaid/interface/Model.h"

#include <exception>

#include "alkaid/alir/ALIRInterpreter.hh"
#include "FWCore/Utilities/interface/Exception.h"

struct cms::alkaid::Model::Impl {
  alir::ALIRInterpreter interpreter;
};

cms::alkaid::Model::Model(const std::string& fullPath, size_t nInputs, size_t nOutputs)
    : path_(fullPath),
      impl_(load(fullPath)),
      nInputs_(impl_->interpreter.get_n_in()),
      nOutputs_(impl_->interpreter.get_n_out()),
      nSlots_(impl_->interpreter.get_n_slots()) {
  if (nInputs_ != nInputs || nOutputs_ != nOutputs || nInputs_ == 0 || nOutputs_ == 0)
    throw cms::Exception("AlkaidModel") << "alkaid model " << fullPath << " has " << nInputs_ << " inputs and "
                                        << nOutputs_ << " outputs, the caller needs " << nInputs << " and " << nOutputs
                                        << " (both > 0)";
}

cms::alkaid::Model::~Model() = default;

void cms::alkaid::Model::evaluate(std::span<const double> inputs, std::span<double> outputs, Scratch& scratch) const {
  const size_t nSamples = inputs.size() / nInputs_;
  if (inputs.size() != nSamples * nInputs_ || outputs.size() != nSamples * nOutputs_)
    throw cms::Exception("AlkaidModel") << "evaluate (" << path_ << "): got " << inputs.size() << " inputs and "
                                        << outputs.size() << " outputs, expected a multiple of " << nInputs_
                                        << " inputs and " << nOutputs_ << " outputs per sample, for the same number "
                                        << "of samples";

  // The interpreter needs nSlots int64 values of scratch per sample evaluated together.
  if (scratch.buffer.size() < nSlots_ * kBatchSize)
    scratch.buffer.resize(nSlots_ * kBatchSize);

  const alir::ALIRInterpreter& interpreter = impl_->interpreter;
  size_t sample = 0;
  for (; sample + kBatchSize <= nSamples; sample += kBatchSize)
    interpreter.exec_batch<static_cast<int>(kBatchSize)>(
        &inputs[sample * nInputs_], &outputs[sample * nOutputs_], kBatchSize, scratch.buffer.data());
  // The batched kernels always run over the full batch width, so a partial batch is evaluated one sample at a time.
  for (; sample < nSamples; ++sample)
    interpreter.exec_batch<1>(&inputs[sample * nInputs_], &outputs[sample * nOutputs_], 1, scratch.buffer.data());
}

std::unique_ptr<const cms::alkaid::Model::Impl> cms::alkaid::Model::load(const std::string& fullPath) {
  auto impl = std::make_unique<Impl>();
  try {
    impl->interpreter.load_from_json_file(fullPath);
  } catch (const std::exception& e) {
    // The runtime throws std::runtime_error (missing or corrupt file, unsupported ALIR spec version, ...)
    throw cms::Exception("AlkaidModel") << "cannot load alkaid model " << fullPath << ": " << e.what();
  }
  return impl;
}
