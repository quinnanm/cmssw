#include "L1Trigger/Phase2L1ParticleFlow/interface/AlkaidModel.h"

#include "FWCore/Utilities/interface/Exception.h"

l1tpf::AlkaidModel::AlkaidModel(const std::string& modelPath) {
  interpreter_.load_from_json_file(modelPath);
  if (nInputs() == 0 || nOutputs() == 0)
    throw cms::Exception("AlkaidModel") << "model " << modelPath << " has " << nInputs() << " inputs and " << nOutputs()
                                        << " outputs";
}

void l1tpf::AlkaidModel::evaluate(std::span<const double> inputs,
                                  std::span<double> outputs,
                                  std::vector<int64_t>& scratch) const {
  const size_t nIn = nInputs(), nOut = nOutputs();
  const size_t nSamples = inputs.size() / nIn;
  if (inputs.size() != nSamples * nIn || outputs.size() != nSamples * nOut)
    throw cms::Exception("AlkaidModel") << "evaluate: got " << inputs.size() << " inputs and " << outputs.size()
                                        << " outputs, expected a multiple of " << nIn << " inputs and " << nOut
                                        << " outputs per sample";

  // The interpreter needs nSlots int64 values of scratch per sample evaluated together.
  const size_t scratchSize = interpreter_.get_n_slots() * kBatchSize;
  if (scratch.size() < scratchSize)
    scratch.resize(scratchSize);

  size_t sample = 0;
  for (; sample + kBatchSize <= nSamples; sample += kBatchSize)
    interpreter_.exec_batch<static_cast<int>(kBatchSize)>(
        &inputs[sample * nIn], &outputs[sample * nOut], kBatchSize, scratch.data());
  // The interpreter's batched kernels run over the full batch width even for a partial batch, so the remainder is
  // evaluated one sample at a time (to be revisited with the per-event timings of Phase 3).
  for (; sample < nSamples; ++sample)
    interpreter_.exec_batch<1>(&inputs[sample * nIn], &outputs[sample * nOut], 1, scratch.data());
}
