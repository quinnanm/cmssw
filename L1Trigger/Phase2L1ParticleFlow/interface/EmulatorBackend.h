#ifndef L1Trigger_Phase2L1ParticleFlow_EmulatorBackend_h
#define L1Trigger_Phase2L1ParticleFlow_EmulatorBackend_h

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "hls4ml/emulator.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/AlkaidEmulatorModel.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/AlkaidModel.h"

namespace l1tpf {

  // NN emulator back-end selection for producers that support both hls4ml and alkaid. Parameters:
  //   backend      "hls4ml" (the compiled model the producer's ModelLoader names) or "alkaid"
  //   alkaidModel  ALIR file for backend = "alkaid", relative to the CMSSW search path. A string rather than a
  //                FileInPath, so that hls4ml jobs don't need the file to exist.
  void addEmulatorBackendParameters(edm::ParameterSetDescription& desc, const std::string& defaultAlkaidModel);

  // The alkaid model for backend = "alkaid", checked to have nInputs inputs and nOutputs outputs (the sizes the
  // producer passes); nullptr for backend = "hls4ml". Throws for any other backend.
  std::shared_ptr<const AlkaidModel> configuredAlkaidModel(const edm::ParameterSet& cfg,
                                                           size_t nInputs,
                                                           size_t nOutputs);

  // One alkaid model per producer module, shared by all its streams (edm::GlobalCache): loaded once per job and
  // read-only afterwards (AlkaidModel::evaluate is const; each stream has its own buffers). Empty for backend = "hls4ml",
  // whose models are stateful and stay one per stream.
  struct SharedAlkaidModel {
    std::shared_ptr<const AlkaidModel> model;
  };

  // For the producers' initializeGlobalCache: configuredAlkaidModel wrapped for the global cache.
  std::unique_ptr<SharedAlkaidModel> loadSharedAlkaidModel(const edm::ParameterSet& cfg,
                                                           size_t nInputs,
                                                           size_t nOutputs);

  // The emulator of one stream, for producers that pass plain arrays (InputType*, OutputType*) to the model: an adapter
  // on the shared alkaid model, or the stream's own hls4ml model (loader must outlive it: it owns the library handle).
  template <typename InputType, typename OutputType>
  std::shared_ptr<hls4mlEmulator::Model> makeEmulatorModel(const SharedAlkaidModel& shared,
                                                           hls4mlEmulator::ModelLoader& loader) {
    if (!shared.model)
      return loader.load_model();
    return std::make_shared<AlkaidEmulatorModel<InputType, OutputType>>(shared.model);
  }

}  // namespace l1tpf

#endif
