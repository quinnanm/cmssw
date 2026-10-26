#include "L1Trigger/Phase2L1ParticleFlow/interface/EmulatorBackend.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

void l1tpf::addEmulatorBackendParameters(edm::ParameterSetDescription& desc, const std::string& defaultAlkaidModel) {
  desc.add<std::string>("backend", "hls4ml")->setComment("NN emulator: 'hls4ml' (compiled model) or 'alkaid'");
  desc.add<std::string>("alkaidModel", defaultAlkaidModel)
      ->setComment("alkaid ALIR model (path relative to the CMSSW search path), used if backend = 'alkaid'");
}

std::shared_ptr<const l1tpf::AlkaidModel> l1tpf::configuredAlkaidModel(const edm::ParameterSet& cfg,
                                                                       size_t nInputs,
                                                                       size_t nOutputs) {
  const std::string backend = cfg.getParameter<std::string>("backend");
  if (backend == "hls4ml")
    return nullptr;
  if (backend != "alkaid")
    throw cms::Exception("Configuration") << "unknown backend '" << backend << "', expected 'hls4ml' or 'alkaid'";
  const std::string path = edm::FileInPath(cfg.getParameter<std::string>("alkaidModel")).fullPath();
  auto model = std::make_shared<const AlkaidModel>(path);
  if (model->nInputs() != nInputs || model->nOutputs() != nOutputs)
    throw cms::Exception("Configuration")
        << "alkaid model " << path << " has " << model->nInputs() << " inputs and " << model->nOutputs()
        << " outputs, the producer needs " << nInputs << " and " << nOutputs;
  return model;
}

std::unique_ptr<l1tpf::SharedAlkaidModel> l1tpf::loadSharedAlkaidModel(const edm::ParameterSet& cfg,
                                                                       size_t nInputs,
                                                                       size_t nOutputs) {
  return std::make_unique<SharedAlkaidModel>(SharedAlkaidModel{configuredAlkaidModel(cfg, nInputs, nOutputs)});
}
