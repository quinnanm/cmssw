#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_JETID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_JETID_H

#include <array>
#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"

// fixed-point NN emulation (alkaid)
#include "ap_fixed.h"
#include "PhysicsTools/Alkaid/interface/Model.h"

struct BJetTFCache {
  BJetTFCache(const std::string &graphPath) : graphDef(tensorflow::loadGraphDef(graphPath)) {
    session = tensorflow::createSession(graphDef.get());
  }
  ~BJetTFCache() { tensorflow::closeSession(session); }
  std::unique_ptr<tensorflow::GraphDef> graphDef;
  tensorflow::Session *session;
};

class JetId {
public:
  // Fixed-point I/O of the TOoLLiP NN (EvaluateNNFixed): the model's input_t/result_t. The inputs are quantized to
  // NNInputType before the model sees them, exactly as for the hls4ml emulator this replaces.
  using NNInputType = ap_fixed<12, 6, AP_TRN, AP_SAT, 0>;
  using NNOutputType = ap_fixed<14, 8, AP_TRN, AP_SAT, 0>;
  static constexpr unsigned int kNNFeaturesPerParticle = 13;              // as filled by setNNVectorVar()
  static constexpr unsigned int kNNInputs = 10 * kNNFeaturesPerParticle;  // 10 particles
  static constexpr unsigned int kNNOutputs = 1;

  // TOoLLiP: fixed-point model (alkaid) with kNNInputs inputs and kNNOutputs outputs; must outlive this JetId.
  // iNParticles * kNNFeaturesPerParticle must not exceed kNNInputs (fewer particles are zero-padded).
  JetId(const std::string &iInput, const std::string &iOutput, const cms::alkaid::Model &model, int iNParticles);
  JetId(const std::string &iInput, const std::string &iOutput, const BJetTFCache *cache, int iNParticles);
  ~JetId() = default;

  void setNNVectorVar();
  float EvaluateNN();
  NNOutputType EvaluateNNFixed();
  float compute(const l1t::PFJet &iJet, float vz, bool useRawPt);
  NNOutputType computeFixed(const l1t::PFJet &iJet, float vz, bool useRawPt);

private:
  std::vector<float> NNvectorVar_;
  std::string fInput_;
  std::string fOutput_;
  int fNParticles_;
  unique_ptr<float[]> fPt_;
  unique_ptr<float[]> fEta_;
  unique_ptr<float[]> fPhi_;
  unique_ptr<int[]> fId_;
  unique_ptr<int[]> fCharge_;
  unique_ptr<float[]> fDZ_;
  unique_ptr<float[]> fDX_;
  unique_ptr<float[]> fDY_;
  tensorflow::Session *sessionRef_;
  const cms::alkaid::Model *nnModel_ = nullptr;
  cms::alkaid::Model::Scratch nnScratch_;
  std::array<double, kNNInputs> nnInput_{};
  std::array<double, kNNOutputs> nnOutput_{};
};
#endif
