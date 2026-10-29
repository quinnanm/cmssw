#include "L1Trigger/Phase2L1ParticleFlow/interface/JetId.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cmath>

JetId::JetId(const std::string &iInput, const std::string &iOutput, const cms::alkaid::Model &model, int iNParticles)
    : nnModel_(&model) {
  if (iNParticles < 0 || static_cast<unsigned int>(iNParticles) * kNNFeaturesPerParticle > kNNInputs)
    throw cms::Exception("Configuration") << "JetId: nParticles = " << iNParticles << ", the NN takes at most "
                                          << kNNInputs / kNNFeaturesPerParticle << " particles";
  NNvectorVar_.clear();
  fNParticles_ = iNParticles;

  fPt_ = std::make_unique<float[]>(fNParticles_);
  fEta_ = std::make_unique<float[]>(fNParticles_);
  fPhi_ = std::make_unique<float[]>(fNParticles_);
  fId_ = std::make_unique<int[]>(fNParticles_);
  fCharge_ = std::make_unique<int[]>(fNParticles_);
  fDZ_ = std::make_unique<float[]>(fNParticles_);
  fDX_ = std::make_unique<float[]>(fNParticles_);
  fDY_ = std::make_unique<float[]>(fNParticles_);
  fInput_ = iInput;
  fOutput_ = iOutput;
}

//--BJet algo specific constructor
JetId::JetId(const std::string &iInput, const std::string &iOutput, const BJetTFCache *cache, int iNParticles)
    : sessionRef_(cache->session) {
  NNvectorVar_.clear();
  fNParticles_ = iNParticles;

  fPt_ = std::make_unique<float[]>(fNParticles_);
  fEta_ = std::make_unique<float[]>(fNParticles_);
  fPhi_ = std::make_unique<float[]>(fNParticles_);
  fId_ = std::make_unique<int[]>(fNParticles_);
  fCharge_ = std::make_unique<int[]>(fNParticles_);
  fDZ_ = std::make_unique<float[]>(fNParticles_);
  fDX_ = std::make_unique<float[]>(fNParticles_);
  fDY_ = std::make_unique<float[]>(fNParticles_);
  fInput_ = iInput;
  fOutput_ = iOutput;
}

void JetId::setNNVectorVar() {
  NNvectorVar_.clear();
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    if (fPt_.get()[i0] == 0) {
      for (int i1 = 0; i1 < 13; i1++)
        NNvectorVar_.push_back(0);
      continue;
    }
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Electron && fCharge_.get()[i0] < 0);       // Electron
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Electron && fCharge_.get()[i0] > 0);       // Positron
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Muon && fCharge_.get()[i0] < 0);           // Muon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Muon && fCharge_.get()[i0] > 0);           // Anti-Muon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::Photon);                                   // Photon
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::NeutralHadron);                            // Neutral Had
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::ChargedHadron && fCharge_.get()[i0] < 0);  // Pion
    NNvectorVar_.push_back(fId_.get()[i0] == l1t::PFCandidate::ChargedHadron && fCharge_.get()[i0] > 0);  // Anti-Pion
    NNvectorVar_.push_back(fDZ_.get()[i0]);                                                               //dZ
    NNvectorVar_.push_back(std::hypot(fDX_.get()[i0], fDY_.get()[i0]));                                   //d0
    NNvectorVar_.push_back(fPt_.get()[i0]);   //pT as a fraction of jet pT
    NNvectorVar_.push_back(fEta_.get()[i0]);  //dEta from jet axis
    NNvectorVar_.push_back(fPhi_.get()[i0]);  //dPhi from jet axis
  }
}
float JetId::EvaluateNN() {
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, (unsigned int)NNvectorVar_.size(), 1});
  for (unsigned int i = 0; i < NNvectorVar_.size(); i++) {
    input.tensor<float, 3>()(0, i, 0) = float(NNvectorVar_[i]);
  }
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::run(sessionRef_, {{fInput_, input}}, {fOutput_}, &outputs);
  return outputs[0].matrix<float>()(0, 0);
}  //end EvaluateNN

JetId::NNOutputType JetId::EvaluateNNFixed() {
  // Quantize the features to the model's input type (as for the hls4ml emulator), zero-padded to kNNInputs.
  // NNInputType -> double is exact.
  nnInput_.fill(0.);
  for (unsigned int i = 0; i < NNvectorVar_.size() && i < kNNInputs; i++) {
    nnInput_[i] = NNInputType(NNvectorVar_[i]).to_double();
  }
  nnModel_->evaluate(nnInput_, nnOutput_, nnScratch_);
  return NNOutputType(nnOutput_[0]);  // exact: the model's output has this type
}  //end EvaluateNNFixed

float JetId::compute(const l1t::PFJet &iJet, float vz, bool useRawPt) {
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    fPt_.get()[i0] = 0;
    fEta_.get()[i0] = 0;
    fPhi_.get()[i0] = 0;
    fId_.get()[i0] = 0;
    fCharge_.get()[i0] = 0;
    fDZ_.get()[i0] = 0;
    fDX_.get()[i0] = 0;
    fDY_.get()[i0] = 0;
  }
  auto iParts = iJet.constituents();
  std::sort(iParts.begin(), iParts.end(), [](edm::Ptr<l1t::PFCandidate> i, edm::Ptr<l1t::PFCandidate> j) {
    return (i->pt() > j->pt());
  });
  float jetpt = useRawPt ? iJet.rawPt() : iJet.pt();
  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 >= (unsigned int)fNParticles_)
      break;
    fPt_.get()[i0] = iParts[i0]->pt() / jetpt;
    fEta_.get()[i0] = iParts[i0]->eta() - iJet.eta();
    fPhi_.get()[i0] = deltaPhi(iParts[i0]->phi(), iJet.phi());
    fId_.get()[i0] = iParts[i0]->id();
    fCharge_.get()[i0] = iParts[i0]->charge();
    if (iParts[i0]->pfTrack().isNonnull()) {
      fDX_.get()[i0] = iParts[i0]->pfTrack()->vx();
      fDY_.get()[i0] = iParts[i0]->pfTrack()->vy();
      fDZ_.get()[i0] = iParts[i0]->pfTrack()->vz() - vz;
    }
  }
  setNNVectorVar();
  return EvaluateNN();
}

JetId::NNOutputType JetId::computeFixed(const l1t::PFJet &iJet, float vz, bool useRawPt) {
  for (int i0 = 0; i0 < fNParticles_; i0++) {
    fPt_.get()[i0] = 0;
    fEta_.get()[i0] = 0;
    fPhi_.get()[i0] = 0;
    fId_.get()[i0] = 0;
    fCharge_.get()[i0] = 0;
    fDZ_.get()[i0] = 0;
    fDX_.get()[i0] = 0;
    fDY_.get()[i0] = 0;
  }
  auto iParts = iJet.constituents();
  std::sort(iParts.begin(), iParts.end(), [](edm::Ptr<l1t::PFCandidate> i, edm::Ptr<l1t::PFCandidate> j) {
    return (i->pt() > j->pt());
  });
  float jetpt = useRawPt ? iJet.rawPt() : iJet.pt();
  for (unsigned int i0 = 0; i0 < iParts.size(); i0++) {
    if (i0 >= (unsigned int)fNParticles_)
      break;
    fPt_.get()[i0] = iParts[i0]->pt() / jetpt;
    fEta_.get()[i0] = iParts[i0]->eta() - iJet.eta();
    fPhi_.get()[i0] = deltaPhi(iParts[i0]->phi(), iJet.phi());
    fId_.get()[i0] = iParts[i0]->id();
    fCharge_.get()[i0] = iParts[i0]->charge();
    if (iParts[i0]->pfTrack().isNonnull()) {
      fDX_.get()[i0] = iParts[i0]->pfTrack()->vx();
      fDY_.get()[i0] = iParts[i0]->pfTrack()->vy();
      fDZ_.get()[i0] = iParts[i0]->pfTrack()->vz() - vz;
    }
  }
  setNNVectorVar();

  return EvaluateNNFixed();
}
