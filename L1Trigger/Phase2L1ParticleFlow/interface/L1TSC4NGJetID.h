#ifndef L1TRIGGER_PHASE2L1PARTICLEFLOWS_L1TSC4NGJetID_H
#define L1TRIGGER_PHASE2L1PARTICLEFLOWS_L1TSC4NGJetID_H

#include <string>
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TParticleFlow/interface/datatypes.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/jetmet/L1SeedConePFJetEmulator.h"

// fixed-point NN emulation (alkaid)
#include <array>
#include "ap_fixed.h"
#include "PhysicsTools/Alkaid/interface/Model.h"

namespace L1TSC4NGJet {

  typedef ap_fixed<64, 32, AP_RND, AP_SAT, 0> inputtype;
  typedef ap_ufixed<14, 0, AP_TRN, AP_SAT> inv_pt_t;
  typedef ap_ufixed<14, 4, AP_TRN, AP_SAT> log_pt_t;

  template <class t>
  t candidate_mass(l1ct::PuppiObj puppicand) {
    // Define lookup table
    static const t PION_MASS = t(0.13);
    static const t PHOTON_MASS = t(0.0);
    static const t ELECTRON_MASS = t(0.005);
    static const t MUON_MASS = t(0.105);
    static const t K_MASS = t(0.5);

    // Default to pion mass
    t massCand = PION_MASS;

    if (puppicand.hwId.bits == l1ct::ParticleID::PHOTON) {
      massCand = PHOTON_MASS;
    } else if (puppicand.hwId.bits == l1ct::ParticleID::ELEPLUS || puppicand.hwId.bits == l1ct::ParticleID::ELEMINUS) {
      massCand = ELECTRON_MASS;
    } else if (puppicand.hwId.bits == l1ct::ParticleID::MUMINUS || puppicand.hwId.bits == l1ct::ParticleID::MUPLUS) {
      massCand = MUON_MASS;
    } else if (puppicand.hwId.bits == l1ct::ParticleID::HADZERO) {
      massCand = K_MASS;
    }

    return massCand;
  }

}  // namespace L1TSC4NGJet
class L1TSC4NGJetID {
public:
  // model: fixed-point model (alkaid) with kModelInputs inputs and kModelOutputs outputs; must outlive this object.
  // iNParticles must not exceed N_candidates (fewer particles are zero-padded).
  L1TSC4NGJetID(const cms::alkaid::Model& model, int iNParticles, bool debug);

  static constexpr int N_candidates = 16;
  static constexpr int N_candidate_features = 21;
  static constexpr int N_candidate_inputs = N_candidates * N_candidate_features;
  static constexpr int N_jet_inputs = 2;
  static constexpr int N_class_outputs = 8;
  static constexpr int N_regression_outputs = 1;
  // Flat model I/O (L1TSC4NGJetModel_v2_0_0): inputs [N_candidate_inputs candidate features, N_jet_inputs jet
  // features], outputs [N_regression_outputs regression score, kModelClassOutputs class scores]. The model scores 9
  // classes; only the first N_class_outputs are used.
  static constexpr int kModelClassOutputs = 9;
  static constexpr int kModelInputs = N_candidate_inputs + N_jet_inputs;
  static constexpr int kModelOutputs = N_regression_outputs + kModelClassOutputs;

  // Intermediate output type to allow full precision multiplication of jet pt by the ratio
  typedef ap_ufixed<22, 12, AP_TRN, AP_SAT> output_regression_type;
  // Intermediate output type for classification score to be loaded into jet word
  typedef std::array<l1ct::jet_tag_score_t, N_class_outputs> output_class_type;
  typedef std::pair<std::array<output_regression_type, N_regression_outputs>, output_class_type> outputpairtype;

  void setVectors();
  outputpairtype EvaluateNNFixed();
  outputpairtype computeFixed(const l1t::PFJet& iJet);

private:
  std::vector<L1TSC4NGJet::inputtype> candidate_vector_;
  std::vector<L1TSC4NGJet::inputtype> jet_vector_;

  int fNParticles_;

  std::unique_ptr<L1TSC4NGJet::inputtype[]> fPt_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fPt_rel_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fDEta_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fDPhi_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fPt_log_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fMass_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fZ0_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fDxy_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fIs_filled_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fPuppi_weight_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fEmID_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fQuality_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fEta_;

  std::unique_ptr<L1TSC4NGJet::inputtype[]> fCharge_;
  std::unique_ptr<L1TSC4NGJet::inputtype[]> fId_;

  L1TSC4NGJet::inputtype fJetEta_;
  L1TSC4NGJet::inputtype fJetPtLog_;

  const cms::alkaid::Model& nnModel_;
  cms::alkaid::Model::Scratch nnScratch_;
  std::array<double, kModelInputs> nnInput_{};
  std::array<double, kModelOutputs> nnOutput_{};

  bool isDebugEnabled_;
};
#endif
