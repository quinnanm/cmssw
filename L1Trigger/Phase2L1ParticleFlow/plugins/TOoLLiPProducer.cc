#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/JetId.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include <cmath>
#include <vector>

#include <string>
#include "ap_fixed.h"
#include "PhysicsTools/Alkaid/interface/Model.h"

using namespace l1t;

// The NN model is loaded once per job and shared read-only by all streams (edm::GlobalCache).
class TOoLLiPProducer : public edm::stream::EDProducer<edm::GlobalCache<cms::alkaid::Model>> {
public:
  TOoLLiPProducer(const edm::ParameterSet&, const cms::alkaid::Model* model);
  ~TOoLLiPProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::unique_ptr<cms::alkaid::Model> initializeGlobalCache(const edm::ParameterSet& cfg);
  static void globalEndJob(const cms::alkaid::Model*) {}

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edm::View<l1t::PFJet>> const jets_;
  bool const fUseRawPt_;
  double const fMinPt_;
  double const fMaxEta_;
  unsigned int const fMaxJets_;
  int const fNParticles_;
  edm::EDGetTokenT<std::vector<l1t::VertexWord>> const fVtxEmu_;

  std::unique_ptr<JetId> fJetId_;
};

std::unique_ptr<cms::alkaid::Model> TOoLLiPProducer::initializeGlobalCache(const edm::ParameterSet& cfg) {
  return std::make_unique<cms::alkaid::Model>(
      cfg.getParameter<edm::FileInPath>("model").fullPath(), JetId::kNNInputs, JetId::kNNOutputs);
}

TOoLLiPProducer::TOoLLiPProducer(const edm::ParameterSet& cfg, const cms::alkaid::Model* model)
    : jets_(consumes<edm::View<l1t::PFJet>>(cfg.getParameter<edm::InputTag>("jets"))),
      fUseRawPt_(cfg.getParameter<bool>("useRawPt")),
      fMinPt_(cfg.getParameter<double>("minPt")),
      fMaxEta_(cfg.getParameter<double>("maxEta")),
      fMaxJets_(cfg.getParameter<int>("maxJets")),
      fNParticles_(cfg.getParameter<int>("nParticles")),
      fVtxEmu_(consumes<std::vector<l1t::VertexWord>>(cfg.getParameter<edm::InputTag>("vtx"))),
      fJetId_(std::make_unique<JetId>(
          cfg.getParameter<std::string>("NNInput"), cfg.getParameter<std::string>("NNOutput"), *model, fNParticles_)) {
  produces<edm::ValueMap<float>>("L1PFLLPJets");
}

void TOoLLiPProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<l1t::PFJet>> jets;
  iEvent.getByToken(jets_, jets);
  float vz = 0.;
  double ptsum = 0;
  edm::Handle<std::vector<l1t::VertexWord>> vtxEmuHandle;
  iEvent.getByToken(fVtxEmu_, vtxEmuHandle);
  for (const auto& vtx : *vtxEmuHandle) {
    if (ptsum == 0 || vtx.pt() > ptsum) {
      ptsum = vtx.pt();
      vz = vtx.z0();
    }
  }

  std::vector<float> LLPScores;
  for (const auto& srcjet : *jets) {
    if (((fUseRawPt_ ? srcjet.rawPt() : srcjet.pt()) < fMinPt_) || std::abs(srcjet.eta()) > fMaxEta_ ||
        LLPScores.size() >= fMaxJets_) {
      LLPScores.push_back(-999.);
      continue;
    }
    JetId::NNOutputType LLPScore = fJetId_->computeFixed(srcjet, vz, fUseRawPt_);
    LLPScores.push_back(LLPScore);
  }

  auto outT = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler fillerT(*outT);
  fillerT.insert(jets, LLPScores.begin(), LLPScores.end());
  fillerT.fill();

  iEvent.put(std::move(outT), "L1PFLLPJets");
}

void TOoLLiPProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("scPFL1Puppi"));
  desc.add<bool>("useRawPt", true);
  desc.add<edm::FileInPath>("model",
                            edm::FileInPath("L1Trigger/Phase2L1ParticleFlow/data/alkaid/TOoLLiP_v3_v3_1_0.json.gz"))
      ->setComment("TOoLLiP model (alkaid ALIR file); TOoLLiP_v3 from cms-hls4ml/TOoLLiP v3.1.0");

  desc.add<std::string>("NNInput", "input:0");
  desc.add<std::string>("NNOutput", "sequential/dense_2/Sigmoid");
  desc.add<int>("maxJets", 10);
  desc.add<int>("nParticles", 10);
  desc.add<double>("minPt", 20);
  desc.add<double>("maxEta", 2.4);
  desc.add<edm::InputTag>("vtx", edm::InputTag("L1VertexFinderEmulator", "L1VerticesEmulation"));
  descriptions.add("TOoLLiPProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TOoLLiPProducer);
