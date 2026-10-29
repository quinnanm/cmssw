#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC4NGJetID.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/L1Trigger/interface/VertexWord.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/corrector.h"
#include "PhysicsTools/Alkaid/interface/Model.h"

#include <cmath>
#include <vector>

// The NN model is loaded once per job and shared read-only by all streams (edm::GlobalCache).
class L1TSC4NGJetProducer : public edm::stream::EDProducer<edm::GlobalCache<cms::alkaid::Model>> {
public:
  L1TSC4NGJetProducer(const edm::ParameterSet&, const cms::alkaid::Model* model);
  ~L1TSC4NGJetProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  static std::unique_ptr<cms::alkaid::Model> initializeGlobalCache(const edm::ParameterSet& cfg);
  static void globalEndJob(const cms::alkaid::Model*) {}

private:
  std::unique_ptr<L1TSC4NGJetID> fJetId_;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<edm::View<l1t::PFJet>> const jets_;
  const bool doJEC;
  const bool returnRawPt_;
  double const fMinPt_;
  double const fMaxEta_;
  unsigned int const fMaxJets_;
  int const fNParticles_;
  bool const isDebugEnabled = false;

  std::vector<l1ct::JetTagClass> classes_;

  std::optional<l1tpf::corrector> corrector;
};

std::unique_ptr<cms::alkaid::Model> L1TSC4NGJetProducer::initializeGlobalCache(const edm::ParameterSet& cfg) {
  return std::make_unique<cms::alkaid::Model>(
      cfg.getParameter<edm::FileInPath>("model").fullPath(), L1TSC4NGJetID::kModelInputs, L1TSC4NGJetID::kModelOutputs);
}

L1TSC4NGJetProducer::L1TSC4NGJetProducer(const edm::ParameterSet& cfg, const cms::alkaid::Model* model)
    : jets_(consumes<edm::View<l1t::PFJet>>(cfg.getParameter<edm::InputTag>("jets"))),
      doJEC(cfg.getParameter<bool>("doJEC")),
      returnRawPt_(cfg.getParameter<bool>("returnRawPt")),
      fMinPt_(cfg.getParameter<double>("minPt")),
      fMaxEta_(cfg.getParameter<double>("maxEta")),
      fMaxJets_(cfg.getParameter<int>("maxJets")),
      fNParticles_(cfg.getParameter<int>("nParticles")),
      isDebugEnabled(edm::isDebugEnabled()) {
  std::vector<std::string> classes = cfg.getParameter<std::vector<std::string>>("classes");
  for (unsigned i = 0; i < classes.size(); i++) {
    classes_.push_back(l1ct::JetTagClass(classes[i]));
  }
  fJetId_ = std::make_unique<L1TSC4NGJetID>(*model, fNParticles_, isDebugEnabled);
  produces<l1t::PFJetCollection>("l1tSC4NGJets");
  if (doJEC) {
    corrector = l1tpf::corrector(cfg.getParameter<std::string>("correctorFile"),
                                 cfg.getParameter<std::string>("correctorDir"),
                                 -1.,
                                 isDebugEnabled,
                                 true);
  }
}

void L1TSC4NGJetProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<l1t::PFJet>> jets;
  iEvent.getByToken(jets_, jets);
  std::vector<l1t::PFJet> taggedJets;

  for (const auto& srcjet : *jets) {
    l1ct::Jet ctHWTaggedJet = l1ct::Jet::unpack(srcjet.encodedJet(l1t::PFJet::HWEncoding::CT));
    if (taggedJets.size() >= fMaxJets_) {
      ctHWTaggedJet.clear();
      continue;
    }
    L1TSC4NGJetID::outputpairtype JetModel_output = fJetId_->computeFixed(srcjet);
    std::vector<float> JetScore_float;
    for (unsigned i = 0; i < classes_.size(); i++) {
      ctHWTaggedJet.hwTagScores[i] = JetModel_output.second[i];
      JetScore_float.push_back((float)JetModel_output.second[i]);
    }
    L1TSC4NGJetID::output_regression_type PtCorrection_ = JetModel_output.first[0];
    L1TSC4NGJetID::output_regression_type tempPt = ctHWTaggedJet.hwPt;

    // If ctHWTaggedJet within eta and pt range, then apply the correction
    l1ct::glbeta_t eta_abs = ctHWTaggedJet.hwEta < 0 ? l1ct::glbeta_t(-ctHWTaggedJet.hwEta) : ctHWTaggedJet.hwEta;
    l1ct::glbeta_t max_eta = fMaxEta_ / l1ct::Scales::ETAPHI_LSB;
    // If we want to update the hwPt of the jet?
    if (!returnRawPt_) {
      if (eta_abs < max_eta && ctHWTaggedJet.hwPt > l1ct::Scales::makePtFromFloat(fMinPt_)) {
        tempPt = ctHWTaggedJet.hwPt * PtCorrection_;
      } else {
        //If outside of the eta and pt range, clear out the tag scores
        JetScore_float.clear();
        for (unsigned i = 0; i < classes_.size(); i++) {
          ctHWTaggedJet.hwTagScores[i] = 0;
          JetScore_float.push_back(0);
        }

        if (doJEC) {
          float correctedPt = corrector->correctedPt(ctHWTaggedJet.floatPt(), ctHWTaggedJet.floatEta());
          tempPt = correctedPt;
        }
      }
    }

    ctHWTaggedJet.hwPt = l1ct::pt_t(tempPt);
    l1gt::Jet gtHWTaggedJet = ctHWTaggedJet.toGT();
    // TODO set the regressed pT instead of the srcjet pt
    l1t::PFJet edmTaggedJet(srcjet.pt(),
                            srcjet.eta(),
                            srcjet.phi(),
                            srcjet.mass(),
                            gtHWTaggedJet.v3.pt.V,
                            gtHWTaggedJet.v3.eta.V,
                            gtHWTaggedJet.v3.phi.V);
    edmTaggedJet.setEncodedJet(l1t::PFJet::HWEncoding::CT, ctHWTaggedJet.pack());
    edmTaggedJet.setEncodedJet(l1t::PFJet::HWEncoding::GT, gtHWTaggedJet.pack());

    // Convert corrected hwPt to float (GeV)
    float correctedPt = ctHWTaggedJet.floatPt();
    edmTaggedJet.calibratePt(correctedPt);

    std::vector<edm::Ptr<l1t::PFCandidate>> constituents;
    std::for_each(srcjet.constituents().begin(), srcjet.constituents().end(), [&](auto constituent) {
      edmTaggedJet.addConstituent(constituent);
    });
    edmTaggedJet.addTagScores(JetScore_float, classes_, PtCorrection_);
    taggedJets.push_back(edmTaggedJet);
  }

  auto taggedJetsCollection = std::make_unique<l1t::PFJetCollection>();
  taggedJetsCollection->swap(taggedJets);
  iEvent.put(std::move(taggedJetsCollection), "l1tSC4NGJets");
}

void L1TSC4NGJetProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jets", edm::InputTag("l1tSC4PFL1PuppiExtendedEmulator"));
  desc.add<bool>("doJEC", false);
  desc.add<bool>("returnRawPt", false);
  desc.add<std::string>("correctorFile", "");
  desc.add<std::string>("correctorDir", "");
  desc.add<edm::FileInPath>(
          "model", edm::FileInPath("L1Trigger/Phase2L1ParticleFlow/data/alkaid/L1TSC4NGJetModel_v2_0_0.json.gz"))
      ->setComment("jet tagger + pT regression (alkaid ALIR file); L1TSC4NGJetModel_v2_0_0 from cms-hls4ml v2.1.0");
  desc.add<int>("maxJets", 16);
  desc.add<int>("nParticles", 16);
  desc.add<double>("minPt", 10);
  desc.add<double>("maxEta", 2.4);
  desc.add<std::vector<std::string>>("classes", {"b", "c", "uds", "g", "tau_p", "tau_n", "mu", "e"});
  descriptions.add("l1tSC4NGJetProducer", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TSC4NGJetProducer);
