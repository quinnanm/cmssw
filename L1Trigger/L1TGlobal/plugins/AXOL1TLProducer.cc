////
/// \class l1t::AnomalyDetectionAEProducer
///
/// Description: Create input for anomaly detection autoencoder model inference
///
///  Modeled after BXVectorInputProducer.cc
///

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#include <vector>
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"

#include "TMath.h"

using namespace std;
using namespace edm;

//HLS4ML compiled emulator modeling
#include <string>
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

//
// class declaration
//

class AXOL1TLProducer : public one::EDProducer<> {
public:
  explicit AXOL1TLProducer(const ParameterSet&);
  ~AXOL1TLProducer() override;

  static void fillDescriptions(ConfigurationDescriptions& descriptions);

private:
  void produce(Event&, EventSetup const&) override;
  void beginJob() override;  //not sure if need
  void endJob() override;
  // void beginRun(Run const& iR, EventSetup const& iE) override;

  // ----------member data ---------------------------

  // Tokens for inputs from other parts of the L1 system
  edm::EDGetToken egToken;
  edm::EDGetToken muToken;
  edm::EDGetToken jetToken;
  edm::EDGetToken etsumToken;

  // edm::EDGetTokenT <l1t::EGammaBxCollection> egToken;
  // edm::EDGetTokenT <l1t::JetBxCollection>    jetToken;
  // edm::EDGetTokenT <l1t::EtSumBxCollection>  etsumToken;
  // edm::EDGetTokenT <l1t::MuonBxCollection>   muToken;

  //HLS4ML emulator objects
  hls4mlEmulator::ModelLoader loader;
  std::shared_ptr<hls4mlEmulator::Model> model;

  //emulator constants
  const int NInputs = 57;
  const int NMuons = 4;
  const int NJets = 10;
  const int NEgammas = 4;
};

AXOL1TLProducer::AXOL1TLProducer(const ParameterSet& iConfig)
    : loader(hls4mlEmulator::ModelLoader(iConfig.getParameter<string>("AXOL1TLModelVersion")))  //local or from cms-dist
{
  egToken = consumes<l1t::EGammaBxCollection>(iConfig.getParameter<InputTag>("egInputTag"));
  muToken = consumes<l1t::MuonBxCollection>(iConfig.getParameter<InputTag>("muInputTag"));
  jetToken = consumes<l1t::JetBxCollection>(iConfig.getParameter<InputTag>("jetInputTag"));
  etsumToken = consumes<l1t::EtSumBxCollection>(iConfig.getParameter<InputTag>("etsumInputTag"));

  // register what you produce
  produces<std::vector<float>>("anomalyInput");
  produces<std::vector<float>>("anomalyResult");
  produces<float>("anomalyScore");

  //AE model
  model = loader.load_model();
}

AXOL1TLProducer::~AXOL1TLProducer() {
  // //delete model
  // loader.destroy_model();
}

//
// member functions
//

// ------------ method called to produce the data ------------
void AXOL1TLProducer::produce(Event& iEvent, const EventSetup& iSetup) {
  LogDebug("l1t|Global") << "AXOL1TLProducer::produce function called...\n";

  // Get input vectors
  edm::Handle<l1t::EGammaBxCollection> inputEgammas;
  edm::Handle<l1t::JetBxCollection> inputJets;
  edm::Handle<l1t::EtSumBxCollection> inputEtsums;
  edm::Handle<l1t::MuonBxCollection> inputMuons;

  iEvent.getByToken(egToken, inputEgammas);
  iEvent.getByToken(jetToken, inputJets);
  iEvent.getByToken(etsumToken, inputEtsums);
  iEvent.getByToken(muToken, inputMuons);

  if (!(iEvent.getByToken(egToken, inputEgammas))) {
    LogTrace("l1t|Global") << ">>> input EG collection not found!" << std::endl;
  };
  if (!(iEvent.getByToken(jetToken, inputJets))) {
    LogTrace("l1t|Global") << ">>> input jet collection not found!" << std::endl;
  };
  if (!(iEvent.getByToken(etsumToken, inputEtsums))) {
    LogTrace("l1t|Global") << ">>> input etsum collection not found!" << std::endl;
  };
  if (!(iEvent.getByToken(muToken, inputMuons))) {
    LogTrace("l1t|Global") << ">>> input Mu collection not found!" << std::endl;
  };

  //outputs
  std::unique_ptr<std::vector<float>> anomalyInput(new std::vector<float>(0));
  std::unique_ptr<std::vector<float>> anomalyResult(new std::vector<float>(0));
  std::unique_ptr<float> anomalyScore(new float);  //store anomaly score

  // Input and output of  the model is in the input_t format as defined in the model's firmware/defines.h
  ap_fixed<18, 13> ADModelInput[57] = {};
  std::array<ap_fixed<10, 7>, 13> result;
  ap_ufixed<18, 14> loss;
  std::pair<std::array<ap_fixed<10, 7>, 13>, ap_ufixed<18, 14>> ADModelResult;

  //////////////////
  // Insert all the bx into the L1 Collections

  //ADModelInput = [EtSum.et(), EtSum.eta(), EtSum.phi(),
  //                4 egammas *(egamma_i.pt(), egamma_i.eta(), egamma_i.phi()),
  //                4 muons *(muon_i.pt(), muon_i.eta(), muon_i.phi()),
  //                10 jets *(jpt_i.et(), jet_i.eta(), jet_i.phi()),  ]

  //default fillval:
  ap_fixed<18, 13> fillzero = 0.0;

  // Fill Etsums (only the first one)
  for (int ibx = inputEtsums->getFirstBX(); ibx <= inputEtsums->getLastBX(); ++ibx) {  //get object
    if (ibx != 0)
      continue;
    for (l1t::EtSumBxCollection::const_iterator it = inputEtsums->begin(ibx); it != inputEtsums->end(ibx); it++) {
      int type = static_cast<int>(it->getType());
      if (type != 2)
        continue;  //we only really care about the MET, which is sum type 2 ?need?
      ADModelInput[0] = it->et();
      ADModelInput[1] = fillzero;
      ADModelInput[2] = it->phi();
    }
  }

  //counter of starting index of next loop
  int starti = 3;

  // Fill Egammas
  int nEGs = 0;
  for (int ibx = inputEgammas->getFirstBX(); ibx <= inputEgammas->getLastBX(); ++ibx) {
    if (ibx != 0)
      continue;
    for (l1t::EGammaBxCollection::const_iterator it = inputEgammas->begin(ibx);
         it != inputEgammas->end(ibx) && nEGs < NEgammas;
         it++) {
      if (it->pt() > 0) {
        nEGs++;
        ADModelInput[starti + 0] = it->et();  //starti=3
        ADModelInput[starti + 1] = it->eta();
        ADModelInput[starti + 2] = it->phi();
        starti += 3;
      }
    }
    if (nEGs < NEgammas) {  //pad array
      while (nEGs < NEgammas) {
        nEGs++;
        ADModelInput[starti + 0] = fillzero;
        ADModelInput[starti + 1] = fillzero;
        ADModelInput[starti + 2] = fillzero;
        starti += 3;
      }
    }
  }

  // Fill Muons
  starti = 15;
  int nMUs = 0;
  for (int ibx = inputMuons->getFirstBX(); ibx <= inputMuons->getLastBX(); ++ibx) {
    if (ibx != 0)
      continue;
    for (l1t::MuonBxCollection::const_iterator it = inputMuons->begin(ibx); it != inputMuons->end(ibx) && nMUs < NMuons;
         it++) {
      if (it->pt() > 0) {
        nMUs++;
        ADModelInput[starti + 0] = it->et();  //starti=3
        ADModelInput[starti + 1] = it->eta();
        ADModelInput[starti + 2] = it->phi();
        starti += 3;
      }
    }
    if (nMUs < NMuons) {  //pad array
      while (nMUs < NMuons) {
        nMUs++;
        ADModelInput[starti + 0] = fillzero;
        ADModelInput[starti + 1] = fillzero;
        ADModelInput[starti + 2] = fillzero;
        starti += 3;
      }
    }
  }

  // Fill Jets
  int nJs = 0;
  starti = 27;
  for (int ibx = inputJets->getFirstBX(); ibx <= inputJets->getLastBX(); ++ibx) {
    if (ibx != 0)
      continue;
    for (l1t::JetBxCollection::const_iterator it = inputJets->begin(ibx); it != inputJets->end(ibx) && nJs < NJets;
         it++) {
      if (it->pt() > 0) {
        nJs++;
        ADModelInput[starti + 0] = it->et();  //starti=3
        ADModelInput[starti + 1] = it->eta();
        ADModelInput[starti + 2] = it->phi();
        starti += 3;
      }
    }
    if (nJs < NJets) {  //pad array
      while (nJs < NJets) {
        nJs++;
        ADModelInput[starti + 0] = fillzero;
        ADModelInput[starti + 1] = fillzero;
        ADModelInput[starti + 2] = fillzero;
        starti += 3;
      }
    }
  }

  // run inference on anomaly model
  model->prepare_input(ADModelInput);  //scaling internal here
  model->predict();
  model->read_result(&ADModelResult);  // this should be the square sum model result

  result = ADModelResult.first;
  loss = ADModelResult.second;
  *anomalyScore = (loss).to_float();  //convert the fixed precision result to a proper c++ floating point

  for (int i = 0; i < NInputs; i++) {
    anomalyInput->push_back((ADModelInput[i]).to_float());
  }

  for (int i = 0; i < 13; i++) {
    anomalyResult->push_back((result[i]).to_float());
  }

  iEvent.put(std::move(anomalyInput), "anomalyInput");
  iEvent.put(std::move(anomalyScore), "anomalyScore");
  iEvent.put(std::move(anomalyResult), "anomalyResult");
}

// ------------ method called once each job just before starting event loop ------------
void AXOL1TLProducer::beginJob() {}

// ------------ method called once each job just after ending the event loop ------------
void AXOL1TLProducer::endJob() {}

// // ------------ method called when starting to processes a run ------------

// void AXOL1TLProducer::beginRun(Run const& iR, EventSetup const& iE) {
//   LogDebug("GtAXOL1TLProducer") << "AXOL1TLProducer::beginRun function called...\n";

//   counter_ = 0;
// }

// // ------------ method called when ending the processing of a run ------------
// void AXOL1TLProducer::endRun(Run const& iR, EventSetup const& iE) {}

// ------------ method fills 'descriptions' with the allowed parameters for the module ------------
void AXOL1TLProducer::fillDescriptions(ConfigurationDescriptions& descriptions) {
  ParameterSetDescription desc;
  desc.add<edm::InputTag>("egInputTag", edm::InputTag(""))
      ->setComment("InputTag for Calo EGamma Trigger (required parameter:  default value is invalid)");
  desc.add<edm::InputTag>("muInputTag", edm::InputTag(""))
      ->setComment("InputTag for Global Muon Trigger (required parameter:  default value is invalid)");
  desc.add<edm::InputTag>("jetInputTag", edm::InputTag(""))
      ->setComment("InputTag for Calo Trigger Jet (required parameter:  default value is invalid)");
  desc.add<edm::InputTag>("etsumInputTag", edm::InputTag(""))
      ->setComment("InputTag for Calo Trigger EtSum (required parameter:  default value is invalid)");
  desc.add<std::string>("AXOL1TLModelVersion", "GTADModel_v1");
  descriptions.add("AXOL1TLProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(AXOL1TLProducer);
