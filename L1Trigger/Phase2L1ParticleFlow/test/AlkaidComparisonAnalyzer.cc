// Compares the outputs of two copies of the same NN producer, one with backend = "hls4ml" (reference) and one with
// backend = "alkaid" (test), jet by jet on real events. Every field the producers set must be identical:
//   valueMaps (TOoLLiPProducer): the score of every jet in the jet collection both copies read
//   jetCollections (L1TSC4NGJetProducer, L1TSC82ProngJetProducer): size, then per jet pt/eta/phi/mass, raw pt, tag
//     scores and classes, pt correction, and the CT/GT/GTWide hardware words
// endJob prints one summary line per pair ("AlkaidComparison <name>: events=… jets=… evaluated=… mismatches=…") and,
// with failOnMismatch, throws if any pair has a mismatch, so the job's exit code says whether the back ends agree.

#include <atomic>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

class AlkaidComparisonAnalyzer : public edm::global::EDAnalyzer<> {
public:
  explicit AlkaidComparisonAnalyzer(const edm::ParameterSet& cfg);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  // Counters of one compared pair; atomic because events are analyzed concurrently.
  struct Counters {
    std::atomic<uint64_t> events{0}, jets{0}, evaluated{0}, mismatches{0};
  };
  struct ValueMapPair {
    std::string name;
    edm::EDGetTokenT<edm::View<l1t::PFJet>> jets;
    edm::EDGetTokenT<edm::ValueMap<float>> reference, test;
    float notEvaluated;  // value the producer stores for jets it doesn't evaluate
  };
  struct JetCollectionPair {
    std::string name;
    edm::EDGetTokenT<l1t::PFJetCollection> reference, test;
  };

  void analyze(edm::StreamID, const edm::Event& event, const edm::EventSetup&) const override;
  void endJob() override;
  // Logs the first few mismatches of a pair, with enough context to reproduce them.
  void reportMismatch(const std::string& name, Counters& counters, const std::string& details) const;
  static bool sameJet(const l1t::PFJet& a, const l1t::PFJet& b);
  static std::string describe(const l1t::PFJet& jet);

  static constexpr uint64_t kReportedMismatches = 5;

  std::vector<ValueMapPair> valueMaps_;
  std::vector<JetCollectionPair> jetCollections_;
  const bool failOnMismatch_;
  // One Counters per pair: valueMaps_ first, then jetCollections_ (std::vector<std::atomic> can't be resized).
  mutable std::vector<Counters> counters_;
};

AlkaidComparisonAnalyzer::AlkaidComparisonAnalyzer(const edm::ParameterSet& cfg)
    : failOnMismatch_(cfg.getParameter<bool>("failOnMismatch")) {
  for (const auto& pair : cfg.getParameter<std::vector<edm::ParameterSet>>("valueMaps"))
    valueMaps_.push_back({pair.getParameter<std::string>("name"),
                          consumes<edm::View<l1t::PFJet>>(pair.getParameter<edm::InputTag>("jets")),
                          consumes<edm::ValueMap<float>>(pair.getParameter<edm::InputTag>("reference")),
                          consumes<edm::ValueMap<float>>(pair.getParameter<edm::InputTag>("test")),
                          static_cast<float>(pair.getParameter<double>("notEvaluated"))});
  for (const auto& pair : cfg.getParameter<std::vector<edm::ParameterSet>>("jetCollections"))
    jetCollections_.push_back({pair.getParameter<std::string>("name"),
                               consumes<l1t::PFJetCollection>(pair.getParameter<edm::InputTag>("reference")),
                               consumes<l1t::PFJetCollection>(pair.getParameter<edm::InputTag>("test"))});
  counters_ = std::vector<Counters>(valueMaps_.size() + jetCollections_.size());
}

void AlkaidComparisonAnalyzer::analyze(edm::StreamID, const edm::Event& event, const edm::EventSetup&) const {
  for (size_t p = 0; p < valueMaps_.size(); ++p) {
    const auto& pair = valueMaps_[p];
    auto& counters = counters_[p];
    const auto jets = event.getHandle(pair.jets);  // the ValueMaps are keyed by this product's id
    const auto& reference = event.get(pair.reference);
    const auto& test = event.get(pair.test);
    ++counters.events;
    for (size_t i = 0; i < jets->size(); ++i) {
      const float referenceScore = reference.get(jets.id(), i);
      const float testScore = test.get(jets.id(), i);
      ++counters.jets;
      if (referenceScore != pair.notEvaluated)
        ++counters.evaluated;
      if (referenceScore != testScore) {
        std::ostringstream details;
        details << "event " << event.id() << " jet " << i << " (" << describe((*jets)[i])
                << "): hls4ml=" << referenceScore << " alkaid=" << testScore;
        reportMismatch(pair.name, counters, details.str());
      }
    }
  }
  for (size_t p = 0; p < jetCollections_.size(); ++p) {
    const auto& pair = jetCollections_[p];
    auto& counters = counters_[valueMaps_.size() + p];
    const auto& reference = event.get(pair.reference);
    const auto& test = event.get(pair.test);
    ++counters.events;
    counters.jets += reference.size();
    counters.evaluated += reference.size();  // these producers only store the jets they evaluated
    if (reference.size() != test.size()) {
      std::ostringstream details;
      details << "event " << event.id() << ": hls4ml " << reference.size() << " jets, alkaid " << test.size();
      reportMismatch(pair.name, counters, details.str());
      continue;
    }
    for (size_t i = 0; i < reference.size(); ++i) {
      if (!sameJet(reference[i], test[i])) {
        std::ostringstream details;
        details << "event " << event.id() << " jet " << i << ": hls4ml {" << describe(reference[i]) << "} alkaid {"
                << describe(test[i]) << "}";
        reportMismatch(pair.name, counters, details.str());
      }
    }
  }
}

void AlkaidComparisonAnalyzer::reportMismatch(const std::string& name,
                                              Counters& counters,
                                              const std::string& details) const {
  const uint64_t previous = counters.mismatches++;
  if (previous < kReportedMismatches)
    edm::LogWarning("AlkaidComparison") << name << " mismatch: " << details;
}

bool AlkaidComparisonAnalyzer::sameJet(const l1t::PFJet& a, const l1t::PFJet& b) {
  using Encoding = l1t::PFJet::HWEncoding;
  return a.pt() == b.pt() && a.eta() == b.eta() && a.phi() == b.phi() && a.mass() == b.mass() &&
         a.rawPt() == b.rawPt() && a.getTagScores() == b.getTagScores() && a.getTagClasses() == b.getTagClasses() &&
         a.getPtCorrection() == b.getPtCorrection() && a.encodedJet(Encoding::CT) == b.encodedJet(Encoding::CT) &&
         a.encodedJet(Encoding::GT) == b.encodedJet(Encoding::GT) &&
         a.encodedJet(Encoding::GTWide) == b.encodedJet(Encoding::GTWide);
}

std::string AlkaidComparisonAnalyzer::describe(const l1t::PFJet& jet) {
  std::ostringstream text;
  text << "pt=" << jet.pt() << " eta=" << jet.eta() << " phi=" << jet.phi() << " ptCorrection=" << jet.getPtCorrection()
       << " scores:";
  for (float score : jet.getTagScores())
    text << ' ' << score;
  return text.str();
}

void AlkaidComparisonAnalyzer::endJob() {
  uint64_t totalMismatches = 0;
  for (size_t p = 0; p < counters_.size(); ++p) {
    const auto& name = p < valueMaps_.size() ? valueMaps_[p].name : jetCollections_[p - valueMaps_.size()].name;
    const auto& counters = counters_[p];
    edm::LogPrint("AlkaidComparison") << "AlkaidComparison " << name << ": events=" << counters.events.load()
                                      << " jets=" << counters.jets.load() << " evaluated=" << counters.evaluated.load()
                                      << " mismatches=" << counters.mismatches.load();
    totalMismatches += counters.mismatches.load();
  }
  if (failOnMismatch_ && totalMismatches > 0)
    throw cms::Exception("AlkaidComparison") << totalMismatches << " jets differ between hls4ml and alkaid";
}

void AlkaidComparisonAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription valueMap;
  valueMap.add<std::string>("name");
  valueMap.add<edm::InputTag>("jets");
  valueMap.add<edm::InputTag>("reference");
  valueMap.add<edm::InputTag>("test");
  valueMap.add<double>("notEvaluated", -999.);
  edm::ParameterSetDescription jetCollection;
  jetCollection.add<std::string>("name");
  jetCollection.add<edm::InputTag>("reference");
  jetCollection.add<edm::InputTag>("test");

  edm::ParameterSetDescription desc;
  desc.addVPSet("valueMaps", valueMap, {});
  desc.addVPSet("jetCollections", jetCollection, {});
  desc.add<bool>("failOnMismatch", true);
  descriptions.add("alkaidComparisonAnalyzer", desc);
}

DEFINE_FWK_MODULE(AlkaidComparisonAnalyzer);
