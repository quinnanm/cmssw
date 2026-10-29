// Checks the NN models of TOoLLiPProducer, L1TSC82ProngJetProducer and L1TSC4NGJetProducer (data/alkaid/) against
// their golden vectors, whose outputs come from the hls4ml emulators these models replaced. This keeps checking that
// equivalence in every release, with the hls4ml emulators gone. Each model is evaluated batched (all samples in one
// call) and one sample per call, and must reproduce the golden outputs bit for bit.
// The model's input/output counts must also be those the producers use, and the ID classes must refuse a particle
// count the model has no room for.
// Uses REQUIRE only: JetId.h/L1TSC4NGJetID.h include TensorFlow, whose logging.h redefines CHECK (it would no longer
// be a Catch2 assertion).

#include "catch2/catch_all.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/JetId.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC4NGJetID.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC82ProngJetID.h"
#include "PhysicsTools/Alkaid/interface/GoldenVectors.h"
#include "PhysicsTools/Alkaid/interface/Model.h"

namespace {

  const std::string kDataDir = "L1Trigger/Phase2L1ParticleFlow/data/alkaid/";

  // name: file name without extension; nInputs/nOutputs: what the producer passes and reads
  void checkAgainstGolden(const std::string& name, size_t nInputs, size_t nOutputs) {
    const auto golden = cms::alkaid::readGoldenVectors(edm::FileInPath(kDataDir + name + ".golden.bin").fullPath());
    REQUIRE(golden.nSamples > 0);
    REQUIRE(golden.nInputs == nInputs);
    REQUIRE(golden.nOutputs == nOutputs);
    const cms::alkaid::Model model(edm::FileInPath(kDataDir + name + ".json.gz").fullPath(), nInputs, nOutputs);
    cms::alkaid::Model::Scratch scratch;

    SECTION("all samples in one call (batched)") {
      std::vector<double> outputs(golden.outputs.size());
      model.evaluate(golden.inputs, outputs, scratch);
      REQUIRE(cms::alkaid::bitIdentical(outputs, golden.outputs));
    }
    SECTION("one sample per call (as the producers do)") {
      std::vector<double> outputs(golden.outputs.size());
      for (uint64_t i = 0; i < golden.nSamples; ++i)
        model.evaluate(std::span(golden.inputs).subspan(i * nInputs, nInputs),
                       std::span(outputs).subspan(i * nOutputs, nOutputs),
                       scratch);
      REQUIRE(cms::alkaid::bitIdentical(outputs, golden.outputs));
    }
  }

}  // namespace

TEST_CASE("TOoLLiP_v3 (TOoLLiP 3.1.0) reproduces its golden (hls4ml) outputs", "[alkaid][TOoLLiP]") {
  checkAgainstGolden("TOoLLiP_v3_v3_1_0", JetId::kNNInputs, JetId::kNNOutputs);
}

TEST_CASE("L1TSC82ProngJetModel_v0 reproduces its golden (hls4ml) outputs", "[alkaid][L1TSC82]") {
  checkAgainstGolden("L1TSC82ProngJetModel_v0", L1TSC82ProngJetID::kNNInputs, L1TSC82ProngJetID::kNNOutputs);
}

TEST_CASE("L1TSC4NGJetModel_v2_0_0 reproduces its golden (hls4ml) outputs", "[alkaid][L1TSC4NG]") {
  checkAgainstGolden("L1TSC4NGJetModel_v2_0_0", L1TSC4NGJetID::kModelInputs, L1TSC4NGJetID::kModelOutputs);
}

TEST_CASE("The ID classes refuse more particles than their model takes", "[alkaid]") {
  SECTION("TOoLLiP (JetId)") {
    const cms::alkaid::Model model(
        edm::FileInPath(kDataDir + "TOoLLiP_v3_v3_1_0.json.gz").fullPath(), JetId::kNNInputs, JetId::kNNOutputs);
    constexpr int kMaxParticles = JetId::kNNInputs / JetId::kNNFeaturesPerParticle;
    REQUIRE_NOTHROW(JetId("input:0", "output", model, kMaxParticles));
    REQUIRE_THROWS_AS(JetId("input:0", "output", model, kMaxParticles + 1), cms::Exception);
  }
  SECTION("L1TSC4NG (L1TSC4NGJetID)") {
    const cms::alkaid::Model model(edm::FileInPath(kDataDir + "L1TSC4NGJetModel_v2_0_0.json.gz").fullPath(),
                                   L1TSC4NGJetID::kModelInputs,
                                   L1TSC4NGJetID::kModelOutputs);
    REQUIRE_NOTHROW(L1TSC4NGJetID(model, L1TSC4NGJetID::N_candidates, false));
    REQUIRE_THROWS_AS(L1TSC4NGJetID(model, L1TSC4NGJetID::N_candidates + 1, false), cms::Exception);
  }
}
