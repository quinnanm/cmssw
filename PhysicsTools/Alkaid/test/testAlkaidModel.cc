// Checks the alkaid external through cms::alkaid::Model with two toy models, independently of any physics model.
// If this fails, the external is not built, linked or found correctly, and nothing that uses it is meaningful.
//
// test/data/ (from the alkaid evaluation, toymodel/):
//   toy_model.json.gz + toy_model.golden.bin            4 -> 3 dense + ReLU, 5 samples
//   toy_model_ext.json.gz + toy_model_ext_21.golden.bin SAT/RND input, sigmoid LUT, var*var, 32-bit SAT;
//                                                       first 21 of 4000 golden samples (2 full batches + 5 single)
// The reference outputs are numpy evaluations of the same functions (no tracing involved).

#include "catch2/catch_all.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/Alkaid/interface/GoldenVectors.h"
#include "PhysicsTools/Alkaid/interface/Model.h"

namespace {

  const std::string kDataDir = "PhysicsTools/Alkaid/test/data/";

  std::string fullPath(const std::string& file) { return edm::FileInPath(kDataDir + file).fullPath(); }

  void checkAgainstGolden(const std::string& modelFile, const std::string& goldenFile) {
    const auto golden = cms::alkaid::readGoldenVectors(fullPath(goldenFile));
    REQUIRE(golden.nSamples > 0);
    const cms::alkaid::Model model(fullPath(modelFile), golden.nInputs, golden.nOutputs);
    cms::alkaid::Model::Scratch scratch;

    SECTION("all samples in one call (batched)") {
      std::vector<double> outputs(golden.outputs.size());
      model.evaluate(golden.inputs, outputs, scratch);
      CHECK(cms::alkaid::bitIdentical(outputs, golden.outputs));
    }
    SECTION("one sample per call") {
      std::vector<double> outputs(golden.outputs.size());
      for (uint64_t i = 0; i < golden.nSamples; ++i)
        model.evaluate(std::span(golden.inputs).subspan(i * golden.nInputs, golden.nInputs),
                       std::span(outputs).subspan(i * golden.nOutputs, golden.nOutputs),
                       scratch);
      CHECK(cms::alkaid::bitIdentical(outputs, golden.outputs));
    }
  }

}  // namespace

TEST_CASE("Toy models reproduce their golden outputs", "[alkaid]") {
  SECTION("toy_model") { checkAgainstGolden("toy_model.json.gz", "toy_model.golden.bin"); }
  SECTION("toy_model_ext") { checkAgainstGolden("toy_model_ext.json.gz", "toy_model_ext_21.golden.bin"); }
}

TEST_CASE("Model rejects bad input", "[alkaid]") {
  SECTION("missing model file") {
    CHECK_THROWS_AS(cms::alkaid::Model("/nonexistent/model.json.gz", 4, 3), cms::Exception);
  }
  SECTION("wrong shape") { CHECK_THROWS_AS(cms::alkaid::Model(fullPath("toy_model.json.gz"), 4, 2), cms::Exception); }

  const cms::alkaid::Model model(fullPath("toy_model.json.gz"), 4, 3);
  cms::alkaid::Model::Scratch scratch;
  SECTION("no samples") {
    std::vector<double> outputs;
    CHECK_NOTHROW(model.evaluate(std::span<const double>(), outputs, scratch));
  }
  SECTION("inputs not a multiple of nInputs") {
    const std::vector<double> inputs(model.nInputs() + 1);
    std::vector<double> outputs(model.nOutputs());
    CHECK_THROWS_AS(model.evaluate(inputs, outputs, scratch), cms::Exception);
  }
  SECTION("outputs for a different number of samples") {
    const std::vector<double> inputs(2 * model.nInputs());
    std::vector<double> outputs(model.nOutputs());
    CHECK_THROWS_AS(model.evaluate(inputs, outputs, scratch), cms::Exception);
  }
  SECTION("not a golden-vector file") {
    CHECK_THROWS_AS(cms::alkaid::readGoldenVectors(fullPath("toy_model.json.gz")), cms::Exception);
  }
}
