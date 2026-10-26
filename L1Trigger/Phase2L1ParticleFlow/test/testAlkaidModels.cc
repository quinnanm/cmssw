// Checks the alkaid models against their golden vectors, without hls4ml:
//  - the toy models: the alkaid external alone, independently of any physics model. If they fail, the external is
//    not built, linked or found correctly, and nothing else is meaningful;
//  - the models the producers use (data/alkaid/): their golden outputs come from the hls4ml emulators they replace,
//    so this keeps checking hls4ml equivalence once the hls4ml externals are gone.
// Every model is evaluated batched (all samples in one call) and one sample per call, and must be bit-identical.
//
// Toy model files in test/alkaid/ (from the alkaid evaluation sandbox, toymodel/):
//   toy_model.json.gz + toy_model.golden.bin            4 -> 3 dense + ReLU, 5 samples
//   toy_model_ext.json.gz + toy_model_ext_21.golden.bin SAT/RND input, sigmoid LUT, var*var, 32-bit SAT;
//                                                       first 21 of 4000 golden samples (2 full batches + 5 single)

#include "catch2/catch_all.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "ap_fixed.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/AlkaidEmulatorModel.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/AlkaidModel.h"
#include "L1Trigger/Phase2L1ParticleFlow/test/AlkaidGolden.h"

namespace {

  using l1tpf::test::bitIdentical;

  const std::string kTestDir = "L1Trigger/Phase2L1ParticleFlow/test/alkaid/";
  const std::string kDataDir = "L1Trigger/Phase2L1ParticleFlow/data/alkaid/";

  // model, golden: paths relative to the CMSSW search path
  void checkAgainstGolden(const std::string& model, const std::string& golden) {
    const l1tpf::AlkaidModel alkaid(edm::FileInPath(model).fullPath());
    const auto reference = l1tpf::test::readAlkaidGolden(golden);
    REQUIRE(reference.nSamples > 0);
    REQUIRE(alkaid.nInputs() == reference.nInputs);
    REQUIRE(alkaid.nOutputs() == reference.nOutputs);
    std::vector<int64_t> scratch;

    SECTION("all samples in one call (batched)") {
      std::vector<double> outputs(reference.outputs.size());
      alkaid.evaluate(reference.inputs, outputs, scratch);
      CHECK(bitIdentical(outputs, reference.outputs));
    }
    SECTION("one sample per call") {
      std::vector<double> outputs(reference.outputs.size());
      for (uint64_t i = 0; i < reference.nSamples; ++i)
        alkaid.evaluate(std::span(reference.inputs).subspan(i * reference.nInputs, reference.nInputs),
                        std::span(outputs).subspan(i * reference.nOutputs, reference.nOutputs),
                        scratch);
      CHECK(bitIdentical(outputs, reference.outputs));
    }
  }

}  // namespace

TEST_CASE("Toy models reproduce their golden outputs", "[alkaid]") {
  SECTION("toy_model") { checkAgainstGolden(kTestDir + "toy_model.json.gz", kTestDir + "toy_model.golden.bin"); }
  SECTION("toy_model_ext") {
    checkAgainstGolden(kTestDir + "toy_model_ext.json.gz", kTestDir + "toy_model_ext_21.golden.bin");
  }
}

TEST_CASE("Producer models reproduce their golden (hls4ml) outputs", "[alkaid]") {
  for (const std::string model : {"TOoLLiP_v3_v3_1_0", "L1TSC82ProngJetModel_v0", "L1TSC4NGJetModel_v2_0_0"}) {
    DYNAMIC_SECTION(model) { checkAgainstGolden(kDataDir + model + ".json.gz", kDataDir + model + ".golden.bin"); }
  }
}

TEST_CASE("AlkaidModel rejects bad input", "[alkaid]") {
  const l1tpf::AlkaidModel alkaid(edm::FileInPath(kTestDir + "toy_model.json.gz").fullPath());
  std::vector<int64_t> scratch;
  std::vector<double> outputs(alkaid.nOutputs());

  SECTION("inputs not a multiple of nInputs") {
    const std::vector<double> inputs(alkaid.nInputs() + 1);
    CHECK_THROWS_AS(alkaid.evaluate(inputs, outputs, scratch), cms::Exception);
  }
  SECTION("outputs of the wrong size") {
    const std::vector<double> inputs(2 * alkaid.nInputs());
    CHECK_THROWS_AS(alkaid.evaluate(inputs, outputs, scratch), cms::Exception);
  }
  SECTION("missing model file") { CHECK_THROWS(l1tpf::AlkaidModel("/nonexistent/model.json.gz")); }
}

TEST_CASE("AlkaidEmulatorModel behaves like an hls4ml emulator", "[alkaid]") {
  // Use the toy model through the hls4mlEmulator::Model interface, with ap_fixed arrays as a producer would.
  // The toy model truncates its inputs to 4 fractional bits. Truncating to 8 first (AP_TRN) changes nothing, since
  // floor(floor(x * 2^8) / 2^4) == floor(x * 2^4), so the outputs must still equal the golden ones exactly.
  using InputType = ap_fixed<16, 8, AP_TRN, AP_SAT>;
  using OutputType = ap_fixed<32, 16, AP_TRN, AP_SAT>;
  const auto reference = l1tpf::test::readAlkaidGolden(kTestDir + "toy_model.golden.bin");
  auto alkaid = std::make_shared<const l1tpf::AlkaidModel>(edm::FileInPath(kTestDir + "toy_model.json.gz").fullPath());
  std::unique_ptr<hls4mlEmulator::Model> emulator =
      std::make_unique<l1tpf::AlkaidEmulatorModel<InputType, OutputType>>(alkaid);

  std::vector<InputType> inputs(reference.nInputs);
  std::vector<OutputType> outputs(reference.nOutputs);
  for (uint64_t i = 0; i < reference.nSamples; ++i) {
    for (uint64_t j = 0; j < reference.nInputs; ++j)
      inputs[j] = reference.inputs[i * reference.nInputs + j];
    emulator->prepare_input(inputs.data());
    emulator->predict();
    emulator->read_result(outputs.data());
    for (uint64_t j = 0; j < reference.nOutputs; ++j)
      CHECK(outputs[j].to_double() == reference.outputs[i * reference.nOutputs + j]);
  }
  // A type mismatch must fail loudly, as with the hls4ml wrappers.
  std::vector<float> wrongType(reference.nInputs);
  CHECK_THROWS_AS(emulator->prepare_input(wrongType.data()), std::bad_any_cast);
}
