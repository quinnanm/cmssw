// The alkaid emulators must reproduce the hls4ml compiled emulators bit for bit, through the interface and
// fixed-point types the producers use (hls4mlEmulator::Model with the ID classes' ap_fixed arrays). Per model:
// the golden vectors (whose outputs must also be reproduced), special patterns (0, max, min, alternating), and
// random raw bit patterns over the whole input range, saturation values included.
//
// Use REQUIRE, not CHECK: JetId.h includes TensorFlow, and tsl/platform/default/logging.h redefines CHECK as an
// abort-on-failure macro, so a Catch2 CHECK here would be neither counted nor reported.

#include "catch2/catch_all.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <malloc.h>
#include <unistd.h>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/AlkaidEmulatorModel.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/AlkaidModel.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/JetId.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC4NGJetID.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/L1TSC82ProngJetID.h"
#include "L1Trigger/Phase2L1ParticleFlow/test/AlkaidGolden.h"
#include "hls4ml/emulator.h"

namespace {

  const std::string kDataDir = "L1Trigger/Phase2L1ParticleFlow/data/alkaid/";
  constexpr uint64_t kSeed = 20261011;

  struct ModelUnderTest {
    std::string hls4mlName;    // compiled model, dlopened as <hls4mlName>.so
    std::string alkaidSymbol;  // data/alkaid/<alkaidSymbol>.{json.gz,golden.bin}
    unsigned int nInputs;
    unsigned int nOutputs;
    // Input array size the hls4ml wrapper actually reads; larger than nInputs for TOoLLiP 3.1.0 (finding F2 of the
    // evaluation). The arrays are zero-padded to it so the reference itself doesn't read out of bounds.
    unsigned int hls4mlWrapperInputs;
    unsigned int randomSamples;
  };

  template <typename InputType, typename OutputType>
  class EmulatorComparison {
  public:
    explicit EmulatorComparison(const ModelUnderTest& model)
        : model_(model),
          loader_(model.hls4mlName),
          hls4ml_(loader_.load_model()),
          alkaid_(std::make_shared<l1tpf::AlkaidEmulatorModel<InputType, OutputType>>(
              std::make_shared<const l1tpf::AlkaidModel>(
                  edm::FileInPath(kDataDir + model.alkaidSymbol + ".json.gz").fullPath()))),
          inputs_(model.hls4mlWrapperInputs),
          hls4mlOutputs_(model.nOutputs),
          alkaidOutputs_(model.nOutputs) {}

    std::vector<InputType>& inputs() { return inputs_; }

    // Evaluates both emulators on inputs(); true if all outputs are identical.
    bool evaluate() {
      run(*hls4ml_, hls4mlOutputs_);
      run(*alkaid_, alkaidOutputs_);
      return hls4mlOutputs_ == alkaidOutputs_;
    }
    const std::vector<OutputType>& hls4mlOutputs() const { return hls4mlOutputs_; }

    std::string describe() const {
      std::ostringstream text;
      text << model_.alkaidSymbol << " hls4ml:";
      for (const auto& value : hls4mlOutputs_)
        text << ' ' << value.to_double();
      text << " alkaid:";
      for (const auto& value : alkaidOutputs_)
        text << ' ' << value.to_double();
      text << " inputs:";
      for (unsigned int i = 0; i < model_.nInputs; ++i)
        text << ' ' << inputs_[i].to_double();
      return text.str();
    }

  private:
    void run(hls4mlEmulator::Model& emulator, std::vector<OutputType>& outputs) {
      std::fill(outputs.begin(), outputs.end(), OutputType(-1));
      emulator.prepare_input(inputs_.data());
      emulator.predict();
      emulator.read_result(outputs.data());
    }

    ModelUnderTest model_;
    hls4mlEmulator::ModelLoader loader_;  // must outlive hls4ml_ (owns the dlopen handle)
    std::shared_ptr<hls4mlEmulator::Model> hls4ml_;
    std::shared_ptr<hls4mlEmulator::Model> alkaid_;
    std::vector<InputType> inputs_;
    std::vector<OutputType> hls4mlOutputs_, alkaidOutputs_;
  };

  template <typename InputType, typename OutputType>
  void compareEmulators(const ModelUnderTest& model) {
    EmulatorComparison<InputType, OutputType> emulators(model);
    auto& inputs = emulators.inputs();

    SECTION("golden vectors: alkaid == hls4ml == golden output") {
      const auto golden = l1tpf::test::readAlkaidGolden(kDataDir + model.alkaidSymbol + ".golden.bin");
      REQUIRE(golden.nInputs == model.nInputs);
      REQUIRE(golden.nOutputs == model.nOutputs);
      unsigned int mismatches = 0;
      for (uint64_t sample = 0; sample < golden.nSamples; ++sample) {
        for (unsigned int i = 0; i < model.nInputs; ++i)
          inputs[i] = golden.inputs[sample * model.nInputs + i];
        bool same = emulators.evaluate();
        for (unsigned int j = 0; j < model.nOutputs; ++j)
          same = same && emulators.hls4mlOutputs()[j] == OutputType(golden.outputs[sample * model.nOutputs + j]);
        if (!same && mismatches++ == 0)
          UNSCOPED_INFO("first mismatch, golden sample " << sample << ": " << emulators.describe());
      }
      REQUIRE(mismatches == 0);
    }

    SECTION("special patterns") {
      // Largest and smallest InputType values: 2^(I-1) - 2^(I-W) and -2^(I-1).
      const double maxValue =
          std::ldexp(1.0, InputType::iwidth - 1) - std::ldexp(1.0, InputType::iwidth - InputType::width);
      const double minValue = -std::ldexp(1.0, InputType::iwidth - 1);
      // {value at even index, value at odd index}
      const std::array<std::pair<double, double>, 4> patterns{
          {{0, 0}, {maxValue, maxValue}, {minValue, minValue}, {maxValue, minValue}}};
      for (const auto& [even, odd] : patterns) {
        for (unsigned int i = 0; i < model.nInputs; ++i)
          inputs[i] = (i % 2 == 0) ? even : odd;
        const bool same = emulators.evaluate();
        INFO(emulators.describe());
        REQUIRE(same);
      }
    }

    SECTION("random raw bit patterns over the full input range") {
      static_assert(InputType::width < 64, "raw bit patterns are drawn as uint64_t");
      std::mt19937_64 generator(kSeed);
      std::uniform_int_distribution<uint64_t> rawBits(0, (uint64_t(1) << InputType::width) - 1);
      unsigned int mismatches = 0;
      for (unsigned int sample = 0; sample < model.randomSamples; ++sample) {
        for (unsigned int i = 0; i < model.nInputs; ++i)
          inputs[i].range() = rawBits(generator);
        if (!emulators.evaluate() && mismatches++ == 0)
          UNSCOPED_INFO("first mismatch, random sample " << sample << ": " << emulators.describe());
      }
      REQUIRE(mismatches == 0);
    }
  }

}  // namespace

TEST_CASE("TOoLLiP_v3 (TOoLLiP 3.1.0): alkaid == hls4ml", "[alkaid][TOoLLiP]") {
  compareEmulators<JetId::NNInputType, JetId::NNOutputType>({.hls4mlName = "TOoLLiP_v3",
                                                             .alkaidSymbol = "TOoLLiP_v3_v3_1_0",
                                                             .nInputs = JetId::kNNInputs,
                                                             .nOutputs = 1,
                                                             .hls4mlWrapperInputs = 140,
                                                             .randomSamples = 100000});
}

TEST_CASE("L1TSC82ProngJetModel_v0: alkaid == hls4ml", "[alkaid][L1TSC82]") {
  // The hls4ml reference costs ~1 ms per call: fewer random samples than for TOoLLiP.
  compareEmulators<L1TSC82ProngJetID::inputtype, L1TSC82ProngJetID::prong_score>(
      {.hls4mlName = "L1TSC82ProngJetModel_v0",
       .alkaidSymbol = "L1TSC82ProngJetModel_v0",
       .nInputs = L1TSC82ProngJetID::kNNInputs,
       .nOutputs = L1TSC82ProngJetID::kNNOutputs,
       .hls4mlWrapperInputs = L1TSC82ProngJetID::kNNInputs,
       .randomSamples = 10000});
}

namespace {

  // L1TSC4NGJetModel_v2_0_0 takes a ModelInputs struct (candidate block + jet block) and fills a ModelOutputs struct,
  // so it gets its own comparison. Both emulators are driven exactly as L1TSC4NGJetID::EvaluateNNFixed does, except
  // that the hls4ml wrapper gets room for all kModelClassOutputs class scores it writes (F2); the first
  // N_class_outputs and the regression score are compared.
  class L1TSC4NGComparison {
  public:
    using Value = L1TSC4NGJet::inputtype;
    static constexpr int kClassesUsed = L1TSC4NGJetID::N_class_outputs;
    static constexpr int kClassesWritten = L1TSC4NGJetID::kModelClassOutputs;

    L1TSC4NGComparison()
        : loader_("L1TSC4NGJetModel_v2_0_0"),
          hls4ml_(loader_.load_model()),
          alkaid_(L1TSC4NGJetID::makeAlkaidEmulator(std::make_shared<const l1tpf::AlkaidModel>(
              edm::FileInPath(kDataDir + "L1TSC4NGJetModel_v2_0_0.json.gz").fullPath()))) {}

    // Flat inputs: [candidate inputs, jet inputs], as in the ALIR model and the golden files.
    Value& input(int i) {
      return i < L1TSC4NGJetID::N_candidate_inputs ? candidates_[i] : jet_[i - L1TSC4NGJetID::N_candidate_inputs];
    }

    // Evaluates both; true if the regression score and the N_class_outputs used class scores are identical.
    bool evaluate() {
      run(*hls4ml_, hls4mlRegression_, hls4mlClasses_);
      run(*alkaid_, alkaidRegression_, alkaidClasses_);
      return hls4mlRegression_ == alkaidRegression_ &&
             std::equal(hls4mlClasses_.begin(), hls4mlClasses_.begin() + kClassesUsed, alkaidClasses_.begin());
    }
    // hls4ml outputs in the ALIR/golden order: regression, then all kClassesWritten class scores.
    std::array<Value, 1 + kClassesWritten> hls4mlOutputs() const {
      std::array<Value, 1 + kClassesWritten> outputs{};
      outputs[0] = hls4mlRegression_[0];
      std::copy(hls4mlClasses_.begin(), hls4mlClasses_.end(), outputs.begin() + 1);
      return outputs;
    }

    std::string describe() const {
      std::ostringstream text;
      text << "regression hls4ml=" << hls4mlRegression_[0].to_double() << " alkaid=" << alkaidRegression_[0].to_double()
           << "; classes hls4ml:";
      for (int i = 0; i < kClassesUsed; ++i)
        text << ' ' << hls4mlClasses_[i].to_double();
      text << " alkaid:";
      for (int i = 0; i < kClassesUsed; ++i)
        text << ' ' << alkaidClasses_[i].to_double();
      return text.str();
    }

  private:
    void run(hls4mlEmulator::Model& emulator,
             std::array<Value, L1TSC4NGJetID::N_regression_outputs>& regression,
             std::array<Value, kClassesWritten>& classes) {
      regression.fill(Value(-1));
      classes.fill(Value(-1));
      ModelInputs inputs{
          candidates_.data(), jet_.data(), L1TSC4NGJetID::N_candidate_features, L1TSC4NGJetID::N_jet_inputs};
      ModelOutputs outputs{classes.data(), regression.data()};
      emulator.prepare_input(inputs);
      emulator.predict();
      emulator.read_result(&outputs);
    }

    hls4mlEmulator::ModelLoader loader_;  // must outlive hls4ml_ (owns the dlopen handle)
    std::shared_ptr<hls4mlEmulator::Model> hls4ml_;
    std::shared_ptr<hls4mlEmulator::Model> alkaid_;
    std::array<Value, L1TSC4NGJetID::N_candidate_inputs> candidates_{};
    std::array<Value, L1TSC4NGJetID::N_jet_inputs> jet_{};
    std::array<Value, L1TSC4NGJetID::N_regression_outputs> hls4mlRegression_{}, alkaidRegression_{};
    std::array<Value, kClassesWritten> hls4mlClasses_{}, alkaidClasses_{};
  };

  // Input domain for v2_0_0: ap_fixed<64,32> values of up to kRawBits significant bits with the same 2^-32 step,
  // i.e. |x| < 2^(kRawBits-33) = 2^17. Exactly representable in double, inside the replay's contract (|x| < 2^29),
  // and still beyond the saturation limits of every input type the model applies (at most 16 integer bits).
  constexpr int kRawBits = 50;
  constexpr int kFractionalBits = 32;
  constexpr unsigned int kL1TSC4NGRandomSamples = 20000;

}  // namespace

TEST_CASE("L1TSC4NGJetModel_v2_0_0: alkaid == hls4ml", "[alkaid][L1TSC4NG]") {
  L1TSC4NGComparison emulators;
  constexpr int nInputs = L1TSC4NGJetID::kAlkaidInputs;

  SECTION("golden vectors: alkaid == hls4ml == golden output") {
    const auto golden = l1tpf::test::readAlkaidGolden(kDataDir + "L1TSC4NGJetModel_v2_0_0.golden.bin");
    REQUIRE(golden.nInputs == static_cast<uint64_t>(nInputs));
    REQUIRE(golden.nOutputs == static_cast<uint64_t>(L1TSC4NGJetID::kAlkaidOutputs));
    unsigned int mismatches = 0;
    for (uint64_t sample = 0; sample < golden.nSamples; ++sample) {
      for (int i = 0; i < nInputs; ++i)
        emulators.input(i) = golden.inputs[sample * nInputs + i];
      bool same = emulators.evaluate();
      const auto outputs = emulators.hls4mlOutputs();
      for (int j = 0; j < L1TSC4NGJetID::kAlkaidOutputs; ++j)
        same = same && outputs[j] == L1TSC4NGComparison::Value(golden.outputs[sample * golden.nOutputs + j]);
      if (!same && mismatches++ == 0)
        UNSCOPED_INFO("first mismatch, golden sample " << sample << ": " << emulators.describe());
    }
    REQUIRE(mismatches == 0);
  }

  SECTION("special patterns") {
    const double maxValue = std::ldexp(1.0, kRawBits - 1 - kFractionalBits) - std::ldexp(1.0, -kFractionalBits);
    const double minValue = -std::ldexp(1.0, kRawBits - 1 - kFractionalBits);
    const std::array<std::pair<double, double>, 4> patterns{
        {{0, 0}, {maxValue, maxValue}, {minValue, minValue}, {maxValue, minValue}}};
    for (const auto& [even, odd] : patterns) {
      for (int i = 0; i < nInputs; ++i)
        emulators.input(i) = (i % 2 == 0) ? even : odd;
      const bool same = emulators.evaluate();
      INFO(emulators.describe());
      REQUIRE(same);
    }
  }

  SECTION("inputs outside the exact domain are rejected by the alkaid emulator") {
    // |x| >= 2^29: the ALIR model would wrap where hls4ml saturates
    emulators.input(0) = std::ldexp(1.0, 30);
    REQUIRE_THROWS_AS(emulators.evaluate(), cms::Exception);
    // |x| < 2^29 but 61 significant bits: not exactly representable in double
    emulators.input(0) = 0;
    emulators.input(5) =
        L1TSC4NGComparison::Value(std::ldexp(1.0, 28)) + L1TSC4NGComparison::Value(std::ldexp(1.0, -32));
    REQUIRE_THROWS_AS(emulators.evaluate(), cms::Exception);
  }

  SECTION("random raw bit patterns over the input domain") {
    std::mt19937_64 generator(kSeed);
    std::uniform_int_distribution<int64_t> raw(-(int64_t(1) << (kRawBits - 1)), (int64_t(1) << (kRawBits - 1)) - 1);
    unsigned int mismatches = 0;
    for (unsigned int sample = 0; sample < kL1TSC4NGRandomSamples; ++sample) {
      for (int i = 0; i < nInputs; ++i)
        emulators.input(i) = std::ldexp(static_cast<double>(raw(generator)), -kFractionalBits);
      if (!emulators.evaluate() && mismatches++ == 0)
        UNSCOPED_INFO("first mismatch, random sample " << sample << ": " << emulators.describe());
    }
    REQUIRE(mismatches == 0);
  }
}

// Phase 3 microbenchmark (hidden: run with `testAlkaidVsHls4ml "[.bench]"`). Per model, in a fresh process:
// startup time of loading each back end (hls4ml: ModelLoader + load_model, i.e. dlopen; alkaid: AlkaidModel, i.e.
// JSON+gzip parse and transpile) and the resident memory it keeps (measured after 100 warm-up calls, so that the
// hls4ml library's code is paged in, and after malloc_trim, so that freed parse buffers don't count), then the mean
// wall time per jet through the interface
// the producers use (prepare_input/predict/read_result, one jet per call), and for alkaid also with 8 jets per call
// (AlkaidModel::evaluate). Inputs cycle through the golden vectors. One line per model, prefixed "BENCH".
namespace {

  constexpr unsigned int kBenchSamples = 64;  // distinct golden inputs cycled through
  constexpr size_t kBatch = l1tpf::AlkaidModel::kBatchSize;

  double residentMegabytes() {
    std::ifstream statm("/proc/self/statm");
    long pages = 0, resident = 0;
    statm >> pages >> resident;
    return resident * static_cast<double>(sysconf(_SC_PAGESIZE)) / (1024. * 1024.);
  }

  template <typename Function>
  double millisecondsFor(Function&& function) {
    const auto start = std::chrono::steady_clock::now();
    function();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }

  // Mean microseconds per call of function(i) over calls calls.
  template <typename Function>
  double microsecondsPerCall(unsigned int calls, Function&& function) {
    return 1000. * millisecondsFor([&] {
             for (unsigned int i = 0; i < calls; ++i)
               function(i);
           }) /
           calls;
  }

  struct Startup {
    double milliseconds, megabytes;
  };

  // load: creates the model; warmUp: runs it a few times. Memory = resident increase after both, with freed heap
  // returned to the OS first.
  template <typename Load, typename WarmUp>
  Startup measureStartup(Load&& load, WarmUp&& warmUp) {
    malloc_trim(0);
    const double before = residentMegabytes();
    const double milliseconds = millisecondsFor(load);
    warmUp();
    malloc_trim(0);
    return {milliseconds, residentMegabytes() - before};
  }

  constexpr unsigned int kWarmUpCalls = 100;

  void printBench(const std::string& name,
                  const Startup& hls4ml,
                  const Startup& alkaid,
                  double hls4mlPerJet,
                  double alkaidPerJet,
                  double alkaidBatchedPerJet) {
    std::cout << std::fixed << std::setprecision(2) << "BENCH " << name << " startup_ms hls4ml=" << hls4ml.milliseconds
              << " alkaid=" << alkaid.milliseconds << " rss_MB hls4ml=" << hls4ml.megabytes
              << " alkaid=" << alkaid.megabytes << " us_per_jet hls4ml=" << hls4mlPerJet
              << " alkaid_B1=" << alkaidPerJet << " alkaid_B8=" << alkaidBatchedPerJet << std::endl;
  }

  // Mean microseconds per jet of AlkaidModel::evaluate with kBatch jets per call.
  double batchedPerJet(const l1tpf::AlkaidModel& model, const l1tpf::test::AlkaidGolden& golden, unsigned int calls) {
    const size_t nIn = model.nInputs(), nOut = model.nOutputs();
    const size_t samples = std::min<size_t>(golden.nSamples, kBenchSamples) / kBatch * kBatch;
    std::vector<double> outputs(kBatch * nOut);
    std::vector<int64_t> scratch;
    return microsecondsPerCall(calls / kBatch,
                               [&](unsigned int i) {
                                 const size_t first = (i * kBatch) % samples;
                                 model.evaluate(
                                     std::span(golden.inputs).subspan(first * nIn, kBatch * nIn), outputs, scratch);
                               }) /
           kBatch;
  }

  template <typename InputType, typename OutputType>
  void benchmarkArrayModel(const ModelUnderTest& model, unsigned int calls) {
    const std::string path = edm::FileInPath(kDataDir + model.alkaidSymbol + ".json.gz").fullPath();
    const auto golden = l1tpf::test::readAlkaidGolden(kDataDir + model.alkaidSymbol + ".golden.bin");
    const unsigned int samples = std::min<unsigned int>(golden.nSamples, kBenchSamples);
    std::vector<std::vector<InputType>> inputs(samples, std::vector<InputType>(model.hls4mlWrapperInputs));
    for (unsigned int s = 0; s < samples; ++s)
      for (unsigned int i = 0; i < model.nInputs; ++i)
        inputs[s][i] = golden.inputs[s * model.nInputs + i];
    std::vector<OutputType> outputs(model.nOutputs);
    const auto perJet = [&](hls4mlEmulator::Model& emulator, unsigned int n) {
      return microsecondsPerCall(n, [&](unsigned int i) {
        emulator.prepare_input(inputs[i % samples].data());
        emulator.predict();
        emulator.read_result(outputs.data());
      });
    };

    hls4mlEmulator::ModelLoader loader(model.hls4mlName);  // must outlive hls4ml
    std::shared_ptr<hls4mlEmulator::Model> hls4ml;
    std::shared_ptr<const l1tpf::AlkaidModel> alkaidModel;
    std::shared_ptr<hls4mlEmulator::Model> alkaid;
    const Startup hls4mlStartup =
        measureStartup([&] { hls4ml = loader.load_model(); }, [&] { perJet(*hls4ml, kWarmUpCalls); });
    const Startup alkaidStartup =
        measureStartup([&] { alkaidModel = std::make_shared<const l1tpf::AlkaidModel>(path); },
                       [&] {
                         alkaid = std::make_shared<l1tpf::AlkaidEmulatorModel<InputType, OutputType>>(alkaidModel);
                         perJet(*alkaid, kWarmUpCalls);
                       });
    const double hls4mlPerJet = perJet(*hls4ml, calls);
    const double alkaidPerJet = perJet(*alkaid, calls);
    printBench(model.alkaidSymbol,
               hls4mlStartup,
               alkaidStartup,
               hls4mlPerJet,
               alkaidPerJet,
               batchedPerJet(*alkaidModel, golden, calls));
  }

  void benchmarkL1TSC4NG(unsigned int calls) {
    using Value = L1TSC4NGJet::inputtype;
    const std::string path = edm::FileInPath(kDataDir + "L1TSC4NGJetModel_v2_0_0.json.gz").fullPath();
    const auto golden = l1tpf::test::readAlkaidGolden(kDataDir + "L1TSC4NGJetModel_v2_0_0.golden.bin");
    const unsigned int samples = std::min<unsigned int>(golden.nSamples, kBenchSamples);
    constexpr int nCandidates = L1TSC4NGJetID::N_candidate_inputs, nJet = L1TSC4NGJetID::N_jet_inputs;
    std::vector<std::array<Value, nCandidates>> candidates(samples);
    std::vector<std::array<Value, nJet>> jets(samples);
    for (unsigned int s = 0; s < samples; ++s) {
      for (int i = 0; i < nCandidates; ++i)
        candidates[s][i] = golden.inputs[s * (nCandidates + nJet) + i];
      for (int i = 0; i < nJet; ++i)
        jets[s][i] = golden.inputs[s * (nCandidates + nJet) + nCandidates + i];
    }
    // Room for all the class scores the hls4ml wrapper writes (F2).
    std::array<Value, L1TSC4NGJetID::kModelClassOutputs> classes{};
    std::array<Value, L1TSC4NGJetID::N_regression_outputs> regression{};
    const auto perJet = [&](hls4mlEmulator::Model& emulator, unsigned int n) {
      return microsecondsPerCall(n, [&](unsigned int i) {
        const unsigned int s = i % samples;
        ModelInputs modelInputs{
            candidates[s].data(), jets[s].data(), L1TSC4NGJetID::N_candidate_features, L1TSC4NGJetID::N_jet_inputs};
        ModelOutputs modelOutputs{classes.data(), regression.data()};
        emulator.prepare_input(modelInputs);
        emulator.predict();
        emulator.read_result(&modelOutputs);
      });
    };

    hls4mlEmulator::ModelLoader loader("L1TSC4NGJetModel_v2_0_0");  // must outlive hls4ml
    std::shared_ptr<hls4mlEmulator::Model> hls4ml;
    std::shared_ptr<const l1tpf::AlkaidModel> alkaidModel;
    std::shared_ptr<hls4mlEmulator::Model> alkaid;
    const Startup hls4mlStartup =
        measureStartup([&] { hls4ml = loader.load_model(); }, [&] { perJet(*hls4ml, kWarmUpCalls); });
    const Startup alkaidStartup =
        measureStartup([&] { alkaidModel = std::make_shared<const l1tpf::AlkaidModel>(path); },
                       [&] {
                         alkaid = L1TSC4NGJetID::makeAlkaidEmulator(alkaidModel);
                         perJet(*alkaid, kWarmUpCalls);
                       });
    const double hls4mlPerJet = perJet(*hls4ml, calls);
    const double alkaidPerJet = perJet(*alkaid, calls);
    printBench("L1TSC4NGJetModel_v2_0_0",
               hls4mlStartup,
               alkaidStartup,
               hls4mlPerJet,
               alkaidPerJet,
               batchedPerJet(*alkaidModel, golden, calls));
  }

}  // namespace

TEST_CASE("Phase 3 microbenchmark: startup, memory and time per jet", "[.bench]") {
  benchmarkArrayModel<JetId::NNInputType, JetId::NNOutputType>({.hls4mlName = "TOoLLiP_v3",
                                                                .alkaidSymbol = "TOoLLiP_v3_v3_1_0",
                                                                .nInputs = JetId::kNNInputs,
                                                                .nOutputs = 1,
                                                                .hls4mlWrapperInputs = 140,
                                                                .randomSamples = 0},
                                                               100000);
  benchmarkArrayModel<L1TSC82ProngJetID::inputtype, L1TSC82ProngJetID::prong_score>(
      {.hls4mlName = "L1TSC82ProngJetModel_v0",
       .alkaidSymbol = "L1TSC82ProngJetModel_v0",
       .nInputs = L1TSC82ProngJetID::kNNInputs,
       .nOutputs = L1TSC82ProngJetID::kNNOutputs,
       .hls4mlWrapperInputs = L1TSC82ProngJetID::kNNInputs,
       .randomSamples = 0},
      5000);
  benchmarkL1TSC4NG(20000);
}
