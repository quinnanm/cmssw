#ifndef PhysicsTools_Alkaid_GoldenVectors_h
#define PhysicsTools_Alkaid_GoldenVectors_h

// Golden vectors: reference inputs and outputs that ship with an alkaid model, so packages can check in their unit
// tests that the model in the release still reproduces its reference (e.g. the hls4ml emulator it replaced, or the
// training framework's bit-accurate model), without that reference being available.
//
// File format (*.golden.bin, little endian): "ALKGOLD1", uint64 nSamples, nInputs, nOutputs, then nSamples * nInputs
// input doubles and nSamples * nOutputs output doubles, sample-major.

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace cms::alkaid {

  struct GoldenVectors {
    uint64_t nSamples = 0, nInputs = 0, nOutputs = 0;
    std::vector<double> inputs, outputs;
  };

  // fullPath: e.g. edm::FileInPath(...).fullPath(). Throws cms::Exception if it isn't a complete golden-vector file.
  GoldenVectors readGoldenVectors(const std::string& fullPath);

  // Exact comparison of the bit patterns: no tolerance, and -0.0 != 0.0.
  bool bitIdentical(std::span<const double> a, std::span<const double> b);

}  // namespace cms::alkaid

#endif
