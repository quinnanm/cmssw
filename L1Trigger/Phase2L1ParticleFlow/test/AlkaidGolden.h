#ifndef L1Trigger_Phase2L1ParticleFlow_test_AlkaidGolden_h
#define L1Trigger_Phase2L1ParticleFlow_test_AlkaidGolden_h

// Test helpers shared by the alkaid unit tests: golden-vector files and exact comparison.
// Golden format (written by the alkaid evaluation's export_golden.py): "ALKGOLD1", uint64 n, nIn, nOut (little
// endian), then n*nIn input and n*nOut output doubles, sample-major.

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace l1tpf::test {

  struct AlkaidGolden {
    uint64_t nSamples = 0, nInputs = 0, nOutputs = 0;
    std::vector<double> inputs, outputs;
  };

  // relativePath: path in the CMSSW search path, e.g. "L1Trigger/Phase2L1ParticleFlow/test/alkaid/x.golden.bin".
  inline AlkaidGolden readAlkaidGolden(const std::string& relativePath) {
    const std::string path = edm::FileInPath(relativePath).fullPath();
    std::ifstream file(path, std::ios::binary);
    std::array<char, 8> magic{};
    file.read(magic.data(), magic.size());
    if (!file || std::memcmp(magic.data(), "ALKGOLD1", magic.size()) != 0)
      throw cms::Exception("AlkaidGolden") << path << ": not a golden-vector file";
    AlkaidGolden golden;
    std::array<uint64_t, 3> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));  // little-endian hosts only
    golden.nSamples = header[0];
    golden.nInputs = header[1];
    golden.nOutputs = header[2];
    golden.inputs.resize(golden.nSamples * golden.nInputs);
    golden.outputs.resize(golden.nSamples * golden.nOutputs);
    file.read(reinterpret_cast<char*>(golden.inputs.data()), golden.inputs.size() * sizeof(double));
    file.read(reinterpret_cast<char*>(golden.outputs.data()), golden.outputs.size() * sizeof(double));
    if (!file)
      throw cms::Exception("AlkaidGolden") << path << ": truncated";
    return golden;
  }

  // Exact comparison of the bit patterns: no tolerance, and -0.0 != 0.0.
  inline bool bitIdentical(std::span<const double> a, std::span<const double> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
  }

}  // namespace l1tpf::test

#endif
