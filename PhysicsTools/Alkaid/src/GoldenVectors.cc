#include "PhysicsTools/Alkaid/interface/GoldenVectors.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>

#include "FWCore/Utilities/interface/Exception.h"

cms::alkaid::GoldenVectors cms::alkaid::readGoldenVectors(const std::string& fullPath) {
  std::ifstream file(fullPath, std::ios::binary);
  std::array<char, 8> magic{};
  file.read(magic.data(), magic.size());
  if (!file || std::memcmp(magic.data(), "ALKGOLD1", magic.size()) != 0)
    throw cms::Exception("AlkaidGoldenVectors") << fullPath << ": not a golden-vector file";
  GoldenVectors golden;
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
    throw cms::Exception("AlkaidGoldenVectors") << fullPath << ": truncated";
  return golden;
}

bool cms::alkaid::bitIdentical(std::span<const double> a, std::span<const double> b) {
  return std::ranges::equal(
      a, b, [](double x, double y) { return std::bit_cast<uint64_t>(x) == std::bit_cast<uint64_t>(y); });
}
