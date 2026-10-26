#ifndef L1Trigger_Phase2L1ParticleFlow_AlkaidModel_h
#define L1Trigger_Phase2L1ParticleFlow_AlkaidModel_h

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "alkaid/alir/ALIRInterpreter.hh"

namespace l1tpf {

  // Bit-exact software emulation of a fixed-point L1 NN stored as an alkaid ALIR program (*.json or *.json.gz).
  // The program is loaded once and never modified: one instance can be shared by all streams, as long as each
  // caller passes its own scratch buffer to evaluate().
  class AlkaidModel {
  public:
    // modelPath: full path of the ALIR file, e.g. edm::FileInPath::fullPath(). Throws if the file is missing,
    // corrupt, or has an unsupported ALIR spec version.
    explicit AlkaidModel(const std::string& modelPath);

    size_t nInputs() const { return interpreter_.get_n_in(); }
    size_t nOutputs() const { return interpreter_.get_n_out(); }

    // Evaluates inputs.size() / nInputs() samples. Both spans are sample-major: sample i reads
    // inputs[i * nInputs(), (i + 1) * nInputs()) and writes outputs[i * nOutputs(), (i + 1) * nOutputs()).
    // Full groups of kBatchSize samples are evaluated together (several times cheaper per sample than one at a
    // time), the remainder one by one. scratch is resized as needed and can be reused across calls.
    void evaluate(std::span<const double> inputs, std::span<double> outputs, std::vector<int64_t>& scratch) const;

    static constexpr size_t kBatchSize = 8;

  private:
    alir::ALIRInterpreter interpreter_;
  };

}  // namespace l1tpf

#endif
