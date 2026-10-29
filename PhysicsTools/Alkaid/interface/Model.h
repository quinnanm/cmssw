#ifndef PhysicsTools_Alkaid_Model_h
#define PhysicsTools_Alkaid_Model_h

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace cms::alkaid {

  // A fixed-point neural network stored as an alkaid ALIR program (*.json or *.json.gz), evaluated bit-exactly by the
  // alkaid runtime. The same program is what alkaid turns into firmware, so the emulation matches it by construction.
  //
  // The program is loaded once and never modified: one instance can be shared by all streams (e.g. in an
  // edm::GlobalCache), as long as every caller passes its own Scratch to evaluate().
  class Model {
  public:
    // Per-caller working memory for evaluate(); reuse it across calls (it grows once, then stays).
    struct Scratch {
      std::vector<int64_t> buffer;
    };

    // fullPath: e.g. edm::FileInPath(...).fullPath(). Throws cms::Exception if the file is missing, corrupt or has an
    // unsupported ALIR spec version, or if the model doesn't have exactly nInputs inputs and nOutputs outputs (the
    // sizes the caller fills and reads).
    Model(const std::string& fullPath, size_t nInputs, size_t nOutputs);
    ~Model();

    // Held in place (e.g. by the GlobalCache's unique_ptr): neither copyable nor movable.
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;

    size_t nInputs() const { return nInputs_; }
    size_t nOutputs() const { return nOutputs_; }
    const std::string& path() const { return path_; }

    // Evaluates n = inputs.size() / nInputs() samples. Both spans are sample-major: sample i reads
    // inputs[i * nInputs(), (i + 1) * nInputs()) and writes outputs[i * nOutputs(), (i + 1) * nOutputs()).
    // Full groups of kBatchSize samples are evaluated together, which is several times cheaper per sample than one
    // at a time; the remainder is evaluated one by one. Throws cms::Exception on inconsistent sizes.
    void evaluate(std::span<const double> inputs, std::span<double> outputs, Scratch& scratch) const;

    static constexpr size_t kBatchSize = 8;

  private:
    // The alkaid interpreter (defined in Model.cc, so that clients don't see the runtime's headers)
    struct Impl;
    static std::unique_ptr<const Impl> load(const std::string& fullPath);

    const std::string path_;
    const std::unique_ptr<const Impl> impl_;
    const size_t nInputs_;
    const size_t nOutputs_;
    const size_t nSlots_;
  };

}  // namespace cms::alkaid

#endif
