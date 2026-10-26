#ifndef L1Trigger_Phase2L1ParticleFlow_AlkaidEmulatorModel_h
#define L1Trigger_Phase2L1ParticleFlow_AlkaidEmulatorModel_h

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "hls4ml/emulator.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/AlkaidModel.h"

namespace l1tpf {

  // Drop-in replacement for an hls4ml compiled emulator (hls4mlEmulator::Model), backed by an AlkaidModel. It lets
  // a producer switch back ends without touching its physics code: prepare_input/read_result take the same ap_fixed
  // array pointers the hls4ml emulator wrapper any_casts to (InputType*, OutputType*), and the model's input/output
  // counts are fixed by the ALIR file instead of by the wrapper.
  // Values pass through double, which is exact for ap_fixed types of up to 53 significant bits.
  // Stateful like the hls4ml models (one instance per stream); the AlkaidModel itself can be shared.
  template <typename InputType, typename OutputType>
  class AlkaidEmulatorModel : public hls4mlEmulator::Model {
  public:
    explicit AlkaidEmulatorModel(std::shared_ptr<const AlkaidModel> model)
        : model_(std::move(model)), inputs_(model_->nInputs()), outputs_(model_->nOutputs()) {}

    void prepare_input(std::any input) override {
      const InputType* values = std::any_cast<InputType*>(input);
      for (size_t i = 0; i < inputs_.size(); ++i)
        inputs_[i] = values[i].to_double();
    }

    void predict() override { model_->evaluate(inputs_, outputs_, scratch_); }

    void read_result(std::any result) override {
      OutputType* values = std::any_cast<OutputType*>(result);
      for (size_t i = 0; i < outputs_.size(); ++i)
        values[i] = outputs_[i];
    }

  private:
    std::shared_ptr<const AlkaidModel> model_;
    std::vector<double> inputs_;
    std::vector<double> outputs_;
    std::vector<int64_t> scratch_;
  };

}  // namespace l1tpf

#endif
