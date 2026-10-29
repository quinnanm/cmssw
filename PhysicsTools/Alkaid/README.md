# PhysicsTools/Alkaid

Wrapper for the [alkaid](https://github.com/calad0i/alkaid) runtime (the `alkaid` external): evaluates fixed-point
neural networks stored as alkaid ALIR programs (`*.json.gz`) bit-exactly. alkaid generates the firmware of a network
from the same program, so the emulation in CMSSW matches the firmware by construction. One external serves all
models: a new model or model version is a data file, not new code.

```cpp
#include "PhysicsTools/Alkaid/interface/Model.h"

// once per job, e.g. in an edm::GlobalCache (the model is read-only and shared by all streams)
cms::alkaid::Model model(edm::FileInPath("My/Package/data/model.json.gz").fullPath(), nInputs, nOutputs);

// per call: inputs/outputs as doubles in the model's flat order, n samples at a time
cms::alkaid::Model::Scratch scratch;  // per stream / caller
model.evaluate(inputs, outputs, scratch);
```

- Inputs are quantized by the model itself to its input types; values pass through `double`, which is exact for
  fixed-point inputs of up to 53 significant bits.
- Evaluating several samples per call (e.g. all jets of an event) is several times cheaper per sample than one
  call per sample.
- `GoldenVectors.h` reads the reference vectors that ship with a model (`*.golden.bin`), for unit tests that check
  a model in the release still reproduces its reference.
- `test/testAlkaidModel` checks the external itself with two toy models (no physics model involved).
