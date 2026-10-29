# alkaid models of the Phase-2 L1 NN jet taggers (temporary location)

The NN models of `TOoLLiPProducer`, `L1TSC82ProngJetProducer` and `L1TSC4NGJetProducer` (parameter `model`, read with
`edm::FileInPath` and evaluated with `PhysicsTools/Alkaid`). They replace the hls4ml compiled emulators of the same
models and give bit-identical outputs. This directory is a **temporary location**: where these files finally live
(e.g. `cms-data/L1Trigger-Phase2L1ParticleFlow`) is not decided yet; the `FileInPath` defaults don't need to change
when they move there.

| file | replaces hls4ml model | from | ALIR ops | sha256 of the decompressed JSON |
|---|---|---|---|---|
| `TOoLLiP_v3_v3_1_0.json.gz` | `TOoLLiP_v3` | `cms-hls4ml/TOoLLiP@v3.1.0` | 3 393 | `a5b364c04c9929d3d73c12852c1ce53e5a6b64333ceb53b0f78930397db90a0a` |
| `L1TSC82ProngJetModel_v0.json.gz` | `L1TSC82ProngJetModel_v0` | `cms-hls4ml/L1TSC82ProngJetModel@v0.0.1` | 174 996 | `748be456748822f1ebda4e7d65e19da3a83bc92174431ab6319206babe35c77b` |
| `L1TSC4NGJetModel_v2_0_0.json.gz` | `L1TSC4NGJetModel_v2_0_0` | `cms-hls4ml/L1TSC4NGJetModel@v2.1.0` | 30 868 | `81479ed60bc246e0f6873f969ba046be42968c2dc4492e26b95340c3eae32238` |

- **Format:** alkaid ALIR, spec version 4, gzipped JSON. The `.gz` bytes change with every re-trace (gzip stores a
  timestamp); identify a model by the hash of its decompressed JSON (`zcat file | sha256sum`).
- **How they were made:** traced with alkaid 0.8.1 (plus a workaround for a float32 constant-encoding bug in its
  tracer) from hand-written replays of the hls4ml projects at the tags above. Each replay reads the project's
  `defines.h` types and weights and quantizes every layer exactly as hls4ml does.
- **Validation:** bit-identical to the compiled hls4ml emulator on 1M random + 600k edge-case inputs per model
  (standalone), on random inputs inside CMSSW through the producers' own types, and jet by jet on 9,000 TTbar PU200
  events (TOoLLiP 93.8k, L1TSC4NG 138.5k, L1TSC82 120.9k jets, 0 differences); done with a version of these producers
  that could run both back ends side by side.
- **Input domain:** exact for |input| < 2²⁴ (TOoLLiP, L1TSC82: their inputs are quantized to 12- and 24-bit types
  first, far inside) and < 2²⁹ (L1TSC4NG v2_0_0: `L1TSC4NGJetID` checks every input and throws outside). Physical
  inputs are far below.

