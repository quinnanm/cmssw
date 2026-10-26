"""Customisations for the alkaid NN emulator back end of the Phase-2 L1 NN jet producers.

customiseAlkaidBackend(process)     run the producers with backend = "alkaid" instead of "hls4ml"; a no-op for
                                    steps without them, so it can be applied to every step of a workflow
                                    (runTheMatrix.py ... --command "--customise ...customiseAlkaidBackend")
e.g. cmsDriver.py ... --customise L1Trigger/Phase2L1ParticleFlow/customiseAlkaid.customiseAlkaidBackend
"""

# Producer module labels in the Phase-2 L1 emulation (L1Trigger/Configuration/python/SimL1Emulator_cff.py).
_PRODUCERS = (
    "l1tTOoLLiPProducer",
    "l1tTOoLLiPProducerCorrectedEmulator",
    "l1tSC4NGJetProducer",
    "l1tSC82ProngJetProducer",
)


def customiseAlkaidBackend(process):
    for label in _PRODUCERS:
        if hasattr(process, label):
            getattr(process, label).backend = "alkaid"
    return process
