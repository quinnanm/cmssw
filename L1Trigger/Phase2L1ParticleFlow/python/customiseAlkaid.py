"""Customisations for the alkaid NN emulator back end of the Phase-2 L1 NN jet producers.

customiseAlkaidBackend(process)     run the producers with backend = "alkaid" instead of "hls4ml"; a no-op for
                                    steps without them, so it can be applied to every step of a workflow
                                    (runTheMatrix.py ... --command "--customise ...customiseAlkaidBackend")
customiseAlkaidComparison(process)  keep them on hls4ml, add an alkaid copy of each (label + "Alkaid") and
                                    AlkaidComparisonAnalyzer, which requires identical outputs jet by jet
e.g. cmsDriver.py ... --customise L1Trigger/Phase2L1ParticleFlow/customiseAlkaid.customiseAlkaidComparison
"""
import FWCore.ParameterSet.Config as cms

# Producer module labels in the Phase-2 L1 emulation (L1Trigger/Configuration/python/SimL1Emulator_cff.py).
_PRODUCERS = (
    "l1tTOoLLiPProducer",
    "l1tTOoLLiPProducerCorrectedEmulator",
    "l1tSC4NGJetProducer",
    "l1tSC82ProngJetProducer",
)
# Jet-collection products of the producers that store tagged jets (TOoLLiPProducer stores a ValueMap instead).
_JET_COLLECTIONS = {
    "L1TSC4NGJetProducer": "l1tSC4NGJets",
    "L1TSC82ProngJetProducer": "l1tSC82ProngJets",
}
_TOOLLIP_VALUE_MAP = "L1PFLLPJets"


def _producers(process):
    missing = [label for label in _PRODUCERS if not hasattr(process, label)]
    if missing:
        raise RuntimeError("customiseAlkaid: the process has no %s (is the Phase-2 L1 step included?)" % ", ".join(missing))
    return [(label, getattr(process, label)) for label in _PRODUCERS]


def customiseAlkaidBackend(process):
    for label in _PRODUCERS:
        if hasattr(process, label):
            getattr(process, label).backend = "alkaid"
    return process


def customiseAlkaidComparison(process):
    valueMaps = cms.VPSet()
    jetCollections = cms.VPSet()
    alkaidProducers = cms.Task()
    for label, producer in _producers(process):
        alkaidLabel = label + "Alkaid"
        setattr(process, alkaidLabel, producer.clone(backend = "alkaid"))
        alkaidProducers.add(getattr(process, alkaidLabel))
        if producer.type_() == "TOoLLiPProducer":
            valueMaps.append(cms.PSet(
                name = cms.string(label),
                jets = producer.jets,
                reference = cms.InputTag(label, _TOOLLIP_VALUE_MAP),
                test = cms.InputTag(alkaidLabel, _TOOLLIP_VALUE_MAP),
            ))
        else:
            instance = _JET_COLLECTIONS[producer.type_()]
            jetCollections.append(cms.PSet(
                name = cms.string(label),
                reference = cms.InputTag(label, instance),
                test = cms.InputTag(alkaidLabel, instance),
            ))
    process.alkaidComparison = cms.EDAnalyzer("AlkaidComparisonAnalyzer",
        valueMaps = valueMaps,
        jetCollections = jetCollections,
        failOnMismatch = cms.bool(True),
    )
    process.alkaidComparisonPath = cms.EndPath(process.alkaidComparison, alkaidProducers)
    if process.schedule is not None:
        process.schedule.append(process.alkaidComparisonPath)
    return process
