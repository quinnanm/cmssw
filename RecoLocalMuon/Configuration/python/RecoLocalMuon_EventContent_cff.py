# The following comments couldn't be translated into the new config version:

import FWCore.ParameterSet.Config as cms

# Full Event content 
RecoLocalMuonFEVT = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_dt1DRecHits_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*')
)
# RECO content
RecoLocalMuonRECO = cms.PSet(
    outputCommands = cms.untracked.vstring('keep *_dt1DRecHits_*_*', 
        'keep *_dt4DSegments_*_*', 
        'keep *_dt1DCosmicRecHits_*_*',
        'keep *_dt4DCosmicSegments_*_*',
        'keep *_csc2DRecHits_*_*', 
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*')
)
# AOD content
RecoLocalMuonAOD = cms.PSet(
    outputCommands = cms.untracked.vstring(
        'keep *_dt4DSegments_*_*', 
        'keep *_dt4DCosmicSegments_*_*',
        'keep *_cscSegments_*_*', 
        'keep *_rpcRecHits_*_*')
)
from Configuration.StandardSequences.Eras import eras
def _updateOutput( era, outputPSets, commands):
   for o in outputPSets:
      era.toModify( o, outputCommands = o.outputCommands + commands )

_outputs = [RecoLocalMuonFEVT, RecoLocalMuonRECO, RecoLocalMuonAOD]
_updateOutput( eras.run3_GEM, _outputs, ['keep *_gemRecHits_*_*'] )
_updateOutput(eras.phase2_muon, _outputs, ['keep *_me0RecHits_*_*', 'keep *_me0Segments_*_*'])
