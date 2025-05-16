import argparse
import subprocess
import shutil
import os
import sys

#example usage: python3 testrepack.py -i l1t_L1REPACK_Full.py --type mu

def main():
    parser = argparse.ArgumentParser(description="add testing output to input cmsDriver commands")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input file name')
    parser.add_argument('-t', '--type', required=True, choices=['mu', 'jet', 'eg', 'met', 'tau'],
                        help='Type of trigger condition to insert (mu, jet, eg, met, tau)')
    parser.add_argument('-o', '--output', help='Optional output file name (default: input_basename + _test.py)')


    args = parser.parse_args()

    # Trigger type mapping
    trigger_map = {
        'mu'  : 'L1_SingleMu22*',
        'jet' : 'HLT_HT*',
        'met' : 'MET*',
        'eg'  : 'EGAMMA*',
        'tau' : 'TAU*'
    }
    trigger_pattern = trigger_map[args.type]
    

    # Derive default output filename if not given
    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_file = f"{base}_{args.type}_test.py"

    log_file = output_file.replace(".py", ".log")

    #delete output and log files if they already exist
    for f in [output_file, log_file]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed existing file: {f}")
    
    #copy to new file
    shutil.copy(args.input, output_file)
    print(f"Copied {args.input} to {output_file}")
        
    block = f'''\nprocess.options.wantSummary = True

process.l1tGtStage2Digis1 = cms.EDProducer( "L1TRawToDigi",
    FedIds = cms.vint32( 1404 ),
    Setup = cms.string( "stage2::GTSetup" ),
    FWId = cms.uint32( 0 ),
    DmxFWId = cms.uint32( 0 ),
    FWOverride = cms.bool( False ),
    TMTCheck = cms.bool( True ),
    CTP7 = cms.untracked.bool( False ),
    MTF7 = cms.untracked.bool( False ),
    InputLabel = cms.InputTag( "rawDataCollector::@skipCurrentProcess" ),
    lenSlinkHeader = cms.untracked.int32( 8 ),
    lenSlinkTrailer = cms.untracked.int32( 8 ),
    lenAMCHeader = cms.untracked.int32( 8 ),
    lenAMCTrailer = cms.untracked.int32( 0 ),
    lenAMC13Header = cms.untracked.int32( 8 ),
    lenAMC13Trailer = cms.untracked.int32( 8 ),
    debug = cms.untracked.bool( False ),
    MinFeds = cms.uint32( 0 )
)

process.l1tMonitor1 = cms.EDFilter( "TriggerResultsFilter",
    usePathStatus = cms.bool( False ),
    hltResults = cms.InputTag( "" ),
    l1tResults = cms.InputTag( "l1tGtStage2Digis1" ),
    l1tIgnoreMaskAndPrescale = cms.bool( True ),
    throw = cms.bool( True ),
    triggerConditions = cms.vstring( '{trigger_pattern}' )
)

process.L1TMonitorPath1 = cms.Path(
    process.l1tGtStage2Digis1
  + process.l1tMonitor1
)

process.schedule.append( process.L1TMonitorPath1 )

process.l1tGtStage2Digis2 = process.l1tGtStage2Digis1.clone(
    InputLabel = "rawDataCollector::@currentProcess"
)

process.l1tMonitor2 = process.l1tMonitor1.clone(
    l1tResults = "l1tGtStage2Digis2"
)

process.L1TMonitorPath2 = cms.Path(
    process.rawDataCollector
  + process.l1tGtStage2Digis2
  + process.l1tMonitor2
)

process.schedule.append( process.L1TMonitorPath2 )
'''

    with open(output_file, 'a') as f:
        f.write(block)

    print(f"Appended block with trigger type '{args.type}' to {output_file}")

        # Run cmsRun and save output to log file
    print(f"\nRunning cmsRun {output_file} and writing output to {log_file}...")
    try:
        with open(log_file, 'w') as logf:
            subprocess.run(['cmsRun', output_file], stdout=logf, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"cmsRun failed with exit code {e.returncode}. See {log_file} for details.")
        sys.exit(1)

    # Grep for MonitorPath lines
    print("\nFirst two MonitorPath lines from log: 1 firmware 2 L1REPACK")
    try:
        result = subprocess.run(['grep', 'MonitorPath', log_file], stdout=subprocess.PIPE, text=True)
        lines = result.stdout.strip().split('\n')[:2]
        for line in lines:
            print(line)
    except Exception as e:
        print(f"Failed to grep MonitorPath: {e}")
    
if __name__ == '__main__':
    main()
