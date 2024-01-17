#ifndef L1Trigger_L1TGlobal_AXOL1TLCondition_h
#define L1Trigger_L1TGlobal_AXOL1TLCondition_h

/**
 * \class AXOL1TLCondition
 *
 * Description: evaluation of a CondAXOL1TL condition.
 */

// system include files
#include <iosfwd>
#include <string>
#include "ap_fixed.h"
#include "hls4ml/emulator.h"

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"
#include "DataFormats/L1Trigger/interface/L1Candidate.h"

// forward declarations
class GlobalCondition;
class AXOL1TLTemplate;

namespace l1t {

  class L1Candidate;
  class GlobalBoard;

  // class declaration
  class AXOL1TLCondition : public ConditionEvaluation {
  public:
    /// constructors
    ///     default
    AXOL1TLCondition();

    ///     from base template condition (from event setup usually)
    // AXOL1TLCondition(const GlobalCondition*, const GlobalBoard*, const hls4mlEmulator::Model*);
    AXOL1TLCondition(const GlobalCondition*, const GlobalBoard*, const std::shared_ptr<hls4mlEmulator::Model>);

    // copy constructor
    AXOL1TLCondition(const AXOL1TLCondition&);
    // destructor
    ~AXOL1TLCondition() override;

    // assign operator
    AXOL1TLCondition& operator=(const AXOL1TLCondition&);

    /// the core function to check if the condition matches
    const bool evaluateCondition(const int bxEval) const override;

    /// print condition
    void print(std::ostream& myCout) const override;

    ///   get / set the pointer to a Condition
    inline const AXOL1TLTemplate* gtAXOL1TLTemplate() const { return m_gtAXOL1TLTemplate; }

    void setGtAXOL1TLTemplate(const AXOL1TLTemplate*);

    ///   get / set the pointer to GTL
    inline const GlobalBoard* gtGTB() const { return m_gtGTB; }

    void setuGtB(const GlobalBoard*);

    ///   get / set the pointer to model 
    // inline const hls4mlEmulator::Model* gtAXOL1TLmodel() const { return m_gtAXOL1TLmodel; }

    // void setGtAXOL1TLModel(const hls4mlEmulator::Model*);

    inline const std::shared_ptr<hls4mlEmulator::Model> gtAXOL1TLmodel() const { return m_gtAXOL1TLmodel; }

    void setGtAXOL1TLModel(const std::shared_ptr<hls4mlEmulator::Model>);

    // //get/set AXOL1TL model version from global board
    // inline const std::string gtModelVerion() const { return m_AXOL1TLmodelversion; }
    
    // void setModelVersion(const std::string modelversionname);

  private:
    /// copy function for copy constructor and operator=
    void copy(const AXOL1TLCondition& cp);

    /// pointer to a AXOL1TLTemplate
    const AXOL1TLTemplate* m_gtAXOL1TLTemplate;

    /// pointer to uGt GlobalBoard, to be able to get the trigger objects
    const GlobalBoard* m_gtGTB;

    ///pointer to preloaded model 
    std::shared_ptr<hls4mlEmulator::Model> m_gtAXOL1TLmodel;
    // const hls4mlEmulator::Model* m_gtAXOL1TLmodel;

    //to set modelversion from globalboard<-globalproducer<-config
    // std::string m_AXOL1TLmodelversion = "NONE";
    
    //for loading model in globalboard
    // std::shared_ptr<hls4mlEmulator::Model> m_AXOL1TLmodel;
    // bool m_modelloaded = false;
    
  };

}  // namespace l1t
#endif
