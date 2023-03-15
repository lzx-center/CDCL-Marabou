
/*********************                                                        */
/*! \file SmtCore.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Parth Shah, Duligur Ibeling
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/

#include "Debug.h"
#include "DivideStrategy.h"
#include "EngineState.h"
#include "FloatUtils.h"
#include "GlobalConfiguration.h"
#include "Engine.h"
#include "MStringf.h"
#include "MarabouError.h"
#include "Options.h"
#include "PseudoImpactTracker.h"
#include "ReluConstraint.h"
#include "SmtCore.h"

SmtCore::SmtCore( Engine *engine )
        : _statistics( NULL )
        , _engine( engine )
        , _context( _engine->getContext() )
        , _needToSplit( false )
        , _constraintForSplitting( NULL )
        , _stateId( 0 )
        , _constraintViolationThreshold( Options::get()->getInt( Options::CONSTRAINT_VIOLATION_THRESHOLD ) )
        , _deepSoIRejectionThreshold( Options::get()->getInt( Options::DEEP_SOI_REJECTION_THRESHOLD ) )
        , _branchingHeuristic( Options::get()->getDivideStrategy() )
        , _scoreTracker( nullptr )
        , _numRejectedPhasePatternProposal( 0 )
{
}

SmtCore::~SmtCore()
{
    freeMemory();
}

void SmtCore::freeMemory()
{
    for ( const auto &stackEntry : _stack )
    {
        delete stackEntry->_engineState;
        delete stackEntry;
    }

    _stack.clear();
}

void SmtCore::reset()
{
    freeMemory();
    _impliedValidSplitsAtRoot.clear();
    _needToSplit = false;
    _constraintForSplitting = NULL;
    _stateId = 0;
    _constraintToViolationCount.clear();
    _numRejectedPhasePatternProposal = 0;
}

void SmtCore::reportViolatedConstraint( PiecewiseLinearConstraint *constraint )
{
    if ( !_constraintToViolationCount.exists( constraint ) )
        _constraintToViolationCount[constraint] = 0;

    ++_constraintToViolationCount[constraint];

    if ( _constraintToViolationCount[constraint] >=
         _constraintViolationThreshold )
    {
        _needToSplit = true;
        if ( !pickSplitPLConstraint() )
            // If pickSplitConstraint failed to pick one, use the native
            // relu-violation based splitting heuristic.
            _constraintForSplitting = constraint;
    }
}

unsigned SmtCore::getViolationCounts( PiecewiseLinearConstraint *constraint ) const
{
    if ( !_constraintToViolationCount.exists( constraint ) )
        return 0;

    return _constraintToViolationCount[constraint];
}

void SmtCore::initializeScoreTrackerIfNeeded( const
                                              List<PiecewiseLinearConstraint *>
                                              &plConstraints )
{
    if ( GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH )
    {
        _scoreTracker = std::unique_ptr<PseudoImpactTracker>
                ( new PseudoImpactTracker() );
        _scoreTracker->initialize( plConstraints );

        SMT_LOG( "\tTracking Pseudo Impact..." );
    }
}

void SmtCore::reportRejectedPhasePatternProposal()
{
    ++_numRejectedPhasePatternProposal;

    if ( _numRejectedPhasePatternProposal >=
         _deepSoIRejectionThreshold )
    {
        _needToSplit = true;
        _engine->applyAllBoundTightenings();
        _engine->applyAllValidConstraintCaseSplits();
        if ( !pickSplitPLConstraint() )
            // If pickSplitConstraint failed to pick one, use the native
            // relu-violation based splitting heuristic.
            _constraintForSplitting = _scoreTracker->topUnfixed();
    }
}

bool SmtCore::needToSplit() const
{
    return _needToSplit;
}

void SmtCore::performSplit()
{
    ASSERT( _needToSplit );
    _numRejectedPhasePatternProposal = 0;
    // Maybe the constraint has already become inactive - if so, ignore
    if ( !_constraintForSplitting->isActive() )
    {
        _needToSplit = false;
        _constraintToViolationCount[_constraintForSplitting] = 0;
        _constraintForSplitting = nullptr;
        return;
    }

    struct timespec start = TimeUtils::sampleMicro();

    ASSERT( _constraintForSplitting->isActive() );
    _needToSplit = false;

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_SPLITS );
        _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );
    }

    // Before storing the state of the engine, we:
    //   1. Obtain the splits.
    //   2. Disable the constraint, so that it is marked as disbaled in the EngineState.
    List<PiecewiseLinearCaseSplit> splits = _constraintForSplitting->getCaseSplits();
    ASSERT( !splits.empty() );
    ASSERT( splits.size() >= 2 ); // Not really necessary, can add code to handle this case.
    _constraintForSplitting->setActiveConstraint( false );

    // Obtain the current state of the engine
    EngineState *stateBeforeSplits = new EngineState;
    stateBeforeSplits->_stateId = _stateId;
    ++_stateId;
    _engine->storeState( *stateBeforeSplits,
                         TableauStateStorageLevel::STORE_BOUNDS_ONLY );
    _engine->preContextPushHook();
    pushContext();
    SmtStackEntry *stackEntry = new SmtStackEntry;
    // Perform the first split: add bounds and equations
    List<PiecewiseLinearCaseSplit>::iterator split = splits.begin();
    ASSERT( split->getEquations().size() == 0 );
    _engine->applySplit( *split );
    stackEntry->_activeSplit = *split;

    // Store the remaining splits on the stack, for later
    stackEntry->_engineState = stateBeforeSplits;
    ++split;
    while ( split != splits.end() )
    {
        stackEntry->_alternativeSplits.append( *split );
        ++split;
    }

    _stack.append( stackEntry );

    if ( _statistics )
    {
        unsigned level = getStackDepth();
        _statistics->setUnsignedAttribute( Statistics::CURRENT_DECISION_LEVEL,
                                           level );
        if ( level > _statistics->getUnsignedAttribute
                ( Statistics::MAX_DECISION_LEVEL ) )
            _statistics->setUnsignedAttribute( Statistics::MAX_DECISION_LEVEL,
                                               level );
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_SMT_CORE_MICRO, TimeUtils::timePassed( start, end ) );
    }

    _constraintForSplitting = nullptr;
}

unsigned SmtCore::getStackDepth() const
{
    ASSERT( _stack.size() == static_cast<unsigned>( _context.getLevel() ) );
    return _stack.size();
}

void SmtCore::popContext()
{
    struct timespec start = TimeUtils::sampleMicro();
    _context.pop();
    struct timespec end = TimeUtils::sampleMicro();

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_CONTEXT_POPS );
        _statistics->incLongAttribute( Statistics::TIME_CONTEXT_POP, TimeUtils::timePassed( start, end ) );
    }
}

void SmtCore::pushContext()
{
    struct timespec start = TimeUtils::sampleMicro();
    _context.push();
    struct timespec end = TimeUtils::sampleMicro();

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_CONTEXT_PUSHES );
        _statistics->incLongAttribute( Statistics::TIME_CONTEXT_PUSH, TimeUtils::timePassed( start, end ) );
    }
}

bool SmtCore::popSplit()
{
    SMT_LOG( "Performing a pop" );

    if ( _stack.empty() )
        return false;

    struct timespec start = TimeUtils::sampleMicro();

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_POPS );
        // A pop always sends us to a state that we haven't seen before - whether
        // from a sibling split, or from a lower level of the tree.
        _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );
    }

    bool inconsistent = true;
    while ( inconsistent )
    {
        // Remove any entries that have no alternatives
        String error;
        while ( _stack.back()->_alternativeSplits.empty() )
        {
            if ( checkSkewFromDebuggingSolution() )
            {
                // Pops should not occur from a compliant stack!
                printf( "Error! Popping from a compliant stack\n" );
                throw MarabouError( MarabouError::DEBUGGING_ERROR );
            }
            if (Options::get()->getBool(Options::CHECK)) {
                if (_stack.size() != 1) {
                    delete _stack.back()->_engineState;
                    delete _stack.back();
                    _stack.popBack();
                    popContext();
                } else {
                    popContext();
                    _engine->postContextPopHook();
                    _engine->restoreState(*(_stack.back()->_engineState));
                    delete _stack.back()->_engineState;
                    delete _stack.back();
                    _stack.popBack();
                }
            } else {
                delete _stack.back()->_engineState;
                delete _stack.back();
                _stack.popBack();
                popContext();
            }
            if ( _stack.empty() )
                return false;
        }

        if ( checkSkewFromDebuggingSolution() )
        {
            // Pops should not occur from a compliant stack!
            printf( "Error! Popping from a compliant stack\n" );
            throw MarabouError( MarabouError::DEBUGGING_ERROR );
        }

        SmtStackEntry *stackEntry = _stack.back();
        popContext();
        _engine->postContextPopHook();
        // Restore the state of the engine
        SMT_LOG( "\tRestoring engine state..." );
        _engine->restoreState( *( stackEntry->_engineState ) );
        SMT_LOG( "\tRestoring engine state - DONE" );

        // Apply the new split and erase it from the list
        auto split = stackEntry->_alternativeSplits.begin();

        // Erase any valid splits that were learned using the split we just
        // popped
        stackEntry->_impliedValidSplits.clear();

        SMT_LOG( "\tApplying new split..." );
        ASSERT( split->getEquations().size() == 0 );
        _engine->preContextPushHook();
        pushContext();
        _engine->applySplit( *split );
        SMT_LOG( "\tApplying new split - DONE" );

        stackEntry->_activeSplit = *split;
        stackEntry->_alternativeSplits.erase( split );

        inconsistent = !_engine->consistentBounds();
    }

    if ( _statistics )
    {
        unsigned level = getStackDepth();
        _statistics->setUnsignedAttribute( Statistics::CURRENT_DECISION_LEVEL,
                                           level );
        if ( level > _statistics->getUnsignedAttribute
                ( Statistics::MAX_DECISION_LEVEL ) )
            _statistics->setUnsignedAttribute( Statistics::MAX_DECISION_LEVEL,
                                               level );
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_SMT_CORE_MICRO, TimeUtils::timePassed( start, end ) );
    }

    checkSkewFromDebuggingSolution();

    return true;
}

void SmtCore::resetSplitConditions()
{
    _constraintToViolationCount.clear();
    _numRejectedPhasePatternProposal = 0;
    _needToSplit = false;
}

void SmtCore::recordImpliedValidSplit( PiecewiseLinearCaseSplit &validSplit )
{
    if ( _stack.empty() )
        _impliedValidSplitsAtRoot.append( validSplit );
    else
        _stack.back()->_impliedValidSplits.append( validSplit );

    checkSkewFromDebuggingSolution();
}



void SmtCore::allSplitsSoFar( List<PiecewiseLinearCaseSplit> &result ) const
{
    result.clear();

    for ( const auto &it : _impliedValidSplitsAtRoot )
        result.append( it );

    for ( const auto &it : _stack )
    {
        result.append( it->_activeSplit );
        for ( const auto &impliedSplit : it->_impliedValidSplits )
            result.append( impliedSplit );
    }
}

void SmtCore::setStatistics( Statistics *statistics )
{
    _statistics = statistics;
}

void SmtCore::storeDebuggingSolution( const Map<unsigned, double> &debuggingSolution )
{
    _debuggingSolution = debuggingSolution;
}

// Return true if stack is currently compliant, false otherwise
// If there is no stored solution, return false --- incompliant.
bool SmtCore::checkSkewFromDebuggingSolution()
{
    if ( _debuggingSolution.empty() )
        return false;

    String error;

    // First check that the valid splits implied at the root level are okay
    for ( const auto &split : _impliedValidSplitsAtRoot )
    {
        if ( !splitAllowsStoredSolution( split, error ) )
        {
            printf( "Error with one of the splits implied at root level:\n\t%s\n", error.ascii() );
            throw MarabouError( MarabouError::DEBUGGING_ERROR );
        }
    }

    // Now go over the stack from oldest to newest and check that each level is compliant
    for ( const auto &stackEntry : _stack )
    {
        // If the active split is non-compliant but there are alternatives, that's fine
        if ( !splitAllowsStoredSolution( stackEntry->_activeSplit, error ) )
        {
            if ( stackEntry->_alternativeSplits.empty() )
            {
                printf( "Error! Have a split that is non-compliant with the stored solution, "
                        "without alternatives:\n\t%s\n", error.ascii() );
                throw MarabouError( MarabouError::DEBUGGING_ERROR );
            }

            // Active split is non-compliant but this is fine, because there are alternatives. We're done.
            return false;
        }

        // Did we learn any valid splits that are non-compliant?
        for ( auto const &split : stackEntry->_impliedValidSplits )
        {
            if ( !splitAllowsStoredSolution( split, error ) )
            {
                printf( "Error with one of the splits implied at this stack level:\n\t%s\n",
                        error.ascii() );
                throw MarabouError( MarabouError::DEBUGGING_ERROR );
            }
        }
    }

    // No problems were detected, the stack is compliant with the stored solution
    return true;
}

bool SmtCore::splitAllowsStoredSolution( const PiecewiseLinearCaseSplit &split, String &error ) const
{
    // False if the split prevents one of the values in the stored solution, true otherwise.
    error = "";
    if ( _debuggingSolution.empty() )
        return true;

    for ( const auto &bound : split.getBoundTightenings() )
    {
        unsigned variable = bound._variable;

        // If the possible solution doesn't care about this variable,
        // ignore it
        if ( !_debuggingSolution.exists( variable ) )
            continue;

        // Otherwise, check that the bound is consistent with the solution
        double solutionValue = _debuggingSolution[variable];
        double boundValue = bound._value;

        if ( ( bound._type == Tightening::LB ) && FloatUtils::gt( boundValue, solutionValue ) )
        {
            error = Stringf( "Variable %u: new LB is %.5lf, which contradicts possible solution %.5lf",
                             variable,
                             boundValue,
                             solutionValue );
            return false;
        }
        else if ( ( bound._type == Tightening::UB ) && FloatUtils::lt( boundValue, solutionValue ) )
        {
            error = Stringf( "Variable %u: new UB is %.5lf, which contradicts possible solution %.5lf",
                             variable,
                             boundValue,
                             solutionValue );
            return false;
        }
    }

    return true;
}

PiecewiseLinearConstraint *SmtCore::chooseViolatedConstraintForFixing( List<PiecewiseLinearConstraint *> &_violatedPlConstraints ) const
{
    ASSERT( !_violatedPlConstraints.empty() );

    if ( !GlobalConfiguration::USE_LEAST_FIX )
        return *_violatedPlConstraints.begin();

    PiecewiseLinearConstraint *candidate;

    // Apply the least fix heuristic
    auto it = _violatedPlConstraints.begin();

    candidate = *it;
    unsigned minFixes = getViolationCounts( candidate );

    PiecewiseLinearConstraint *contender;
    unsigned contenderFixes;
    while ( it != _violatedPlConstraints.end() )
    {
        contender = *it;
        contenderFixes = getViolationCounts( contender );
        if ( contenderFixes < minFixes )
        {
            minFixes = contenderFixes;
            candidate = contender;
        }

        ++it;
    }

    return candidate;
}

void SmtCore::replaySmtStackEntry( SmtStackEntry *stackEntry )
{
    struct timespec start = TimeUtils::sampleMicro();

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_SPLITS );
        _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );
    }

    // Obtain the current state of the engine
    EngineState *stateBeforeSplits = new EngineState;
    stateBeforeSplits->_stateId = _stateId;
    ++_stateId;
    _engine->storeState( *stateBeforeSplits,
                         TableauStateStorageLevel::STORE_ENTIRE_TABLEAU_STATE );
    stackEntry->_engineState = stateBeforeSplits;

    // Apply all the splits
    _engine->applySplit( stackEntry->_activeSplit );
    for ( const auto &impliedSplit : stackEntry->_impliedValidSplits )
        _engine->applySplit( impliedSplit );

    _stack.append( stackEntry );

    if ( _statistics )
    {
        unsigned level = getStackDepth();
        _statistics->setUnsignedAttribute( Statistics::CURRENT_DECISION_LEVEL,
                                           level );
        if ( level > _statistics->getUnsignedAttribute
                ( Statistics::MAX_DECISION_LEVEL ) )
            _statistics->setUnsignedAttribute( Statistics::MAX_DECISION_LEVEL,
                                               level );
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_SMT_CORE_MICRO, TimeUtils::timePassed( start, end ) );
    }
}

void SmtCore::storeSmtState( SmtState &smtState )
{
    smtState._impliedValidSplitsAtRoot = _impliedValidSplitsAtRoot;

    for ( auto &stackEntry : _stack )
        smtState._stack.append( stackEntry->duplicateSmtStackEntry() );

    smtState._stateId = _stateId;
}

bool SmtCore::pickSplitPLConstraint()
{
    if ( _needToSplit )
    {
        _constraintForSplitting = _engine->pickSplitPLConstraint
                ( _branchingHeuristic );
    }
    return _constraintForSplitting != NULL;
}

void SmtCore::printStackInfo() {
    printf("Total stack depth: %d\n", getStackDepth());
    int level = 0;
    for (auto& stack : _stack) {
        printf("==================================================\n");
        printf("Stack level: %d\n", level ++);
        printf("=======Active case split=======\n");
        stack->_activeSplit.dump();
        printf("\n=======Implied case splits=======\n");
        for (auto& split : stack->_impliedValidSplits) {
            split.dump();
        }
        printf("\n");
    }
}

void SmtCore::recordStackInfo() {
    std::vector<PathElement> path;
    for (auto& stack : _stack) {
        PathElement element;
        element.setSplit(stack->_activeSplit.getInfo());
        for (auto& split : stack->_impliedValidSplits) {
            element.appendImpliedSplit(split.getInfo());
        }
        path.push_back(std::move(element));
    }
    _searchPath.appendPath(path);
}

void SmtCore::setConstraintForSplit(PiecewiseLinearConstraint *constraint, CaseSplitType splitType) {
    assert(_needToSplit == false);
    _needToSplit = true;
    _constraintForSplitting = constraint;
    _splitType = splitType;
}

void SmtCore::popToBottom() {
    SMT_LOG( "Performing a pop to Bottom" );

    if ( _stack.empty() )
        return;

    struct timespec start = TimeUtils::sampleMicro();

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_POPS );
        // A pop always sends us to a state that we haven't seen before - whether
        // from a sibling split, or from a lower level of the tree.
        _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );
    }

    while ( !_stack.empty() )
    {
        // Remove any entries that have no alternatives
        String error;

        if ( checkSkewFromDebuggingSolution() )
        {
            // Pops should not occur from a compliant stack!
            printf( "Error! Popping from a compliant stack\n" );
            throw MarabouError( MarabouError::DEBUGGING_ERROR );
        }

        if (_stack.size() != 1) {
            delete _stack.back()->_engineState;
            delete _stack.back();
            _stack.popBack();
            popContext();
        } else {
            popContext();
            _engine->postContextPopHook();
            _engine->restoreState(*(_stack.back()->_engineState));
            printf("Hello!\n");
            delete _stack.back()->_engineState;
            delete _stack.back();
            _stack.popBack();
        }

        if ( checkSkewFromDebuggingSolution() )
        {
            // Pops should not occur from a compliant stack!
            printf( "Error! Popping from a compliant stack\n" );
            throw MarabouError( MarabouError::DEBUGGING_ERROR );
        }
    }

    if ( _statistics )
    {
        unsigned level = getStackDepth();
        _statistics->setUnsignedAttribute( Statistics::CURRENT_DECISION_LEVEL,
                                           level );
        if ( level > _statistics->getUnsignedAttribute
                ( Statistics::MAX_DECISION_LEVEL ) )
            _statistics->setUnsignedAttribute( Statistics::MAX_DECISION_LEVEL,
                                               level );
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_SMT_CORE_MICRO, TimeUtils::timePassed( start, end ) );
    }

    checkSkewFromDebuggingSolution();
}

void SmtCore::performCheckSplit() {
    ASSERT( _needToSplit );
    _numRejectedPhasePatternProposal = 0;
    // Maybe the constraint has already become inactive - if so, ignore
    if (!_constraintForSplitting or !_constraintForSplitting->isActive() )
    {
        printf("Constraint become not valid!\n");
        _needToSplit = false;
        _constraintToViolationCount[_constraintForSplitting] = 0;
        _constraintForSplitting = nullptr;
        return;
    }

    struct timespec start = TimeUtils::sampleMicro();

    ASSERT( _constraintForSplitting->isActive() );
    _needToSplit = false;

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_SPLITS );
        _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );
    }

    // Before storing the state of the engine, we:
    //   1. Obtain the splits.
    //   2. Disable the constraint, so that it is marked as disbaled in the EngineState.
    List<PiecewiseLinearCaseSplit> oSplits = _constraintForSplitting->getCaseSplits(), splits;
    for (auto& split : oSplits) {
        if (split.getType() == _splitType) {
            splits.append(split);
            break;
        }
    }
    assert(splits.size() == 1);
    _constraintForSplitting->setActiveConstraint( false );

    // Obtain the current state of the engine
    EngineState *stateBeforeSplits = new EngineState;
    stateBeforeSplits->_stateId = _stateId;
    ++_stateId;
    _engine->storeState( *stateBeforeSplits,
                         TableauStateStorageLevel::STORE_BOUNDS_ONLY );
    _engine->preContextPushHook();
    pushContext();
    SmtStackEntry *stackEntry = new SmtStackEntry;
    // Perform the first split: add bounds and equations
    List<PiecewiseLinearCaseSplit>::iterator split = splits.begin();
    ASSERT( split->getEquations().size() == 0 );
    _engine->applySplit( *split );
    stackEntry->_activeSplit = *split;

    // Store the remaining splits on the stack, for later
    stackEntry->_engineState = stateBeforeSplits;
    ++split;
    while ( split != splits.end() )
    {
        stackEntry->_alternativeSplits.append( *split );
        ++split;
    }

    _stack.append( stackEntry );

    if ( _statistics )
    {
        unsigned level = getStackDepth();
        _statistics->setUnsignedAttribute( Statistics::CURRENT_DECISION_LEVEL,
                                           level );
        if ( level > _statistics->getUnsignedAttribute
                ( Statistics::MAX_DECISION_LEVEL ) )
            _statistics->setUnsignedAttribute( Statistics::MAX_DECISION_LEVEL,
                                               level );
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_SMT_CORE_MICRO, TimeUtils::timePassed( start, end ) );
    }

    _constraintForSplitting = nullptr;
}

void SmtCore::printSimpleStackInfo() {
    printf("Total stack depth: %d\n", getStackDepth());
    int level = 0;
    for (auto& stack : _stack) {
        printf("Stack level: %d, ", level ++);
        stack->_activeSplit.getPosition().dump();
        printf("\n");
    }
}

bool SmtCore::popCheckSplit() {
    if ( _stack.empty() )
        return false;

    struct timespec start = TimeUtils::sampleMicro();

    if ( _statistics )
    {
        _statistics->incUnsignedAttribute( Statistics::NUM_POPS );
        // A pop always sends us to a state that we haven't seen before - whether
        // from a sibling split, or from a lower level of the tree.
        _statistics->incUnsignedAttribute( Statistics::NUM_VISITED_TREE_STATES );
    }

    bool inconsistent = true;
    while ( inconsistent )
    {
        // Remove any entries that have no alternatives
        String error;
        while ( _stack.back()->_alternativeSplits.empty() )
        {
            if ( checkSkewFromDebuggingSolution() )
            {
                // Pops should not occur from a compliant stack!
                printf( "Error! Popping from a compliant stack\n" );
                throw MarabouError( MarabouError::DEBUGGING_ERROR );
            }

            if (_stack.size() != 1) {
                delete _stack.back()->_engineState;
                delete _stack.back();
                _stack.popBack();
                popContext();
            } else {
                popContext();
                _engine->postContextPopHook();
                _engine->restoreState(*(_stack.back()->_engineState));
                delete _stack.back()->_engineState;
                delete _stack.back();
                _stack.popBack();
            }


            if ( _stack.empty() )
                return false;
        }

        if ( checkSkewFromDebuggingSolution() )
        {
            // Pops should not occur from a compliant stack!
            printf( "Error! Popping from a compliant stack\n" );
            throw MarabouError( MarabouError::DEBUGGING_ERROR );
        }

        SmtStackEntry *stackEntry = _stack.back();

        popContext();
        _engine->postContextPopHook();
        // Restore the state of the engine
        SMT_LOG( "\tRestoring engine state..." );
        _engine->restoreState( *( stackEntry->_engineState ) );
        SMT_LOG( "\tRestoring engine state - DONE" );

        // Apply the new split and erase it from the list
        auto split = stackEntry->_alternativeSplits.begin();

        // Erase any valid splits that were learned using the split we just
        // popped
        stackEntry->_impliedValidSplits.clear();

        SMT_LOG( "\tApplying new split..." );
        ASSERT( split->getEquations().size() == 0 );
        _engine->preContextPushHook();
        pushContext();
        _engine->applySplit( *split );
        SMT_LOG( "\tApplying new split - DONE" );

        stackEntry->_activeSplit = *split;
        stackEntry->_alternativeSplits.erase( split );

        inconsistent = !_engine->consistentBounds();
    }

    if ( _statistics )
    {
        unsigned level = getStackDepth();
        _statistics->setUnsignedAttribute( Statistics::CURRENT_DECISION_LEVEL,
                                           level );
        if ( level > _statistics->getUnsignedAttribute
                ( Statistics::MAX_DECISION_LEVEL ) )
            _statistics->setUnsignedAttribute( Statistics::MAX_DECISION_LEVEL,
                                               level );
        struct timespec end = TimeUtils::sampleMicro();
        _statistics->incLongAttribute( Statistics::TOTAL_TIME_SMT_CORE_MICRO, TimeUtils::timePassed( start, end ) );
    }

    checkSkewFromDebuggingSolution();

    return true;
}

unsigned int SmtCore::AtLeastBackTrackTo(unsigned level) {
    if ( _stack.empty() )
        return 0;

    String error;
    if (getStackDepth() > level) {
        _stack.back()->_alternativeSplits.clear();
    }
    while ( getStackDepth() > level and _stack.back()->_alternativeSplits.empty() )
    {
        if ( checkSkewFromDebuggingSolution() )
        {
            // Pops should not occur from a compliant stack!
            printf( "Error! Popping from a compliant stack\n" );
            throw MarabouError( MarabouError::DEBUGGING_ERROR );
        }
        delete _stack.back()->_engineState;
        delete _stack.back();
        _stack.popBack();
        popContext();
        if (getStackDepth() > level) {
            _stack.back()->_alternativeSplits.clear();
        }
        if ( _stack.empty() )
            return 0;
    }

    SmtStackEntry *stackEntry = _stack.back();
    popContext();
    _engine->postContextPopHook();
    // Restore the state of the engine
    _engine->restoreState( *( stackEntry->_engineState ) );

    // Apply the new split and erase it from the list
    auto& split = stackEntry->_activeSplit;

    // Erase any valid splits that were learned using the split we just
    // popped
    _engine->preContextPushHook();
    pushContext();
    _engine->applySplit( split );
//    printf("apply split : "); split.dump();
//    printf("\n");
    auto& learnt_clause = _searchPath._learnt.back();
    auto position = learnt_clause.back()._caseSplit._position;
//    printf("-----Learnt-----\n");
//    for (auto learn : learnt_clause) {
//        learn._caseSplit.dump();
//        printf("\n");
//    }
//    printf("")
    CaseSplitType type = learnt_clause.back()._caseSplit._type;
//    printf("============================================================================\n");
//    printf("Learnt size: %d, statck depth:%d \n", learnt_clause.size(),getStackDepth() );
//    printf("-------------------------------\n");

    for (auto& split : stackEntry->_impliedValidSplits) {
        auto pos = split.getPosition();
        auto type = split.getType();
        auto constraint = _engine->getConstraintByPosition(pos);
        _engine->performTargetSplit(constraint, type, 0);
    }

    // get learned path
//    printf("-------------------------------\n");
    auto constraint = _engine->getConstraintByPosition(position);
    type = reverseCaseSplitType(type);

    _engine->performTargetSplit(constraint, type, 1);


    return getStackDepth();
}

CaseSplitType SmtCore::reverseCaseSplitType(CaseSplitType type) {
    switch (type) {
        case RELU_INACTIVE:
            return RELU_ACTIVE;
        case RELU_ACTIVE:
            return RELU_INACTIVE;
        case DISJUNCTION_LOWER:
            return DISJUNCTION_UPPER;
        case DISJUNCTION_UPPER:
            return DISJUNCTION_LOWER;
        default:
            return UNKNOWN;
    }
}

void SmtCore::recordSatImpliedValidSplit(PiecewiseLinearCaseSplit &validSplit) {
    if ( _stack.empty() )
        _satImpliedValidSplitsAtRoot.append( validSplit );
    else
        _stack.back()->_satImpliedValidSplits.append( validSplit );
}

unsigned int SmtCore::backTrackToGivenLevelAndPerformSplit(unsigned int level, CaseSplitTypeInfo& info) {
    if ( _stack.empty() )
        return 0;

    String error;
    if (getStackDepth() > level) {
        _stack.back()->_alternativeSplits.clear();
    }
    while ( getStackDepth() > level and _stack.back()->_alternativeSplits.empty() )
    {
        if ( checkSkewFromDebuggingSolution() )
        {
            // Pops should not occur from a compliant stack!
            printf( "Error! Popping from a compliant stack\n" );
            throw MarabouError( MarabouError::DEBUGGING_ERROR );
        }
        delete _stack.back()->_engineState;
        delete _stack.back();
        _stack.popBack();
        popContext();
        if (getStackDepth() > level) {
            _stack.back()->_alternativeSplits.clear();
        }
        if ( _stack.empty() )
            return 0;
    }

    SmtStackEntry *stackEntry = _stack.back();
    popContext();
    _engine->postContextPopHook();
    // Restore the state of the engine
    _engine->restoreState( *( stackEntry->_engineState ) );

    // Apply the new split and erase it from the list
    auto& split = stackEntry->_activeSplit;

    // Erase any valid splits that were learned using the split we just
    // popped
    _engine->preContextPushHook();
    pushContext();
    _engine->applySplit( split );
//    printf("apply split : "); split.dump();
//    printf("\n");
//    printf("-----Learnt-----\n");
//    for (auto learn : learnt_clause) {
//        learn._caseSplit.dump();
//        printf("\n");
//    }
//    printf("")

//    printf("============================================================================\n");
//    printf("Learnt size: %d, statck depth:%d \n", learnt_clause.size(),getStackDepth() );
//    printf("-------------------------------\n");

    for (auto& split : stackEntry->_impliedValidSplits) {
        auto pos = split.getPosition();
        auto type = split.getType();
        auto constraint = _engine->getConstraintByPosition(pos);
        _engine->performTargetSplit(constraint, type, 0);
    }

    // get learned path
//    printf("-------------------------------\n");

    auto position = info._position;
    CaseSplitType type = info._type;
    auto constraint = _engine->getConstraintByPosition(position);
    performGivenSplit(constraint, type);


    return getStackDepth();
}

CaseSplitTypeInfo SmtCore::getActiveCaseSplitInfo() {
    return _stack.back()->_activeSplit.getInfo();
}

void SmtCore::performGivenSplit(PiecewiseLinearConstraint *constraint, CaseSplitType type) {
    _numRejectedPhasePatternProposal = 0;
    // Maybe the constraint has already become inactive - if so, ignore
    if (!constraint or !constraint->isActive() )
    {
        printf("Constraint become not valid!\n");
        return;
    }

    // Before storing the state of the engine, we:
    //   1. Obtain the splits.
    //   2. Disable the constraint, so that it is marked as disbaled in the EngineState.
    List<PiecewiseLinearCaseSplit> oSplits = constraint->getCaseSplits(), splits;
    for (auto& split : oSplits) {
        if (split.getType() == type) {
            splits.append(split);
            break;
        }
    }
    assert(splits.size() == 1);
    constraint->setActiveConstraint( false );
    // Obtain the current state of the engine
    EngineState *stateBeforeSplits = new EngineState;
    stateBeforeSplits->_stateId = _stateId;
    ++_stateId;
    _engine->storeState( *stateBeforeSplits,
                         TableauStateStorageLevel::STORE_BOUNDS_ONLY );
    _engine->preContextPushHook();
    pushContext();
    SmtStackEntry *stackEntry = new SmtStackEntry;
    // Perform the first split: add bounds and equations
    List<PiecewiseLinearCaseSplit>::iterator split = splits.begin();
    ASSERT( split->getEquations().size() == 0 );
    _engine->applySplit( *split );
    stackEntry->_activeSplit = *split;

    // Store the remaining splits on the stack, for later
    stackEntry->_engineState = stateBeforeSplits;
    _stack.append( stackEntry );
}

unsigned int SmtCore::backTrackTo(unsigned int level) {
    if ( _stack.empty() )
        return 0;

    String error;
    if (getStackDepth() > level) {
        _stack.back()->_alternativeSplits.clear();
    }
    while ( getStackDepth() > level and _stack.back()->_alternativeSplits.empty() )
    {
        if ( checkSkewFromDebuggingSolution() )
        {
            // Pops should not occur from a compliant stack!
            printf( "Error! Popping from a compliant stack\n" );
            throw MarabouError( MarabouError::DEBUGGING_ERROR );
        }
        delete _stack.back()->_engineState;
        delete _stack.back();
        _stack.popBack();
        popContext();
        if (getStackDepth() > level) {
            _stack.back()->_alternativeSplits.clear();
        }
        if ( _stack.empty() )
            return 0;
    }

    SmtStackEntry *stackEntry = _stack.back();
    popContext();
    _engine->postContextPopHook();
    // Restore the state of the engine
    _engine->restoreState( *( stackEntry->_engineState ) );

    // Apply the new split and erase it from the list
    auto& split = stackEntry->_activeSplit;

    // Erase any valid splits that were learned using the split we just
    // popped
    _engine->preContextPushHook();
    pushContext();
    _engine->applySplit( split );
//    printf("apply split : "); split.dump();
//    printf("\n");
    auto& learnt_clause = _searchPath._learnt.back();
    auto position = learnt_clause.back()._caseSplit._position;
//    printf("-----Learnt-----\n");
//    for (auto learn : learnt_clause) {
//        learn._caseSplit.dump();
//        printf("\n");
//    }
//    printf("")
    CaseSplitType type = learnt_clause.back()._caseSplit._type;
//    printf("============================================================================\n");
//    printf("Learnt size: %d, statck depth:%d \n", learnt_clause.size(),getStackDepth() );
//    printf("-------------------------------\n");

    for (auto& split : stackEntry->_impliedValidSplits) {
        auto pos = split.getPosition();
        auto type = split.getType();
        auto constraint = _engine->getConstraintByPosition(pos);
        _engine->performTargetSplit(constraint, type, 0);
    }
    return getStackDepth();
}

    