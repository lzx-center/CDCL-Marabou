
/*********************                                                        */
/*! \file Engine.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Duligur Ibeling, Andrew Wu
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** \brief [[ Add one-line brief description here ]]
 **
 ** [[ Add lengthier description here ]]
 **/

#include "AutoConstraintMatrixAnalyzer.h"
#include "Debug.h"
#include "DisjunctionConstraint.h"
#include "Engine.h"
#include "EngineState.h"
#include "InfeasibleQueryException.h"
#include "InputQuery.h"
#include "MStringf.h"
#include "MalformedBasisException.h"
#include "MarabouError.h"
#include "NLRError.h"
#include "PiecewiseLinearConstraint.h"
#include "Preprocessor.h"
#include "TableauRow.h"
#include "TimeUtils.h"
#include "VariableOutOfBoundDuringOptimizationException.h"
#include "Vector.h"
#include <random>

std::map<std::string, long long> CenterStatics::_functionTime;
std::map<int, int> CenterStatics::_backTrackStatics;
Engine::Engine()
        : _context(), _boundManager(_context), _tableau(_boundManager), _preprocessedQuery(nullptr),
          _rowBoundTightener(*_tableau), _smtCore(this), _numPlConstraintsDisabledByValidSplits(0),
          _preprocessingEnabled(false), _initialStateStored(false), _work(NULL),
          _basisRestorationRequired(Engine::RESTORATION_NOT_NEEDED),
          _basisRestorationPerformed(Engine::NO_RESTORATION_PERFORMED), _costFunctionManager(_tableau),
          _quitRequested(false), _exitCode(Engine::NOT_DONE), _numVisitedStatesAtPreviousRestoration(0),
          _networkLevelReasoner(NULL), _verbosity(Options::get()->getInt(Options::VERBOSITY)), _lastNumVisitedStates(0),
          _lastIterationWithProgress(0), _symbolicBoundTighteningType(Options::get()->getSymbolicBoundTighteningType()),
          _solveWithMILP(Options::get()->getBool(Options::SOLVE_WITH_MILP)),
          _lpSolverType(Options::get()->getLPSolverType()), _gurobi(nullptr), _milpEncoder(nullptr),
          _soiManager(nullptr), _simulationSize(Options::get()->getInt(Options::NUMBER_OF_SIMULATIONS)),
          _isGurobyEnabled(Options::get()->gurobiEnabled()),
          _performLpTighteningAfterSplit(Options::get()->getBool(Options::PERFORM_LP_TIGHTENING_AFTER_SPLIT)),
          _milpSolverBoundTighteningType(Options::get()->getMILPSolverBoundTighteningType()), _sncMode(false),
          _queryId("") {
    _smtCore.setStatistics(&_statistics);
    _tableau->setStatistics(&_statistics);
    _rowBoundTightener->setStatistics(&_statistics);
    _preprocessor.setStatistics(&_statistics);

    _activeEntryStrategy = _projectedSteepestEdgeRule;
    _activeEntryStrategy->setStatistics(&_statistics);
    _statistics.stampStartingTime();
    setRandomSeed(Options::get()->getInt(Options::SEED));

    _statisticsPrintingFrequency =
            (_lpSolverType == LPSolverType::NATIVE) ?
            GlobalConfiguration::STATISTICS_PRINTING_FREQUENCY :
            GlobalConfiguration::STATISTICS_PRINTING_FREQUENCY_GUROBI;
}

Engine::~Engine() {
    if (_work) {
        delete[] _work;
        _work = NULL;
    }
}

void Engine::setVerbosity(unsigned verbosity) {
    _verbosity = verbosity;
}

void Engine::adjustWorkMemorySize() {
    if (_work) {
        delete[] _work;
        _work = NULL;
    }

    _work = new double[_tableau->getM()];
    if (!_work)
        throw MarabouError(MarabouError::ALLOCATION_FAILED, "Engine::work");
}


void Engine::applySnCSplit(PiecewiseLinearCaseSplit sncSplit, String queryId) {
    _sncMode = true;
    _sncSplit = sncSplit;
    _queryId = queryId;
    applySplit(sncSplit);
}

void Engine::setRandomSeed(unsigned seed) {
    srand(seed);
}

InputQuery Engine::prepareSnCInputQuery() {
    List<Tightening> bounds = _sncSplit.getBoundTightenings();
    List<Equation> equations = _sncSplit.getEquations();

    InputQuery sncIPQ = *_preprocessedQuery;
    for (auto &equation: equations)
        sncIPQ.addEquation(equation);

    for (auto &bound: bounds) {
        switch (bound._type) {
            case Tightening::LB:
                sncIPQ.setLowerBound(bound._variable, bound._value);
                break;

            case Tightening::UB:
                sncIPQ.setUpperBound(bound._variable, bound._value);
        }
    }

    return sncIPQ;
}

void Engine::exportInputQueryWithError(String errorMessage) {
    String ipqFileName = (_queryId.length() > 0) ? _queryId + ".ipq" : "failedMarabouQuery.ipq";
    prepareSnCInputQuery().saveQuery(ipqFileName);
    printf("Engine: %s!\nInput query has been saved as %s. Please attach the input query when you open the issue on GitHub.\n",
           errorMessage.ascii(), ipqFileName.ascii());
}

bool Engine::solve(unsigned timeoutInSeconds) {
    SignalHandler::getInstance()->initialize();
    SignalHandler::getInstance()->registerClient(this);

    // Register the boundManager with all the PL constraints
    for (auto &plConstraint: _plConstraints) {
        plConstraint->registerBoundManager(&_boundManager);
        auto pos = plConstraint->getPosition();
        _positionToConstraint[pos] = plConstraint;
    }
    for (auto& plConstraint : _preprocessor.getEliminatedConstraintsList()) {
        auto pos = plConstraint->getPosition();
        CaseSplitType type = CaseSplitType::UNKNOWN;
        if (plConstraint->getPhaseStatus() == RELU_PHASE_ACTIVE) {
            type = CaseSplitType::RELU_ACTIVE;
        } else if (plConstraint->getPhaseStatus() == RELU_PHASE_INACTIVE) {
            type = CaseSplitType::RELU_INACTIVE;
        } else {
            printf("Can not handle!\n");
        }
        _smtCore._searchPath.addEliminatedConstraint(pos._layer, pos._node, RELU_ACTIVE);
    }

    if (_solveWithMILP)
        return solveWithMILPEncoding(timeoutInSeconds);

    updateDirections();
    storeState(_initial, TableauStateStorageLevel::STORE_ENTIRE_TABLEAU_STATE );
    initial_lower.resize(_tableau->getN());
    initial_upper.resize(_tableau->getN());
    for (size_t i = 0; i < _tableau->getN(); ++ i) {
        initial_lower[i] = _tableau->getLowerBound(i);
        initial_upper[i] = _tableau->getUpperBound(i);
    }

    if (_lpSolverType == LPSolverType::NATIVE) {
        storeInitialEngineState();
    } else if (_lpSolverType == LPSolverType::GUROBI) {
        ENGINE_LOG("Encoding convex relaxation into Gurobi...");
        _milpEncoder->encodeInputQuery(*_gurobi, *_preprocessedQuery, true);
        ENGINE_LOG("Encoding convex relaxation into Gurobi - done");
    }
    mainLoopStatistics();
    if (_verbosity > 0) {
        printf("\nEngine::solve: Initial statistics\n");
        _statistics.print();
        printf("\n---\n");
    }
    applyAllValidConstraintCaseSplits();

    bool splitJustPerformed = true;
    struct timespec mainLoopStart = TimeUtils::sampleMicro();
    int num = 0;
    while (true) {
        struct timespec mainLoopEnd = TimeUtils::sampleMicro();
        _statistics.incLongAttribute(Statistics::TIME_MAIN_LOOP_MICRO,
                                     TimeUtils::timePassed(mainLoopStart,
                                                           mainLoopEnd));
        mainLoopStart = mainLoopEnd;

        if (shouldExitDueToTimeout(timeoutInSeconds)) {
            if (_verbosity > 0) {
                printf("\n\nEngine: quitting due to timeout...\n\n");
                printf("Final statistics:\n");
                _statistics.print();
            }

            _exitCode = Engine::TIMEOUT;
            _statistics.timeout();
            return false;
        }

        if (_quitRequested) {
            if (_verbosity > 0) {
                printf("\n\nEngine: quitting due to external request...\n\n");
                printf("Final statistics:\n");
                _statistics.print();
            }

            _exitCode = Engine::QUIT_REQUESTED;
            return false;
        }

        try {
            DEBUG(_tableau->verifyInvariants());

            mainLoopStatistics();
            if (_verbosity > 1 &&
                _statistics.getLongAttribute
                        (Statistics::NUM_MAIN_LOOP_ITERATIONS) %
                _statisticsPrintingFrequency == 0)
                _statistics.print();

            if (_lpSolverType == LPSolverType::NATIVE) {
                checkOverallProgress();
                // Check whether progress has been made recently

                if (performPrecisionRestorationIfNeeded())
                    continue;

                if (_tableau->basisMatrixAvailable()) {
                    explicitBasisBoundTightening();
                    applyAllBoundTightenings();
                    applyAllValidConstraintCaseSplits();
                }
            }

            // If true, we just entered a new subproblem
            if (splitJustPerformed) {
                performBoundTighteningAfterCaseSplit();
                informLPSolverOfBounds();
                splitJustPerformed = false;
            }
            // Perform any SmtCore-initiated case splits
            if (_smtCore.needToSplit()) {
                _smtCore.performSplit();
                splitJustPerformed = true;
                continue;
            }

            if (!_tableau->allBoundsValid()) {
                // Some variable bounds are invalid, so the query is unsat
                throw InfeasibleQueryException();
            }

            if (allVarsWithinBounds()) {
                // The linear portion of the problem has been solved.
                // Check the status of the PL constraints
                bool solutionFound =
                        adjustAssignmentToSatisfyNonLinearConstraints();
                if (solutionFound) {
                    struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                    _statistics.incLongAttribute
                            (Statistics::TIME_MAIN_LOOP_MICRO,
                             TimeUtils::timePassed(mainLoopStart,
                                                   mainLoopEnd));
                    if (_verbosity > 0) {
                        printf("\nEngine::solve: sat assignment found\n");
                        _statistics.print();
                    }
                    _smtCore._searchPath._paths.clear();
                    _smtCore.recordStackInfo();
                    _exitCode = Engine::SAT;

                    return true;
                } else
                    continue;
            }

            // We have out-of-bounds variables.
            if (_lpSolverType == LPSolverType::NATIVE)
                performSimplexStep();
            else {
                ENGINE_LOG("Checking LP feasibility with Gurobi...");
                DEBUG({ checkGurobiBoundConsistency(); });
                ASSERT(_lpSolverType == LPSolverType::GUROBI);
                LinearExpression dontCare;
                minimizeCostWithGurobi(dontCare);
            }
            continue;
        }
        catch (const MalformedBasisException &) {
            _tableau->toggleOptimization(false);
            if (!handleMalformedBasisException()) {
                ASSERT(_lpSolverType == LPSolverType::NATIVE);
                _exitCode = Engine::ERROR;
                exportInputQueryWithError("Cannot restore tableau");
                struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                _statistics.incLongAttribute
                        (Statistics::TIME_MAIN_LOOP_MICRO,
                         TimeUtils::timePassed(mainLoopStart,
                                               mainLoopEnd));
                return false;
            }
        }
        catch (const InfeasibleQueryException &) {
            _tableau->toggleOptimization(false);
            // The current query is unsat, and we need to pop.
            // If we're at level 0, the whole query is unsat.
            _smtCore.recordStackInfo();
            auto& searchPath = getSearchPath();
            auto& back = searchPath._paths.back();
            std::vector<PathElement> new_path;

            searchPath._learnt.push_back(std::move(new_path));
            if (!_smtCore.popSplit()) {
                struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                _statistics.incLongAttribute
                        (Statistics::TIME_MAIN_LOOP_MICRO,
                         TimeUtils::timePassed(mainLoopStart,
                                               mainLoopEnd));
                if (_verbosity > 0) {
                    printf("\nEngine::solve: unsat query\n");
                    _statistics.print();
                }
                _exitCode = Engine::UNSAT;
                return false;
            } else {
                splitJustPerformed = true;
            }
        }
        catch (const VariableOutOfBoundDuringOptimizationException &) {
            _tableau->toggleOptimization(false);
            continue;
        }
        catch (MarabouError &e) {
            String message =
                    Stringf("Caught a MarabouError. Code: %u. Message: %s ",
                            e.getCode(), e.getUserMessage());
            _exitCode = Engine::ERROR;
            exportInputQueryWithError(message);
            struct timespec mainLoopEnd = TimeUtils::sampleMicro();
            _statistics.incLongAttribute
                    (Statistics::TIME_MAIN_LOOP_MICRO,
                     TimeUtils::timePassed(mainLoopStart,
                                           mainLoopEnd));
            return false;
        } catch (...) {
            _exitCode = Engine::ERROR;
            exportInputQueryWithError("Unknown error");
            struct timespec mainLoopEnd = TimeUtils::sampleMicro();
            _statistics.incLongAttribute
                    (Statistics::TIME_MAIN_LOOP_MICRO,
                     TimeUtils::timePassed(mainLoopStart,
                                           mainLoopEnd));
            return false;
        }
    }
}

void Engine::mainLoopStatistics() {
    struct timespec start = TimeUtils::sampleMicro();

    unsigned activeConstraints = 0;
    for (const auto &constraint: _plConstraints)
        if (constraint->isActive())
            ++activeConstraints;

    _statistics.setUnsignedAttribute(Statistics::NUM_ACTIVE_PL_CONSTRAINTS,
                                     activeConstraints);
    _statistics.setUnsignedAttribute(Statistics::NUM_PL_VALID_SPLITS,
                                     _numPlConstraintsDisabledByValidSplits);
    _statistics.setUnsignedAttribute(Statistics::NUM_PL_SMT_ORIGINATED_SPLITS,
                                     _plConstraints.size() - activeConstraints
                                     - _numPlConstraintsDisabledByValidSplits);

    _statistics.incLongAttribute(Statistics::NUM_MAIN_LOOP_ITERATIONS);

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute(Statistics::TOTAL_TIME_HANDLING_STATISTICS_MICRO,
                                 TimeUtils::timePassed(start, end));
}

void Engine::performBoundTighteningAfterCaseSplit() {
    CenterStatics time("performBoundTighteningAfterCaseSplit");
    // Tighten bounds of a first hidden layer with MILP solver
    performMILPSolverBoundedTighteningForSingleLayer(1);
    do {
        performSymbolicBoundTightening();
    } while (applyAllValidConstraintCaseSplits());

    // Tighten bounds of an output layer with MILP solver
    if (_networkLevelReasoner)    // to avoid failing of system test.
        performMILPSolverBoundedTighteningForSingleLayer
                (_networkLevelReasoner->getLayerIndexToLayer().size() - 1);
}

bool Engine::adjustAssignmentToSatisfyNonLinearConstraints() {
    ENGINE_LOG("Linear constraints satisfied. Now trying to satisfy non-linear"
               " constraints...");
    collectViolatedPlConstraints();
    // If all constraints are satisfied, we are possibly done
    if (allPlConstraintsHold()) {
        if (_lpSolverType == LPSolverType::NATIVE &&
            _tableau->getBasicAssignmentStatus() !=
            ITableau::BASIC_ASSIGNMENT_JUST_COMPUTED) {
            if (_verbosity > 0) {
                printf("Before declaring sat, recomputing...\n");
            }
            // Make sure that the assignment is precise before declaring success
            _tableau->computeAssignment();
            // If we actually have a real satisfying assignment,
            return false;
        } else
            return true;
    } else if (!GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH) {
        // We have violated piecewise-linear constraints.
        performConstraintFixingStep();

        // Finally, take this opporunity to tighten any bounds
        // and perform any valid case splits.
        tightenBoundsOnConstraintMatrix();
        applyAllBoundTightenings();
        // For debugging purposes
        checkBoundCompliancyWithDebugSolution();

        while (applyAllValidConstraintCaseSplits())
            performSymbolicBoundTightening();
        return false;
    } else {
        return performDeepSoILocalSearch();
    }
}

bool Engine::performPrecisionRestorationIfNeeded() {
    // If the basis has become malformed, we need to restore it
    if (basisRestorationNeeded()) {
        if (_basisRestorationRequired == Engine::STRONG_RESTORATION_NEEDED) {
            performPrecisionRestoration(PrecisionRestorer::RESTORE_BASICS);
            _basisRestorationPerformed = Engine::PERFORMED_STRONG_RESTORATION;
        } else {
            performPrecisionRestoration(PrecisionRestorer::DO_NOT_RESTORE_BASICS);
            _basisRestorationPerformed = Engine::PERFORMED_WEAK_RESTORATION;
        }

        _numVisitedStatesAtPreviousRestoration =
                _statistics.getUnsignedAttribute(Statistics::NUM_VISITED_TREE_STATES);
        _basisRestorationRequired = Engine::RESTORATION_NOT_NEEDED;
        return true;
    }

    // Restoration is not required
    _basisRestorationPerformed = Engine::NO_RESTORATION_PERFORMED;

    // Possible restoration due to preceision degradation
    if (shouldCheckDegradation() && highDegradation()) {
        performPrecisionRestoration(PrecisionRestorer::RESTORE_BASICS);
        return true;
    }

    return false;
}

bool Engine::handleMalformedBasisException() {
    // Debug
    printf("MalformedBasisException caught!\n");
    //

    if (_basisRestorationPerformed == Engine::NO_RESTORATION_PERFORMED) {
        if (_numVisitedStatesAtPreviousRestoration !=
            _statistics.getUnsignedAttribute
                    (Statistics::NUM_VISITED_TREE_STATES)) {
            // We've tried a strong restoration before, and it didn't work. Do a weak restoration
            _basisRestorationRequired = Engine::WEAK_RESTORATION_NEEDED;
        } else {
            _basisRestorationRequired = Engine::STRONG_RESTORATION_NEEDED;
        }
        return true;
    } else if (_basisRestorationPerformed == Engine::PERFORMED_STRONG_RESTORATION) {
        _basisRestorationRequired = Engine::WEAK_RESTORATION_NEEDED;
        return true;
    } else
        return false;
}

void Engine::performConstraintFixingStep() {
    // Statistics
    _statistics.incLongAttribute(Statistics::NUM_CONSTRAINT_FIXING_STEPS);
    struct timespec start = TimeUtils::sampleMicro();

    // Select a violated constraint as the target
    selectViolatedPlConstraint();

    // Report the violated constraint to the SMT engine
    reportPlViolation();

    // Attempt to fix the constraint
    fixViolatedPlConstraintIfPossible();

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute(Statistics::TIME_CONSTRAINT_FIXING_STEPS_MICRO,
                                 TimeUtils::timePassed(start, end));
}

bool Engine::performSimplexStep() {
    // Statistics
    _statistics.incLongAttribute(Statistics::NUM_SIMPLEX_STEPS);
    struct timespec start = TimeUtils::sampleMicro();

    /*
      In order to increase numerical stability, we attempt to pick a
      "good" entering/leaving combination, by trying to avoid tiny pivot
      values. We do this as follows:

      1. Pick an entering variable according to the strategy in use.
      2. Find the entailed leaving variable.
      3. If the combination is bad, go back to (1) and find the
         next-best entering variable.
    */

    if (_tableau->isOptimizing())
        _costFunctionManager->computeGivenCostFunction
                (_heuristicCost._addends);
    if (_costFunctionManager->costFunctionInvalid())
        _costFunctionManager->computeCoreCostFunction();
    else
        _costFunctionManager->adjustBasicCostAccuracy();

    DEBUG({
              // Since we're performing a simplex step, there are out-of-bounds variables.
              // Therefore, if the cost function is fresh, it should not be zero.
              if (_costFunctionManager->costFunctionJustComputed()) {
                  const double *costFunction = _costFunctionManager->getCostFunction();
                  unsigned size = _tableau->getN() - _tableau->getM();
                  bool found = false;
                  for (unsigned i = 0; i < size; ++i) {
                      if (!FloatUtils::isZero(costFunction[i])) {
                          found = true;
                          break;
                      }
                  }

                  if (!found) {
                      printf("Error! Have OOB vars but cost function is zero.\n"
                             "Recomputing cost function. New one is:\n");
                      _costFunctionManager->computeCoreCostFunction();
                      _costFunctionManager->dumpCostFunction();
                      throw MarabouError(MarabouError::DEBUGGING_ERROR,
                                         "Have OOB vars but cost function is zero");
                  }
              }
          });

    // Obtain all eligible entering varaibles
    List<unsigned> enteringVariableCandidates;
    _tableau->getEntryCandidates(enteringVariableCandidates);

    unsigned bestLeaving = 0;
    double bestChangeRatio = 0.0;
    Set<unsigned> excludedEnteringVariables;
    bool haveCandidate = false;
    unsigned bestEntering = 0;
    double bestPivotEntry = 0.0;
    unsigned tries = GlobalConfiguration::MAX_SIMPLEX_PIVOT_SEARCH_ITERATIONS;

    while (tries > 0) {
        --tries;

        // Attempt to pick the best entering variable from the available candidates
        if (!_activeEntryStrategy->select(_tableau,
                                          enteringVariableCandidates,
                                          excludedEnteringVariables)) {
            // No additional candidates can be found.
            break;
        }

        // We have a candidate!
        haveCandidate = true;

        // We don't want to re-consider this candidate in future
        // iterations
        excludedEnteringVariables.insert(_tableau->getEnteringVariableIndex());

        // Pick a leaving variable
        _tableau->computeChangeColumn();
        _tableau->pickLeavingVariable();

        // A fake pivot always wins
        if (_tableau->performingFakePivot()) {
            bestEntering = _tableau->getEnteringVariableIndex();
            bestLeaving = _tableau->getLeavingVariableIndex();
            bestChangeRatio = _tableau->getChangeRatio();
            memcpy(_work, _tableau->getChangeColumn(), sizeof(double) * _tableau->getM());
            break;
        }

        // Is the newly found pivot better than the stored one?
        unsigned leavingIndex = _tableau->getLeavingVariableIndex();
        double pivotEntry = FloatUtils::abs(_tableau->getChangeColumn()[leavingIndex]);
        if (pivotEntry > bestPivotEntry) {
            bestEntering = _tableau->getEnteringVariableIndex();
            bestPivotEntry = pivotEntry;
            bestLeaving = leavingIndex;
            bestChangeRatio = _tableau->getChangeRatio();
            memcpy(_work, _tableau->getChangeColumn(), sizeof(double) * _tableau->getM());
        }

        // If the pivot is greater than the sought-after threshold, we
        // are done.
        if (bestPivotEntry >= GlobalConfiguration::ACCEPTABLE_SIMPLEX_PIVOT_THRESHOLD)
            break;
        else
            _statistics.incLongAttribute
                    (Statistics::NUM_SIMPLEX_PIVOT_SELECTIONS_IGNORED_FOR_STABILITY);
    }

    // If we don't have any candidates, this simplex step has failed.
    if (!haveCandidate) {
        if (_tableau->getBasicAssignmentStatus() != ITableau::BASIC_ASSIGNMENT_JUST_COMPUTED) {
            // This failure might have resulted from a corrupt basic assignment.
            _tableau->computeAssignment();
            struct timespec end = TimeUtils::sampleMicro();
            _statistics.incLongAttribute(Statistics::TIME_SIMPLEX_STEPS_MICRO,
                                         TimeUtils::timePassed(start, end));
            return false;
        } else if (!_costFunctionManager->costFunctionJustComputed()) {
            // This failure might have resulted from a corrupt cost function.
            ASSERT(_costFunctionManager->getCostFunctionStatus() ==
                   ICostFunctionManager::COST_FUNCTION_UPDATED);
            _costFunctionManager->invalidateCostFunction();
            struct timespec end = TimeUtils::sampleMicro();
            _statistics.incLongAttribute(Statistics::TIME_SIMPLEX_STEPS_MICRO,
                                         TimeUtils::timePassed(start, end));
            return false;
        } else {
            // Cost function is fresh --- failure is real.
            struct timespec end = TimeUtils::sampleMicro();
            _statistics.incLongAttribute(Statistics::TIME_SIMPLEX_STEPS_MICRO,
                                         TimeUtils::timePassed(start, end));
            if (_tableau->isOptimizing()) {
                // The current solution is optimal.
                return true;
            } else
                throw InfeasibleQueryException();
        }
    }

    // Set the best choice in the tableau
    _tableau->setEnteringVariableIndex(bestEntering);
    _tableau->setLeavingVariableIndex(bestLeaving);
    _tableau->setChangeColumn(_work);
    _tableau->setChangeRatio(bestChangeRatio);

    bool fakePivot = _tableau->performingFakePivot();

    if (!fakePivot &&
        bestPivotEntry < GlobalConfiguration::ACCEPTABLE_SIMPLEX_PIVOT_THRESHOLD) {
        /*
          Despite our efforts, we are stuck with a small pivot. If basis factorization
          isn't fresh, refresh it and terminate this step - perhaps in the next iteration
          a better pivot will be found
        */
        if (!_tableau->basisMatrixAvailable()) {
            _tableau->refreshBasisFactorization();
            return false;
        }

        _statistics.incLongAttribute(Statistics::NUM_SIMPLEX_UNSTABLE_PIVOTS);
    }

    if (!fakePivot) {
        _tableau->computePivotRow();
        _rowBoundTightener->examinePivotRow();
    }

    // Perform the actual pivot
    _activeEntryStrategy->prePivotHook(_tableau, fakePivot);
    _tableau->performPivot();
    _activeEntryStrategy->postPivotHook(_tableau, fakePivot);

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute(Statistics::TIME_SIMPLEX_STEPS_MICRO, TimeUtils::timePassed(start, end));
    return false;
}

void Engine::fixViolatedPlConstraintIfPossible() {
    List<PiecewiseLinearConstraint::Fix> fixes;

    if (GlobalConfiguration::USE_SMART_FIX)
        fixes = _plConstraintToFix->getSmartFixes(_tableau);
    else
        fixes = _plConstraintToFix->getPossibleFixes();

    // First, see if we can fix without pivoting. We are looking for a fix concerning a
    // non-basic variable, that doesn't set that variable out-of-bounds.
    for (const auto &fix: fixes) {
        if (!_tableau->isBasic(fix._variable)) {
            if (_tableau->checkValueWithinBounds(fix._variable, fix._value)) {
                _tableau->setNonBasicAssignment(fix._variable, fix._value, true);
                return;
            }
        }
    }

    // No choice, have to pivot. Look for a fix concerning a basic variable, that
    // doesn't set that variable out-of-bounds. If smart-fix is enabled and implemented,
    // we should probably not reach this point.
    bool found = false;
    auto it = fixes.begin();
    while (!found && it != fixes.end()) {
        if (_tableau->isBasic(it->_variable)) {
            if (_tableau->checkValueWithinBounds(it->_variable, it->_value)) {
                found = true;
            }
        }
        if (!found) {
            ++it;
        }
    }

    // If we couldn't find an eligible fix, give up
    if (!found)
        return;

    PiecewiseLinearConstraint::Fix fix = *it;
    ASSERT(_tableau->isBasic(fix._variable));

    TableauRow row(_tableau->getN() - _tableau->getM());
    _tableau->getTableauRow(_tableau->variableToIndex(fix._variable), &row);

    // Pick the variable with the largest coefficient in this row for pivoting,
    // to increase numerical stability.
    unsigned bestCandidate = row._row[0]._var;
    double bestValue = FloatUtils::abs(row._row[0]._coefficient);

    unsigned n = _tableau->getN();
    unsigned m = _tableau->getM();
    for (unsigned i = 1; i < n - m; ++i) {
        double contenderValue = FloatUtils::abs(row._row[i]._coefficient);
        if (FloatUtils::gt(contenderValue, bestValue)) {
            bestValue = contenderValue;
            bestCandidate = row._row[i]._var;
        }
    }

    if (FloatUtils::isZero(bestValue)) {
        // This can happen, e.g., if we have an equation x = 5, and is legal behavior.
        return;
    }

    // Switch between nonBasic and the variable we need to fix
    _tableau->setEnteringVariableIndex(_tableau->variableToIndex(bestCandidate));
    _tableau->setLeavingVariableIndex(_tableau->variableToIndex(fix._variable));

    // Make sure the change column and pivot row are up-to-date - strategies
    // such as projected steepest edge need these for their internal updates.
    _tableau->computeChangeColumn();
    _tableau->computePivotRow();

    _activeEntryStrategy->prePivotHook(_tableau, false);
    _tableau->performDegeneratePivot();
    _activeEntryStrategy->postPivotHook(_tableau, false);

    ASSERT(!_tableau->isBasic(fix._variable));
    _tableau->setNonBasicAssignment(fix._variable, fix._value, true);
}

bool Engine::processInputQuery(InputQuery &inputQuery) {
    return processInputQuery(inputQuery, GlobalConfiguration::PREPROCESS_INPUT_QUERY);
}

void Engine::informConstraintsOfInitialBounds(InputQuery &inputQuery) const {
    for (const auto &plConstraint: inputQuery.getPiecewiseLinearConstraints()) {
        List<unsigned> variables = plConstraint->getParticipatingVariables();
        for (unsigned variable: variables) {
            plConstraint->notifyLowerBound(variable, inputQuery.getLowerBound(variable));
            plConstraint->notifyUpperBound(variable, inputQuery.getUpperBound(variable));
        }
    }

    for (const auto &tsConstraint: inputQuery.getTranscendentalConstraints()) {
        List<unsigned> variables = tsConstraint->getParticipatingVariables();
        for (unsigned variable: variables) {
            tsConstraint->notifyLowerBound(variable, inputQuery.getLowerBound(variable));
            tsConstraint->notifyUpperBound(variable, inputQuery.getUpperBound(variable));
        }
    }
}

void Engine::invokePreprocessor(const InputQuery &inputQuery, bool preprocess) {
    if (_verbosity > 0)
        printf("Engine::processInputQuery: Input query (before preprocessing): "
               "%u equations, %u variables\n",
               inputQuery.getEquations().size(),
               inputQuery.getNumberOfVariables());

    // If processing is enabled, invoke the preprocessor
    _preprocessingEnabled = preprocess;
    if (_preprocessingEnabled)
        _preprocessedQuery = _preprocessor.preprocess
                (inputQuery, GlobalConfiguration::PREPROCESSOR_ELIMINATE_VARIABLES);
    else
        _preprocessedQuery = std::unique_ptr<InputQuery>
                (new InputQuery(inputQuery));

    if (_verbosity > 0)
        printf("Engine::processInputQuery: Input query (after preprocessing): "
               "%u equations, %u variables\n\n",
               _preprocessedQuery->getEquations().size(),
               _preprocessedQuery->getNumberOfVariables());

    unsigned infiniteBounds = _preprocessedQuery->countInfiniteBounds();
    if (infiniteBounds != 0) {
        _exitCode = Engine::ERROR;
        throw MarabouError(MarabouError::UNBOUNDED_VARIABLES_NOT_YET_SUPPORTED,
                           Stringf("Error! Have %u infinite bounds", infiniteBounds).ascii());
    }
}

void Engine::printInputBounds(const InputQuery &inputQuery) const {
    printf("Input bounds:\n");
    for (unsigned i = 0; i < inputQuery.getNumInputVariables(); ++i) {
        unsigned variable = inputQuery.inputVariableByIndex(i);
        double lb, ub;
        bool fixed = false;
        if (_preprocessingEnabled) {
            // Fixed variables are easy: return the value they've been fixed to.
            if (_preprocessor.variableIsFixed(variable)) {
                fixed = true;
                lb = _preprocessor.getFixedValue(variable);
                ub = lb;
            } else {
                // Has the variable been merged into another?
                while (_preprocessor.variableIsMerged(variable))
                    variable = _preprocessor.getMergedIndex(variable);

                // We know which variable to look for, but it may have been assigned
                // a new index, due to variable elimination
                variable = _preprocessor.getNewIndex(variable);

                lb = _preprocessedQuery->getLowerBound(variable);
                ub = _preprocessedQuery->getUpperBound(variable);
            }
        } else {
            lb = inputQuery.getLowerBound(variable);
            ub = inputQuery.getUpperBound(variable);
        }

        printf("\tx%u: [%8.4lf, %8.4lf] %s\n", i, lb, ub, fixed ? "[FIXED]" : "");
    }
    printf("\n");
}

void Engine::storeEquationsInDegradationChecker() {
    _degradationChecker.storeEquations(*_preprocessedQuery);
}

double *Engine::createConstraintMatrix() {
    const List<Equation> &equations(_preprocessedQuery->getEquations());
    unsigned m = equations.size();
    unsigned n = _preprocessedQuery->getNumberOfVariables();

    // Step 1: create a constraint matrix from the equations
    double *constraintMatrix = new double[n * m];
    if (!constraintMatrix)
        throw MarabouError(MarabouError::ALLOCATION_FAILED, "Engine::constraintMatrix");
    std::fill_n(constraintMatrix, n * m, 0.0);

    unsigned equationIndex = 0;
    for (const auto &equation: equations) {
        if (equation._type != Equation::EQ) {
            _exitCode = Engine::ERROR;
            throw MarabouError(MarabouError::NON_EQUALITY_INPUT_EQUATION_DISCOVERED);
        }

        for (const auto &addend: equation._addends)
            constraintMatrix[equationIndex * n + addend._variable] = addend._coefficient;

        ++equationIndex;
    }

    return constraintMatrix;
}

void Engine::removeRedundantEquations(const double *constraintMatrix) {
    const List<Equation> &equations(_preprocessedQuery->getEquations());
    unsigned m = equations.size();
    unsigned n = _preprocessedQuery->getNumberOfVariables();

    // Step 1: analyze the matrix to identify redundant rows
    AutoConstraintMatrixAnalyzer analyzer;
    analyzer->analyze(constraintMatrix, m, n);

    ENGINE_LOG(Stringf("Number of redundant rows: %u out of %u",
                       analyzer->getRedundantRows().size(), m).ascii());

    // Step 2: remove any equations corresponding to redundant rows
    Set<unsigned> redundantRows = analyzer->getRedundantRows();

    if (!redundantRows.empty()) {
        _preprocessedQuery->removeEquationsByIndex(redundantRows);
        m = equations.size();
    }
}

void Engine::selectInitialVariablesForBasis(const double *constraintMatrix, List<unsigned> &initialBasis,
                                            List<unsigned> &basicRows) {
    /*
      This method permutes rows and columns in the constraint matrix (prior
      to the addition of auxiliary variables), in order to obtain a set of
      column that constitue a lower triangular matrix. The variables
      corresponding to the columns of this matrix join the initial basis.

      (It is possible that not enough variables are obtained this way, in which
      case the initial basis will have to be augmented later).
    */

    const List<Equation> &equations(_preprocessedQuery->getEquations());

    unsigned m = equations.size();
    unsigned n = _preprocessedQuery->getNumberOfVariables();

    // Trivial case, or if a trivial basis is requested
    if ((m == 0) || (n == 0) || GlobalConfiguration::ONLY_AUX_INITIAL_BASIS) {
        for (unsigned i = 0; i < m; ++i)
            basicRows.append(i);

        return;
    }

    unsigned *nnzInRow = new unsigned[m];
    unsigned *nnzInColumn = new unsigned[n];

    std::fill_n(nnzInRow, m, 0);
    std::fill_n(nnzInColumn, n, 0);

    unsigned *columnOrdering = new unsigned[n];
    unsigned *rowOrdering = new unsigned[m];

    for (unsigned i = 0; i < m; ++i)
        rowOrdering[i] = i;

    for (unsigned i = 0; i < n; ++i)
        columnOrdering[i] = i;

    // Initialize the counters
    for (unsigned i = 0; i < m; ++i) {
        for (unsigned j = 0; j < n; ++j) {
            if (!FloatUtils::isZero(constraintMatrix[i * n + j])) {
                ++nnzInRow[i];
                ++nnzInColumn[j];
            }
        }
    }

    DEBUG({
              for (unsigned i = 0; i < m; ++i) {
                  ASSERT(nnzInRow[i] > 0);
              }
          });

    unsigned numExcluded = 0;
    unsigned numTriangularRows = 0;
    unsigned temp;

    while (numExcluded + numTriangularRows < n) {
        // Do we have a singleton row?
        unsigned singletonRow = m;
        for (unsigned i = numTriangularRows; i < m; ++i) {
            if (nnzInRow[i] == 1) {
                singletonRow = i;
                break;
            }
        }

        if (singletonRow < m) {
            // Have a singleton row! Swap it to the top and update counters
            temp = rowOrdering[singletonRow];
            rowOrdering[singletonRow] = rowOrdering[numTriangularRows];
            rowOrdering[numTriangularRows] = temp;

            temp = nnzInRow[numTriangularRows];
            nnzInRow[numTriangularRows] = nnzInRow[singletonRow];
            nnzInRow[singletonRow] = temp;

            // Find the non-zero entry in the row and swap it to the diagonal
            DEBUG(bool foundNonZero = false);
            for (unsigned i = numTriangularRows; i < n - numExcluded; ++i) {
                if (!FloatUtils::isZero(constraintMatrix[rowOrdering[numTriangularRows] * n + columnOrdering[i]])) {
                    temp = columnOrdering[i];
                    columnOrdering[i] = columnOrdering[numTriangularRows];
                    columnOrdering[numTriangularRows] = temp;

                    temp = nnzInColumn[numTriangularRows];
                    nnzInColumn[numTriangularRows] = nnzInColumn[i];
                    nnzInColumn[i] = temp;

                    DEBUG(foundNonZero = true);
                    break;
                }
            }

            ASSERT(foundNonZero);

            // Remove all entries under the diagonal entry from the row counters
            for (unsigned i = numTriangularRows + 1; i < m; ++i) {
                if (!FloatUtils::isZero(constraintMatrix[rowOrdering[i] * n + columnOrdering[numTriangularRows]]))
                    --nnzInRow[i];
            }

            ++numTriangularRows;
        } else {
            // No singleton rows. Exclude the densest column
            unsigned maxDensity = nnzInColumn[numTriangularRows];
            unsigned column = numTriangularRows;

            for (unsigned i = numTriangularRows; i < n - numExcluded; ++i) {
                if (nnzInColumn[i] > maxDensity) {
                    maxDensity = nnzInColumn[i];
                    column = i;
                }
            }

            // Update the row counters to account for the excluded column
            for (unsigned i = numTriangularRows; i < m; ++i) {
                double element = constraintMatrix[rowOrdering[i] * n + columnOrdering[column]];
                if (!FloatUtils::isZero(element)) {
                    ASSERT(nnzInRow[i] > 1);
                    --nnzInRow[i];
                }
            }

            columnOrdering[column] = columnOrdering[n - 1 - numExcluded];
            nnzInColumn[column] = nnzInColumn[n - 1 - numExcluded];
            ++numExcluded;
        }
    }

    // Final basis: diagonalized columns + non-diagonalized rows
    List<unsigned> result;

    for (unsigned i = 0; i < numTriangularRows; ++i) {
        initialBasis.append(columnOrdering[i]);
    }

    for (unsigned i = numTriangularRows; i < m; ++i) {
        basicRows.append(rowOrdering[i]);
    }

    // Cleanup
    delete[] nnzInRow;
    delete[] nnzInColumn;
    delete[] columnOrdering;
    delete[] rowOrdering;
}

void Engine::addAuxiliaryVariables() {
    List<Equation> &equations(_preprocessedQuery->getEquations());

    unsigned m = equations.size();
    unsigned originalN = _preprocessedQuery->getNumberOfVariables();
    unsigned n = originalN + m;

    _preprocessedQuery->setNumberOfVariables(n);

    // Add auxiliary variables to the equations and set their bounds
    unsigned count = 0;
    for (auto &eq: equations) {
        unsigned auxVar = originalN + count;
        eq.addAddend(-1, auxVar);
        _preprocessedQuery->setLowerBound(auxVar, eq._scalar);
        _preprocessedQuery->setUpperBound(auxVar, eq._scalar);
        eq.setScalar(0);

        ++count;
    }
}

void Engine::augmentInitialBasisIfNeeded(List<unsigned> &initialBasis, const List<unsigned> &basicRows) {
    unsigned m = _preprocessedQuery->getEquations().size();
    unsigned n = _preprocessedQuery->getNumberOfVariables();
    unsigned originalN = n - m;

    if (initialBasis.size() != m) {
        for (const auto &basicRow: basicRows)
            initialBasis.append(basicRow + originalN);
    }
}

void Engine::initializeTableau(const double *constraintMatrix, const List<unsigned> &initialBasis) {
    const List<Equation> &equations(_preprocessedQuery->getEquations());
    unsigned m = equations.size();
    unsigned n = _preprocessedQuery->getNumberOfVariables();

    _tableau->setDimensions(m, n);

    adjustWorkMemorySize();

    unsigned equationIndex = 0;
    for (const auto &equation: equations) {
        _tableau->setRightHandSide(equationIndex, equation._scalar);
        ++equationIndex;
    }

    // Populate constriant matrix
    _tableau->setConstraintMatrix(constraintMatrix);

    _tableau->registerToWatchAllVariables(_rowBoundTightener);
    _tableau->registerResizeWatcher(_rowBoundTightener);

    _rowBoundTightener->setDimensions();

    initializeBoundsAndConstraintWatchersInTableau(n);

    _tableau->initializeTableau(initialBasis);

    _costFunctionManager->initialize();
    _tableau->registerCostFunctionManager(_costFunctionManager);
    _activeEntryStrategy->initialize(_tableau);
}

void Engine::initializeBoundsAndConstraintWatchersInTableau(unsigned
                                                            numberOfVariables) {
    _plConstraints = _preprocessedQuery->getPiecewiseLinearConstraints();
    for (const auto &constraint: _plConstraints) {
        constraint->registerAsWatcher(_tableau);
        constraint->setStatistics(&_statistics);
    }

    _tsConstraints = _preprocessedQuery->getTranscendentalConstraints();
    for (const auto &constraint: _tsConstraints) {
        constraint->registerAsWatcher(_tableau);
        constraint->setStatistics(&_statistics);
    }

    for (unsigned i = 0; i < numberOfVariables; ++i) {
        _tableau->setLowerBound(i, _preprocessedQuery->getLowerBound(i));
        _tableau->setUpperBound(i, _preprocessedQuery->getUpperBound(i));
    }

    _statistics.setUnsignedAttribute(Statistics::NUM_PL_CONSTRAINTS,
                                     _plConstraints.size());
}

void Engine::initializeNetworkLevelReasoning() {
    _networkLevelReasoner = _preprocessedQuery->getNetworkLevelReasoner();

    if (_networkLevelReasoner)
        _networkLevelReasoner->setTableau(_tableau);
}

bool Engine::processInputQuery(InputQuery &inputQuery, bool preprocess) {
    ENGINE_LOG("processInputQuery starting\n");
    struct timespec start = TimeUtils::sampleMicro();

    try {
        informConstraintsOfInitialBounds(inputQuery);
        invokePreprocessor(inputQuery, preprocess);
        if (_verbosity > 0)
            printInputBounds(inputQuery);

        initializeNetworkLevelReasoning();
        if (preprocess) {
            performSymbolicBoundTightening(&(*_preprocessedQuery));
            performSimulation();
            performMILPSolverBoundedTightening(&(*_preprocessedQuery));
        }

        if (GlobalConfiguration::PL_CONSTRAINTS_ADD_AUX_EQUATIONS_AFTER_PREPROCESSING)
            for (auto &plConstraint: _preprocessedQuery->getPiecewiseLinearConstraints())
                plConstraint->addAuxiliaryEquationsAfterPreprocessing
                        (*_preprocessedQuery);

        if (_lpSolverType == LPSolverType::NATIVE) {
            double *constraintMatrix = createConstraintMatrix();
            removeRedundantEquations(constraintMatrix);

            // The equations have changed, recreate the constraint matrix
            delete[] constraintMatrix;
            constraintMatrix = createConstraintMatrix();

            List<unsigned> initialBasis;
            List<unsigned> basicRows;
            selectInitialVariablesForBasis(constraintMatrix, initialBasis, basicRows);
            addAuxiliaryVariables();
            augmentInitialBasisIfNeeded(initialBasis, basicRows);

            storeEquationsInDegradationChecker();

            // The equations have changed, recreate the constraint matrix
            delete[] constraintMatrix;
            constraintMatrix = createConstraintMatrix();

            unsigned n = _preprocessedQuery->getNumberOfVariables();
            _boundManager.initialize(n);

            initializeTableau(constraintMatrix, initialBasis);

            delete[] constraintMatrix;
        } else {
            ASSERT(_lpSolverType == LPSolverType::GUROBI);

            ASSERT(GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH == true);

            if (_verbosity > 0)
                printf("Using Gurobi to solve LP...\n");

            _gurobi = std::unique_ptr<GurobiWrapper>(new GurobiWrapper());
            _milpEncoder = std::unique_ptr<MILPEncoder>
                    (new MILPEncoder(*_tableau));
            _milpEncoder->setStatistics(&_statistics);
            _tableau->setGurobi(&(*_gurobi));

            unsigned n = _preprocessedQuery->getNumberOfVariables();
            unsigned m = _preprocessedQuery->getEquations().size();
            // Only use BoundManager to store the bounds.
            _boundManager.initialize(n);
            _tableau->setDimensions(m, n);
            initializeBoundsAndConstraintWatchersInTableau(n);

            for (const auto &constraint: _plConstraints) {
                constraint->registerGurobi(&(*_gurobi));
            }
        }

        for (const auto &constraint: _plConstraints) {
            constraint->registerTableau(_tableau);
        }

        if (Options::get()->getBool(Options::DUMP_BOUNDS))
            _networkLevelReasoner->dumpBounds();

        if (GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH) {
            _soiManager = std::unique_ptr<SumOfInfeasibilitiesManager>
                    (new SumOfInfeasibilitiesManager(*_preprocessedQuery,
                                                     *_tableau));
            _soiManager->setStatistics(&_statistics);
        }

        if (GlobalConfiguration::WARM_START)
            warmStart();

        decideBranchingHeuristics();

        struct timespec end = TimeUtils::sampleMicro();
        _statistics.setLongAttribute(Statistics::PREPROCESSING_TIME_MICRO,
                                     TimeUtils::timePassed(start, end));

        if (!_tableau->allBoundsValid()) {
            // Some variable bounds are invalid, so the query is unsat
            throw InfeasibleQueryException();
        }
    }
    catch (const InfeasibleQueryException &) {
        ENGINE_LOG("processInputQuery done\n");

        struct timespec end = TimeUtils::sampleMicro();
        _statistics.setLongAttribute(Statistics::PREPROCESSING_TIME_MICRO,
                                     TimeUtils::timePassed(start, end));

        _exitCode = Engine::UNSAT;
        return false;
    }

    ENGINE_LOG("processInputQuery done\n");

    DEBUG({
              // Initially, all constraints should be active
              for (const auto &plc: _plConstraints) {
                  ASSERT(plc->isActive());
              }
          });

    _smtCore.storeDebuggingSolution(_preprocessedQuery->_debuggingSolution);
    return true;
}

void Engine::performMILPSolverBoundedTightening(InputQuery *inputQuery) {
    if (_networkLevelReasoner && Options::get()->gurobiEnabled()) {
        // Obtain from and store bounds into inputquery if it is not null.
        if (inputQuery)
            _networkLevelReasoner->obtainCurrentBounds(*inputQuery);
        else
            _networkLevelReasoner->obtainCurrentBounds();

        // TODO: Remove this block after getting ready to support sigmoid with MILP Bound Tightening.
        if (Options::get()->getMILPSolverBoundTighteningType() != MILPSolverBoundTighteningType::NONE
            && _preprocessedQuery->getTranscendentalConstraints().size() > 0)
            throw MarabouError(MarabouError::FEATURE_NOT_YET_SUPPORTED,
                               "Marabou doesn't support sigmoid with MILP Bound Tightening");

        switch (Options::get()->getMILPSolverBoundTighteningType()) {
            case MILPSolverBoundTighteningType::LP_RELAXATION:
            case MILPSolverBoundTighteningType::LP_RELAXATION_INCREMENTAL:
                _networkLevelReasoner->lpRelaxationPropagation();
                break;

            case MILPSolverBoundTighteningType::MILP_ENCODING:
            case MILPSolverBoundTighteningType::MILP_ENCODING_INCREMENTAL:
                _networkLevelReasoner->MILPPropagation();
                break;
            case MILPSolverBoundTighteningType::ITERATIVE_PROPAGATION:
                _networkLevelReasoner->iterativePropagation();
                break;
            case MILPSolverBoundTighteningType::NONE:
                return;
        }
        List<Tightening> tightenings;
        _networkLevelReasoner->getConstraintTightenings(tightenings);


        if (inputQuery) {
            for (const auto &tightening: tightenings) {

                if (tightening._type == Tightening::LB &&
                    FloatUtils::gt(tightening._value,
                                   inputQuery->getLowerBound
                                           (tightening._variable)))
                    inputQuery->setLowerBound(tightening._variable,
                                              tightening._value);
                if (tightening._type == Tightening::UB &&
                    FloatUtils::lt(tightening._value,
                                   inputQuery->getUpperBound
                                           (tightening._variable)))
                    inputQuery->setUpperBound(tightening._variable,
                                              tightening._value);
            }
        } else {
            for (const auto &tightening: tightenings) {
                if (tightening._type == Tightening::LB)
                    _tableau->tightenLowerBound(tightening._variable, tightening._value);

                else if (tightening._type == Tightening::UB)
                    _tableau->tightenUpperBound(tightening._variable, tightening._value);
            }
        }
    }
}

void Engine::performMILPSolverBoundedTighteningForSingleLayer(unsigned targetIndex) {
    if (_networkLevelReasoner && _isGurobyEnabled && !_performLpTighteningAfterSplit
        && _milpSolverBoundTighteningType != MILPSolverBoundTighteningType::NONE) {
        _networkLevelReasoner->obtainCurrentBounds();
        _networkLevelReasoner->clearConstraintTightenings();

        switch (_milpSolverBoundTighteningType) {
            case MILPSolverBoundTighteningType::LP_RELAXATION:
                _networkLevelReasoner->LPTighteningForOneLayer(targetIndex);
                break;
            case MILPSolverBoundTighteningType::LP_RELAXATION_INCREMENTAL:
                return;

            case MILPSolverBoundTighteningType::MILP_ENCODING:
                _networkLevelReasoner->MILPTighteningForOneLayer(targetIndex);
                break;
            case MILPSolverBoundTighteningType::MILP_ENCODING_INCREMENTAL:
                return;

            case MILPSolverBoundTighteningType::ITERATIVE_PROPAGATION:
            case MILPSolverBoundTighteningType::NONE:
                return;
        }
        List<Tightening> tightenings;
        _networkLevelReasoner->getConstraintTightenings(tightenings);

        for (const auto &tightening: tightenings) {
            if (tightening._type == Tightening::LB)
                _tableau->tightenLowerBound(tightening._variable, tightening._value);

            else if (tightening._type == Tightening::UB)
                _tableau->tightenUpperBound(tightening._variable, tightening._value);
        }
    }
}

void Engine::extractSolution(InputQuery &inputQuery) {
    if (_solveWithMILP) {
        extractSolutionFromGurobi(inputQuery);
        return;
    }

    for (unsigned i = 0; i < inputQuery.getNumberOfVariables(); ++i) {
        if (_preprocessingEnabled) {
            // Has the variable been merged into another?
            unsigned variable = i;
            while (_preprocessor.variableIsMerged(variable))
                variable = _preprocessor.getMergedIndex(variable);

            // Fixed variables are easy: return the value they've been fixed to.
            if (_preprocessor.variableIsFixed(variable)) {
                inputQuery.setSolutionValue(i, _preprocessor.getFixedValue(variable));
                inputQuery.setLowerBound(i, _preprocessor.getFixedValue(variable));
                inputQuery.setUpperBound(i, _preprocessor.getFixedValue(variable));
                continue;
            }

            // We know which variable to look for, but it may have been assigned
            // a new index, due to variable elimination
            variable = _preprocessor.getNewIndex(variable);

            // Finally, set the assigned value
            inputQuery.setSolutionValue(i, _tableau->getValue(variable));
            inputQuery.setLowerBound(i, _tableau->getLowerBound(variable));
            inputQuery.setUpperBound(i, _tableau->getUpperBound(variable));
        } else {
            inputQuery.setSolutionValue(i, _tableau->getValue(i));
            inputQuery.setLowerBound(i, _tableau->getLowerBound(i));
            inputQuery.setUpperBound(i, _tableau->getUpperBound(i));
        }
    }
}

bool Engine::allVarsWithinBounds() const {
    if (_lpSolverType == LPSolverType::GUROBI) {
        ASSERT(_gurobi);
        return _gurobi->haveFeasibleSolution();
    } else
        return !_tableau->existsBasicOutOfBounds();
}

void Engine::collectViolatedPlConstraints() {
    _violatedPlConstraints.clear();
    for (const auto &constraint: _plConstraints) {
        if (constraint->isActive() && !constraint->satisfied())
            _violatedPlConstraints.append(constraint);
    }
}

bool Engine::allPlConstraintsHold() {
    return _violatedPlConstraints.empty();
}

void Engine::selectViolatedPlConstraint() {
    ASSERT(!_violatedPlConstraints.empty());

    _plConstraintToFix = _smtCore.chooseViolatedConstraintForFixing(_violatedPlConstraints);

    ASSERT(_plConstraintToFix);
}

void Engine::reportPlViolation() {
    _smtCore.reportViolatedConstraint(_plConstraintToFix);
}

void Engine::storeState(EngineState &state, TableauStateStorageLevel level) const {
    _tableau->storeState(state._tableauState, level);
    state._tableauStateStorageLevel = level;

    for (const auto &constraint: _plConstraints) {
        state._plConstraintToState[constraint] = constraint->duplicateConstraint();
    }
    state._numPlConstraintsDisabledByValidSplits = _numPlConstraintsDisabledByValidSplits;
}

void Engine::restoreState(const EngineState &state) {
    ENGINE_LOG("Restore state starting");

    if (state._tableauStateStorageLevel == TableauStateStorageLevel::STORE_NONE)
        throw MarabouError(MarabouError::RESTORING_ENGINE_FROM_INVALID_STATE);

    ENGINE_LOG("\tRestoring tableau state");
    _tableau->restoreState(state._tableauState,
                           state._tableauStateStorageLevel);

    ENGINE_LOG("\tRestoring constraint states");
    for (auto &constraint: _plConstraints) {
        if (!state._plConstraintToState.exists(constraint))
            throw MarabouError(MarabouError::MISSING_PL_CONSTRAINT_STATE);

        constraint->restoreState(state._plConstraintToState[constraint]);
    }

    _numPlConstraintsDisabledByValidSplits = state._numPlConstraintsDisabledByValidSplits;

    if (_lpSolverType == LPSolverType::NATIVE) {
        // Make sure the data structures are initialized to the correct size
        _rowBoundTightener->setDimensions();
        adjustWorkMemorySize();
        _activeEntryStrategy->resizeHook(_tableau);
        _costFunctionManager->initialize();
    }

    // Reset the violation counts in the SMT core
    _smtCore.resetSplitConditions();
}

void Engine::setNumPlConstraintsDisabledByValidSplits(unsigned numConstraints) {
    _numPlConstraintsDisabledByValidSplits = numConstraints;
}

bool Engine::attemptToMergeVariables(unsigned x1, unsigned x2) {
    /*
      First, we need to ensure that the variables are both non-basic.
    */

    unsigned n = _tableau->getN();
    unsigned m = _tableau->getM();

    if (_tableau->isBasic(x1)) {
        TableauRow x1Row(n - m);
        _tableau->getTableauRow(_tableau->variableToIndex(x1), &x1Row);

        bool found = false;
        double bestCoefficient = 0.0;
        unsigned nonBasic = 0;
        for (unsigned i = 0; i < n - m; ++i) {
            if (x1Row._row[i]._var != x2) {
                double contender = FloatUtils::abs(x1Row._row[i]._coefficient);
                if (FloatUtils::gt(contender, bestCoefficient)) {
                    found = true;
                    nonBasic = x1Row._row[i]._var;
                    bestCoefficient = contender;
                }
            }
        }

        if (!found)
            return false;

        _tableau->setEnteringVariableIndex(_tableau->variableToIndex(nonBasic));
        _tableau->setLeavingVariableIndex(_tableau->variableToIndex(x1));

        // Make sure the change column and pivot row are up-to-date - strategies
        // such as projected steepest edge need these for their internal updates.
        _tableau->computeChangeColumn();
        _tableau->computePivotRow();

        _activeEntryStrategy->prePivotHook(_tableau, false);
        _tableau->performDegeneratePivot();
        _activeEntryStrategy->postPivotHook(_tableau, false);
    }

    if (_tableau->isBasic(x2)) {
        TableauRow x2Row(n - m);
        _tableau->getTableauRow(_tableau->variableToIndex(x2), &x2Row);

        bool found = false;
        double bestCoefficient = 0.0;
        unsigned nonBasic = 0;
        for (unsigned i = 0; i < n - m; ++i) {
            if (x2Row._row[i]._var != x1) {
                double contender = FloatUtils::abs(x2Row._row[i]._coefficient);
                if (FloatUtils::gt(contender, bestCoefficient)) {
                    found = true;
                    nonBasic = x2Row._row[i]._var;
                    bestCoefficient = contender;
                }
            }
        }

        if (!found)
            return false;

        _tableau->setEnteringVariableIndex(_tableau->variableToIndex(nonBasic));
        _tableau->setLeavingVariableIndex(_tableau->variableToIndex(x2));

        // Make sure the change column and pivot row are up-to-date - strategies
        // such as projected steepest edge need these for their internal updates.
        _tableau->computeChangeColumn();
        _tableau->computePivotRow();

        _activeEntryStrategy->prePivotHook(_tableau, false);
        _tableau->performDegeneratePivot();
        _activeEntryStrategy->postPivotHook(_tableau, false);
    }

    // Both variables are now non-basic, so we can merge their columns
    _tableau->mergeColumns(x1, x2);
    DEBUG(_tableau->verifyInvariants());

    // Reset the entry strategy
    _activeEntryStrategy->initialize(_tableau);

    return true;
}

void Engine::applySplit(const PiecewiseLinearCaseSplit &split) {
    ENGINE_LOG("");
    ENGINE_LOG("Applying a split. ");
    CenterStatics time("applySplit");

    DEBUG(_tableau->verifyInvariants());

    List<Tightening> bounds = split.getBoundTightenings();
    List<Equation> equations = split.getEquations();

    // We assume that case splits only apply new bounds but do not apply
    // new equations. This can always be made possible.
    if (_lpSolverType != LPSolverType::NATIVE && equations.size() > 0)
        throw MarabouError(MarabouError::FEATURE_NOT_YET_SUPPORTED,
                           "Can only update bounds when using non-native"
                           "simplex engine!");

    for (auto &equation: equations) {
        /*
          First, adjust the equation if any variables have been merged.
          E.g., if the equation is x1 + x2 + x3 = 0, and x1 and x2 have been
          merged, the equation becomes 2x1 + x3 = 0
        */
        for (auto &addend: equation._addends)
            addend._variable = _tableau->getVariableAfterMerging(addend._variable);

        List<Equation::Addend>::iterator addend;
        List<Equation::Addend>::iterator otherAddend;

        addend = equation._addends.begin();
        while (addend != equation._addends.end()) {
            otherAddend = addend;
            ++otherAddend;

            while (otherAddend != equation._addends.end()) {
                if (otherAddend->_variable == addend->_variable) {
                    addend->_coefficient += otherAddend->_coefficient;
                    otherAddend = equation._addends.erase(otherAddend);
                } else
                    ++otherAddend;
            }

            if (FloatUtils::isZero(addend->_coefficient))
                addend = equation._addends.erase(addend);
            else
                ++addend;
        }

        /*
          In the general case, we just add the new equation to the tableau.
          However, we also support a very common case: equations of the form
          x1 = x2, which are common, e.g., with ReLUs. For these equations we
          may be able to merge two columns of the tableau.
        */
        unsigned x1, x2;
        bool canMergeColumns =
                // Only if the flag is on
                GlobalConfiguration::USE_COLUMN_MERGING_EQUATIONS &&
                // Only if the equation has the correct form
                equation.isVariableMergingEquation(x1, x2) &&
                // And only if the variables are not out of bounds
                (!_tableau->isBasic(x1) ||
                 !_tableau->basicOutOfBounds(_tableau->variableToIndex(x1)))
                &&
                (!_tableau->isBasic(x2) ||
                 !_tableau->basicOutOfBounds(_tableau->variableToIndex(x2)));

        bool columnsSuccessfullyMerged = false;
        if (canMergeColumns)
            columnsSuccessfullyMerged = attemptToMergeVariables(x1, x2);

        if (!columnsSuccessfullyMerged) {
            // General case: add a new equation to the tableau
            unsigned auxVariable = _tableau->addEquation(equation);
            _activeEntryStrategy->resizeHook(_tableau);

            switch (equation._type) {
                case Equation::GE:
                    bounds.append(Tightening(auxVariable, 0.0, Tightening::UB));
                    break;

                case Equation::LE:
                    bounds.append(Tightening(auxVariable, 0.0, Tightening::LB));
                    break;

                case Equation::EQ:
                    bounds.append(Tightening(auxVariable, 0.0, Tightening::LB));
                    bounds.append(Tightening(auxVariable, 0.0, Tightening::UB));
                    break;

                default: ASSERT(false);
                    break;
            }
        }
    }

    if (_lpSolverType == LPSolverType::NATIVE) {
        adjustWorkMemorySize();
    }

    for (auto &bound: bounds) {
        unsigned variable = _tableau->getVariableAfterMerging(bound._variable);

        if (bound._type == Tightening::LB) {
            ENGINE_LOG(Stringf("x%u: lower bound set to %.3lf", variable, bound._value).ascii());
            _tableau->tightenLowerBound(variable, bound._value);
        } else {
            ENGINE_LOG(Stringf("x%u: upper bound set to %.3lf", variable, bound._value).ascii());
            _tableau->tightenUpperBound(variable, bound._value);
        }
    }

    DEBUG(_tableau->verifyInvariants());
    ENGINE_LOG("Done with split\n");
}

void Engine::applyBoundTightenings() {
    List<Tightening> tightenings;
    _boundManager.getTightenings(tightenings);

    for (const auto &tightening: tightenings) {
        if (tightening._type == Tightening::LB)
            _tableau->tightenLowerBound(tightening._variable, tightening._value);
        else
            _tableau->tightenUpperBound(tightening._variable, tightening._value);
    }
}

void Engine::applyAllRowTightenings() {
    applyBoundTightenings();
}

void Engine::applyAllConstraintTightenings() {
    applyBoundTightenings();
}

void Engine::applyAllBoundTightenings() {
    struct timespec start = TimeUtils::sampleMicro();

    if (_lpSolverType == LPSolverType::NATIVE)
        applyAllRowTightenings();
    applyAllConstraintTightenings();

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute(Statistics::TOTAL_TIME_APPLYING_STORED_TIGHTENINGS_MICRO,
                                 TimeUtils::timePassed(start, end));
}

bool Engine::applyAllValidConstraintCaseSplits() {
    struct timespec start = TimeUtils::sampleMicro();

    bool appliedSplit = false;
    for (auto &constraint: _plConstraints)
        if (applyValidConstraintCaseSplit(constraint))
            appliedSplit = true;

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute(Statistics::TOTAL_TIME_PERFORMING_VALID_CASE_SPLITS_MICRO,
                                 TimeUtils::timePassed(start, end));

    return appliedSplit;
}

bool Engine::applyValidConstraintCaseSplit(PiecewiseLinearConstraint *constraint) {
    if (constraint->isActive() && constraint->phaseFixed()) {
        String constraintString;
        constraint->dump(constraintString);
        ENGINE_LOG(Stringf("A constraint has become valid. Dumping constraint: %s",
                           constraintString.ascii()).ascii());

        constraint->setActiveConstraint(false);
        PiecewiseLinearCaseSplit validSplit = constraint->getValidCaseSplit();
        _smtCore.recordImpliedValidSplit(validSplit);
        applySplit(validSplit);
        recordSplit(validSplit);
        if (_soiManager)
            _soiManager->removeCostComponentFromHeuristicCost(constraint);
        ++_numPlConstraintsDisabledByValidSplits;

        return true;
    }

    return false;
}

bool Engine::shouldCheckDegradation() {
    return _statistics.getLongAttribute(Statistics::NUM_MAIN_LOOP_ITERATIONS) %
           GlobalConfiguration::DEGRADATION_CHECKING_FREQUENCY == 0;
}

bool Engine::highDegradation() {
    struct timespec start = TimeUtils::sampleMicro();

    double degradation = _degradationChecker.computeDegradation(*_tableau);
    _statistics.setDoubleAttribute(Statistics::CURRENT_DEGRADATION, degradation);
    if (FloatUtils::gt(degradation,
                       _statistics.getDoubleAttribute
                               (Statistics::MAX_DEGRADATION)))
        _statistics.setDoubleAttribute(Statistics::MAX_DEGRADATION,
                                       degradation);

    bool result = FloatUtils::gt(degradation, GlobalConfiguration::DEGRADATION_THRESHOLD);

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute(Statistics::TOTAL_TIME_DEGRADATION_CHECKING,
                                 TimeUtils::timePassed(start, end));

    // Debug
    if (result)
        printf("High degradation found!\n");
    //

    return result;
}

void Engine::tightenBoundsOnConstraintMatrix() {
    struct timespec start = TimeUtils::sampleMicro();

    if (_statistics.getLongAttribute(Statistics::NUM_MAIN_LOOP_ITERATIONS) %
        GlobalConfiguration::BOUND_TIGHTING_ON_CONSTRAINT_MATRIX_FREQUENCY == 0) {
        _rowBoundTightener->examineConstraintMatrix(true);
        _statistics.incLongAttribute
                (Statistics::NUM_BOUND_TIGHTENINGS_ON_CONSTRAINT_MATRIX);
    }

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute
            (Statistics::TOTAL_TIME_CONSTRAINT_MATRIX_BOUND_TIGHTENING_MICRO,
             TimeUtils::timePassed(start, end));
}

void Engine::explicitBasisBoundTightening() {
    struct timespec start = TimeUtils::sampleMicro();

    bool saturation = GlobalConfiguration::EXPLICIT_BOUND_TIGHTENING_UNTIL_SATURATION;

    _statistics.incLongAttribute(Statistics::NUM_BOUND_TIGHTENINGS_ON_EXPLICIT_BASIS);

    switch (GlobalConfiguration::EXPLICIT_BASIS_BOUND_TIGHTENING_TYPE) {
        case GlobalConfiguration::COMPUTE_INVERTED_BASIS_MATRIX:
            _rowBoundTightener->examineInvertedBasisMatrix(saturation);
            break;

        case GlobalConfiguration::USE_IMPLICIT_INVERTED_BASIS_MATRIX:
            _rowBoundTightener->examineImplicitInvertedBasisMatrix(saturation);
            break;

        case GlobalConfiguration::DISABLE_EXPLICIT_BASIS_TIGHTENING:
            break;
    }

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute
            (Statistics::TOTAL_TIME_EXPLICIT_BASIS_BOUND_TIGHTENING_MICRO,
             TimeUtils::timePassed(start, end));
}

void Engine::performPrecisionRestoration(PrecisionRestorer::RestoreBasics restoreBasics) {
    struct timespec start = TimeUtils::sampleMicro();

    // debug
    double before = _degradationChecker.computeDegradation(*_tableau);
    //

    _precisionRestorer.restorePrecision(*this, *_tableau, _smtCore, restoreBasics);
    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute(Statistics::TOTAL_TIME_PRECISION_RESTORATION,
                                 TimeUtils::timePassed(start, end));

    _statistics.incUnsignedAttribute(Statistics::NUM_PRECISION_RESTORATIONS);

    // debug
    double after = _degradationChecker.computeDegradation(*_tableau);
    if (_verbosity > 0)
        printf("Performing precision restoration. Degradation before: %.15lf. After: %.15lf\n",
               before,
               after);
    //

    if (highDegradation() && (restoreBasics == PrecisionRestorer::RESTORE_BASICS)) {
        // First round, with basic restoration, still resulted in high degradation.
        // Try again!
        start = TimeUtils::sampleMicro();
        _precisionRestorer.restorePrecision(*this, *_tableau, _smtCore,
                                            PrecisionRestorer::DO_NOT_RESTORE_BASICS);
        end = TimeUtils::sampleMicro();
        _statistics.incLongAttribute(Statistics::TOTAL_TIME_PRECISION_RESTORATION,
                                     TimeUtils::timePassed(start, end));
        _statistics.incUnsignedAttribute(Statistics::NUM_PRECISION_RESTORATIONS);

        // debug
        double afterSecond = _degradationChecker.computeDegradation(*_tableau);
        if (_verbosity > 0)
            printf("Performing 2nd precision restoration. Degradation before: %.15lf. After: %.15lf\n",
                   after,
                   afterSecond);

        if (highDegradation())
            throw MarabouError(MarabouError::RESTORATION_FAILED_TO_RESTORE_PRECISION);
    }
}

void Engine::storeInitialEngineState() {
    if (!_initialStateStored) {
        _precisionRestorer.storeInitialEngineState(*this);
        _initialStateStored = true;
    }
}

bool Engine::basisRestorationNeeded() const {
    return
            _basisRestorationRequired == Engine::STRONG_RESTORATION_NEEDED ||
            _basisRestorationRequired == Engine::WEAK_RESTORATION_NEEDED;
}

const Statistics *Engine::getStatistics() const {
    return &_statistics;
}

InputQuery *Engine::getInputQuery() {
    return &(*_preprocessedQuery);
}

void Engine::checkBoundCompliancyWithDebugSolution() {
    if (_smtCore.checkSkewFromDebuggingSolution()) {
        // The stack is compliant, we should not have learned any non-compliant bounds
        for (const auto &var: _preprocessedQuery->_debuggingSolution) {
            // printf( "Looking at var %u\n", var.first );

            if (FloatUtils::gt(_tableau->getLowerBound(var.first), var.second, 1e-5)) {
                printf("Error! The stack is compliant, but learned an non-compliant bound: "
                       "Solution for x%u is %.15lf, but learned lower bound %.15lf\n",
                       var.first,
                       var.second,
                       _tableau->getLowerBound(var.first));

                throw MarabouError(MarabouError::DEBUGGING_ERROR);
            }

            if (FloatUtils::lt(_tableau->getUpperBound(var.first), var.second, 1e-5)) {
                printf("Error! The stack is compliant, but learned an non-compliant bound: "
                       "Solution for %u is %.15lf, but learned upper bound %.15lf\n",
                       var.first,
                       var.second,
                       _tableau->getUpperBound(var.first));

                throw MarabouError(MarabouError::DEBUGGING_ERROR);
            }
        }
    }
}

void Engine::quitSignal() {
    _quitRequested = true;
}

Engine::ExitCode Engine::getExitCode() const {
    return _exitCode;
}

std::atomic_bool *Engine::getQuitRequested() {
    return &_quitRequested;
}

List<unsigned> Engine::getInputVariables() const {
    return _preprocessedQuery->getInputVariables();
}

void Engine::performSimulation() {
    if (_simulationSize == 0 || !_networkLevelReasoner ||
        _milpSolverBoundTighteningType == MILPSolverBoundTighteningType::NONE) {
        ENGINE_LOG(Stringf("Skip simulation...").ascii());
        return;
    }

    // outer vector is for neuron
    // inner vector is for simulation value
    Vector<Vector<double>> simulations;

    std::mt19937 mt(GlobalConfiguration::SIMULATION_RANDOM_SEED);

    for (unsigned i = 0; i < _networkLevelReasoner->getLayer(0)->getSize(); ++i) {
        std::uniform_real_distribution<double> distribution(_networkLevelReasoner->getLayer(0)->getLb(i),
                                                            _networkLevelReasoner->getLayer(0)->getUb(i));
        Vector<double> simulationInput(_simulationSize);

        for (unsigned j = 0; j < _simulationSize; ++j)
            simulationInput[j] = distribution(mt);
        simulations.append(simulationInput);
    }
    _networkLevelReasoner->simulate(&simulations);
}

void Engine::performSymbolicBoundTightening(InputQuery *inputQuery) {
    if (_symbolicBoundTighteningType == SymbolicBoundTighteningType::NONE ||
        (!_networkLevelReasoner))
        return;

    struct timespec start = TimeUtils::sampleMicro();

    unsigned numTightenedBounds = 0;

    // Step 1: tell the NLR about the current bounds
    if (inputQuery) {
        // Obtain from and store bounds into inputquery if it is not null.
        _networkLevelReasoner->obtainCurrentBounds(*inputQuery);
    } else {
        // Get bounds from Tableau.
        _networkLevelReasoner->obtainCurrentBounds();
    }

    // Step 2: perform SBT
    if (_symbolicBoundTighteningType ==
        SymbolicBoundTighteningType::SYMBOLIC_BOUND_TIGHTENING)
        _networkLevelReasoner->symbolicBoundPropagation();
    else if (_symbolicBoundTighteningType ==
             SymbolicBoundTighteningType::DEEP_POLY)
        _networkLevelReasoner->deepPolyPropagation();

    // Step 3: Extract the bounds
    List<Tightening> tightenings;
    _networkLevelReasoner->getConstraintTightenings(tightenings);

    if (inputQuery) {
        for (const auto &tightening: tightenings) {

            if (tightening._type == Tightening::LB &&
                FloatUtils::gt(tightening._value,
                               inputQuery->getLowerBound
                                       (tightening._variable))) {
                inputQuery->setLowerBound(tightening._variable,
                                          tightening._value);
                ++numTightenedBounds;
            }

            if (tightening._type == Tightening::UB &&
                FloatUtils::lt(tightening._value,
                               inputQuery->getUpperBound
                                       (tightening._variable))) {
                inputQuery->setUpperBound(tightening._variable,
                                          tightening._value);
                ++numTightenedBounds;
            }
        }
    } else {
        for (const auto &tightening: tightenings) {

            if (tightening._type == Tightening::LB &&
                FloatUtils::gt(tightening._value, _tableau->getLowerBound(tightening._variable))) {
                _tableau->tightenLowerBound(tightening._variable, tightening._value);
                ++numTightenedBounds;
            }

            if (tightening._type == Tightening::UB &&
                FloatUtils::lt(tightening._value, _tableau->getUpperBound(tightening._variable))) {
                _tableau->tightenUpperBound(tightening._variable, tightening._value);
                ++numTightenedBounds;
            }
        }
    }

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute(Statistics::TOTAL_TIME_PERFORMING_SYMBOLIC_BOUND_TIGHTENING,
                                 TimeUtils::timePassed(start, end));
    _statistics.incLongAttribute(Statistics::NUM_TIGHTENINGS_FROM_SYMBOLIC_BOUND_TIGHTENING,
                                 numTightenedBounds);
}

bool Engine::shouldExitDueToTimeout(unsigned timeout) const {
    // A timeout value of 0 means no time limit
    if (timeout == 0)
        return false;

    return _statistics.getTotalTimeInMicro() / MICROSECONDS_TO_SECONDS > timeout;
}

void Engine::preContextPushHook() {
    struct timespec start = TimeUtils::sampleMicro();
    _boundManager.storeLocalBounds();
    struct timespec end = TimeUtils::sampleMicro();

    _statistics.incLongAttribute(Statistics::TIME_CONTEXT_PUSH_HOOK, TimeUtils::timePassed(start, end));
}

void Engine::postContextPopHook() {
    struct timespec start = TimeUtils::sampleMicro();

    _boundManager.restoreLocalBounds();
    _tableau->postContextPopHook();

    struct timespec end = TimeUtils::sampleMicro();
    _statistics.incLongAttribute(Statistics::TIME_CONTEXT_POP_HOOK, TimeUtils::timePassed(start, end));
}

void Engine::reset() {
    resetStatistics();
    clearViolatedPLConstraints();
    resetSmtCore();
    resetBoundTighteners();
    resetExitCode();
}

void Engine::resetStatistics() {
    Statistics statistics;
    _statistics = statistics;
    _smtCore.setStatistics(&_statistics);
    _tableau->setStatistics(&_statistics);
    _rowBoundTightener->setStatistics(&_statistics);
    _preprocessor.setStatistics(&_statistics);
    _activeEntryStrategy->setStatistics(&_statistics);

    _statistics.stampStartingTime();
}

void Engine::clearViolatedPLConstraints() {
    _violatedPlConstraints.clear();
    _plConstraintToFix = NULL;
}

void Engine::resetSmtCore() {
    _smtCore.reset();
    _smtCore.initializeScoreTrackerIfNeeded(_plConstraints);
}

void Engine::resetExitCode() {
    _exitCode = Engine::NOT_DONE;
}

void Engine::resetBoundTighteners() {
}

void Engine::warmStart() {
    // An NLR is required for a warm start
    if (!_networkLevelReasoner)
        return;

    // First, choose an arbitrary assignment for the input variables
    unsigned numInputVariables = _preprocessedQuery->getNumInputVariables();
    unsigned numOutputVariables = _preprocessedQuery->getNumOutputVariables();

    if (numInputVariables == 0) {
        // Trivial case: all inputs are fixed, nothing to evaluate
        return;
    }

    double *inputAssignment = new double[numInputVariables];
    double *outputAssignment = new double[numOutputVariables];

    for (unsigned i = 0; i < numInputVariables; ++i) {
        unsigned variable = _preprocessedQuery->inputVariableByIndex(i);
        inputAssignment[i] = _tableau->getLowerBound(variable);
    }

    // Evaluate the network for this assignment
    _networkLevelReasoner->evaluate(inputAssignment, outputAssignment);

    // Try to update as many variables as possible to match their assignment
    for (unsigned i = 0; i < _networkLevelReasoner->getNumberOfLayers(); ++i) {
        const NLR::Layer *layer = _networkLevelReasoner->getLayer(i);
        unsigned layerSize = layer->getSize();
        const double *assignment = layer->getAssignment();

        for (unsigned j = 0; j < layerSize; ++j) {
            if (layer->neuronHasVariable(j)) {
                unsigned variable = layer->neuronToVariable(j);
                if (!_tableau->isBasic(variable))
                    _tableau->setNonBasicAssignment(variable, assignment[j], false);
            }
        }
    }

    // We did what we could for the non-basics; now let the tableau compute
    // the basic assignment
    _tableau->computeAssignment();

    delete[] outputAssignment;
    delete[] inputAssignment;
}

void Engine::checkOverallProgress() {
    // Get fresh statistics
    unsigned numVisitedStates =
            _statistics.getUnsignedAttribute(Statistics::NUM_VISITED_TREE_STATES);
    unsigned long long currentIteration = _statistics.getLongAttribute
            (Statistics::NUM_MAIN_LOOP_ITERATIONS);

    if (numVisitedStates > _lastNumVisitedStates) {
        // Progress has been made
        _lastNumVisitedStates = numVisitedStates;
        _lastIterationWithProgress = currentIteration;
    } else {
        // No progress has been made. If it's been too long, request a restoration
        if (currentIteration >
            _lastIterationWithProgress +
            GlobalConfiguration::MAX_ITERATIONS_WITHOUT_PROGRESS) {
            ENGINE_LOG("checkOverallProgress detected cycling. Requesting a precision restoration");
            _basisRestorationRequired = Engine::STRONG_RESTORATION_NEEDED;
            _lastIterationWithProgress = currentIteration;
        }
    }
}

void Engine::updateDirections() {
    if (GlobalConfiguration::USE_POLARITY_BASED_DIRECTION_HEURISTICS)
        for (const auto &constraint: _plConstraints)
            if (constraint->supportPolarity() &&
                constraint->isActive() && !constraint->phaseFixed())
                constraint->updateDirection();
}

void Engine::decideBranchingHeuristics() {
    DivideStrategy divideStrategy = Options::get()->getDivideStrategy();
    if (divideStrategy == DivideStrategy::Auto) {
        if (_preprocessedQuery->getInputVariables().size() <
            GlobalConfiguration::INTERVAL_SPLITTING_THRESHOLD) {
            divideStrategy = DivideStrategy::LargestInterval;
            if (_verbosity >= 2)
                printf("Branching heuristics set to LargestInterval\n");
        } else {
            if (GlobalConfiguration::USE_DEEPSOI_LOCAL_SEARCH) {
                divideStrategy = DivideStrategy::PseudoImpact;
                if (_verbosity >= 2)
                    printf("Branching heuristics set to PseudoImpact\n");
            } else {
                divideStrategy = DivideStrategy::ReLUViolation;
                if (_verbosity >= 2)
                    printf("Branching heuristics set to ReLUViolation\n");
            }
        }
    }
    ASSERT(divideStrategy != DivideStrategy::Auto);
    _smtCore.setBranchingHeuristics(divideStrategy);
    _smtCore.initializeScoreTrackerIfNeeded(_plConstraints);
}

PiecewiseLinearConstraint *Engine::pickSplitPLConstraintBasedOnPolarity() {
    ENGINE_LOG(Stringf("Using Polarity-based heuristics...").ascii());

    if (!_networkLevelReasoner)
        throw MarabouError(MarabouError::NETWORK_LEVEL_REASONER_NOT_AVAILABLE);

    List<PiecewiseLinearConstraint *> constraints =
            _networkLevelReasoner->getConstraintsInTopologicalOrder();

    Map<double, PiecewiseLinearConstraint *> scoreToConstraint;
    for (auto &plConstraint: constraints) {
        if (plConstraint->supportPolarity() &&
            plConstraint->isActive() && !plConstraint->phaseFixed()) {
            plConstraint->updateScoreBasedOnPolarity();
            scoreToConstraint[plConstraint->getScore()] = plConstraint;
            if (scoreToConstraint.size() >=
                GlobalConfiguration::POLARITY_CANDIDATES_THRESHOLD)
                break;
        }
    }
    if (scoreToConstraint.size() > 0) {
        ENGINE_LOG(Stringf("Score of the picked ReLU: %f",
                           (*scoreToConstraint.begin()).first).ascii());
        return (*scoreToConstraint.begin()).second;
    } else
        return NULL;
}

PiecewiseLinearConstraint *Engine::pickSplitPLConstraintBasedOnTopology() {
    // We push the first unfixed ReLU in the topology order to the _candidatePlConstraints
    ENGINE_LOG(Stringf("Using EarliestReLU heuristics...").ascii());

    if (!_networkLevelReasoner)
        throw MarabouError(MarabouError::NETWORK_LEVEL_REASONER_NOT_AVAILABLE);

    List<PiecewiseLinearConstraint *> constraints =
            _networkLevelReasoner->getConstraintsInTopologicalOrder();

    for (auto &plConstraint: constraints) {
        if (plConstraint->isActive() && !plConstraint->phaseFixed())
            return plConstraint;
    }
    return NULL;
}

PiecewiseLinearConstraint *Engine::pickSplitPLConstraintBasedOnIntervalWidth() {
    // We push the first unfixed ReLU in the topology order to the _candidatePlConstraints
    ENGINE_LOG(Stringf("Using LargestInterval heuristics...").ascii());

    unsigned inputVariableWithLargestInterval = 0;
    double largestIntervalSoFar = 0;
    for (const auto &variable: _preprocessedQuery->getInputVariables()) {
        double interval = _tableau->getUpperBound(variable) -
                          _tableau->getLowerBound(variable);
        if (interval > largestIntervalSoFar) {
            inputVariableWithLargestInterval = variable;
            largestIntervalSoFar = interval;
        }
    }

    if (largestIntervalSoFar == 0)
        return nullptr;
    else {
        return getDisjunctionConstraintBasedOnIntervalWidth(inputVariableWithLargestInterval);
    }
}

PiecewiseLinearConstraint *Engine::pickSplitPLConstraint(DivideStrategy
                                                         strategy) {
    ENGINE_LOG(Stringf("Picking a split PLConstraint...").ascii());

    PiecewiseLinearConstraint *candidatePLConstraint = NULL;
    if (strategy == DivideStrategy::PseudoImpact) {
        if (_smtCore.getStackDepth() > 3)
            candidatePLConstraint = _smtCore.getConstraintsWithHighestScore();
        else if (_preprocessedQuery->getInputVariables().size() <
                 GlobalConfiguration::INTERVAL_SPLITTING_THRESHOLD)
            candidatePLConstraint = pickSplitPLConstraintBasedOnIntervalWidth();
        else
            candidatePLConstraint = pickSplitPLConstraintBasedOnPolarity();
    } else if (strategy == DivideStrategy::Polarity)
        candidatePLConstraint = pickSplitPLConstraintBasedOnPolarity();
    else if (strategy == DivideStrategy::EarliestReLU)
        candidatePLConstraint = pickSplitPLConstraintBasedOnTopology();
    else if (strategy == DivideStrategy::LargestInterval &&
             ((_centerStack.size() - 1) %
              GlobalConfiguration::INTERVAL_SPLITTING_FREQUENCY == 0)
            ) {
        // Conduct interval splitting periodically.
        candidatePLConstraint = pickSplitPLConstraintBasedOnIntervalWidth();
    }
    ENGINE_LOG(Stringf(( candidatePLConstraint ?
                       "Picked..." :
                       "Unable to pick using the current strategy..." )).ascii());
    return candidatePLConstraint;
}

PiecewiseLinearConstraint *Engine::pickSplitPLConstraintSnC(SnCDivideStrategy strategy) {
    PiecewiseLinearConstraint *candidatePLConstraint = NULL;
    if (strategy == SnCDivideStrategy::Polarity)
        candidatePLConstraint = pickSplitPLConstraintBasedOnPolarity();
    else if (strategy == SnCDivideStrategy::EarliestReLU)
        candidatePLConstraint = pickSplitPLConstraintBasedOnTopology();

    ENGINE_LOG(Stringf("Done updating scores...").ascii());
    ENGINE_LOG(Stringf(( candidatePLConstraint ?
                       "Picked..." :
                       "Unable to pick using the current strategy..." )).ascii());
    return candidatePLConstraint;
}

bool Engine::restoreSmtState(SmtState &smtState) {
    try {
        ASSERT(_smtCore.getStackDepth() == 0);

        // Step 1: all implied valid splits at root
        for (auto &validSplit: smtState._impliedValidSplitsAtRoot) {
            applySplit(validSplit);
            _smtCore.recordImpliedValidSplit(validSplit);
        }

        tightenBoundsOnConstraintMatrix();
        applyAllBoundTightenings();
        // For debugging purposes
        checkBoundCompliancyWithDebugSolution();
        do
            performSymbolicBoundTightening();
        while (applyAllValidConstraintCaseSplits());

        // Step 2: replay the stack
        for (auto &stackEntry: smtState._stack) {
            _smtCore.replaySmtStackEntry(stackEntry);
            // Do all the bound propagation, and set ReLU constraints to inactive at
            // least the one corresponding to the _activeSplit applied above.
            tightenBoundsOnConstraintMatrix();
            applyAllBoundTightenings();
            // For debugging purposes
            checkBoundCompliancyWithDebugSolution();
            do
                performSymbolicBoundTightening();
            while (applyAllValidConstraintCaseSplits());

        }
    }
    catch (const InfeasibleQueryException &) {
        // The current query is unsat, and we need to pop.
        // If we're at level 0, the whole query is unsat.
        printf("Restore failed!\n");
        _smtCore.recordStackInfo();
        if (!_smtCore.popSplit()) {
            if (_verbosity > 0) {
                printf("\nEngine::solve: UNSAT query\n");
                _statistics.print();
            }
            _exitCode = Engine::UNSAT;
            for (PiecewiseLinearConstraint *p: _plConstraints)
                p->setActiveConstraint(true);
            return false;
        }
    }
    return true;
}

void Engine::storeSmtState(SmtState &smtState) {
    _smtCore.storeSmtState(smtState);
}

bool Engine::solveWithMILPEncoding(unsigned timeoutInSeconds) {
    try {
        if (_lpSolverType == LPSolverType::NATIVE && _tableau->basisMatrixAvailable()) {
            explicitBasisBoundTightening();
            applyAllBoundTightenings();
            applyAllValidConstraintCaseSplits();
        }

        while (applyAllValidConstraintCaseSplits()) {
            performSymbolicBoundTightening();
        }
    }
    catch (const InfeasibleQueryException &) {
        _exitCode = Engine::UNSAT;
        return false;
    }

    ENGINE_LOG("Encoding the input query with Gurobi...\n");
    _gurobi = std::unique_ptr<GurobiWrapper>(new GurobiWrapper());
    _milpEncoder = std::unique_ptr<MILPEncoder>(new MILPEncoder(*_tableau));
    _milpEncoder->encodeInputQuery(*_gurobi, *_preprocessedQuery);
    ENGINE_LOG("Query encoded in Gurobi...\n");

    double timeoutForGurobi = (timeoutInSeconds == 0 ? FloatUtils::infinity()
                                                     : timeoutInSeconds);
    ENGINE_LOG(Stringf("Gurobi timeout set to %f\n", timeoutForGurobi).ascii())
    _gurobi->setTimeLimit(timeoutForGurobi);
    if (!_sncMode)
        _gurobi->setNumberOfThreads(Options::get()->getInt(Options::NUM_WORKERS));
    _gurobi->setVerbosity(_verbosity > 1);
    _gurobi->solve();

    if (_gurobi->haveFeasibleSolution()) {
        // Return UNKNOWN if input query has transcendental constratints.
        if (_preprocessedQuery->getTranscendentalConstraints().size() > 0) {
            // TODO: Return UNKNOW exitCode insted of throwing Error after implementing python interface to support UNKNOWN.
            throw MarabouError(MarabouError::FEATURE_NOT_YET_SUPPORTED,
                               "UNKNOWN (Marabou doesn't support UNKNOWN cases with exitCode yet.)");
            // _exitCode = IEngine::UNKNOWN;
            // return false;
        }
        _exitCode = IEngine::SAT;
        return true;
    } else if (_gurobi->infeasible())
        _exitCode = IEngine::UNSAT;
    else if (_gurobi->timeout())
        _exitCode = IEngine::TIMEOUT;
    else
        throw NLRError(NLRError::UNEXPECTED_RETURN_STATUS_FROM_GUROBI);
    return false;
}

void Engine::extractSolutionFromGurobi(InputQuery &inputQuery) {
    ASSERT(_gurobi != nullptr);
    Map<String, double> assignment;
    double costOrObjective;
    _gurobi->extractSolution(assignment, costOrObjective);

    for (unsigned i = 0; i < inputQuery.getNumberOfVariables(); ++i) {
        if (_preprocessingEnabled) {
            // Has the variable been merged into another?
            unsigned variable = i;
            while (_preprocessor.variableIsMerged(variable))
                variable = _preprocessor.getMergedIndex(variable);

            // Fixed variables are easy: return the value they've been fixed to.
            if (_preprocessor.variableIsFixed(variable)) {
                inputQuery.setSolutionValue(i, _preprocessor.getFixedValue(variable));
                inputQuery.setLowerBound(i, _preprocessor.getFixedValue(variable));
                inputQuery.setUpperBound(i, _preprocessor.getFixedValue(variable));
                continue;
            }

            // We know which variable to look for, but it may have been assigned
            // a new index, due to variable elimination
            variable = _preprocessor.getNewIndex(variable);

            // Finally, set the assigned value
            String variableName = _milpEncoder->getVariableNameFromVariable(variable);
            inputQuery.setSolutionValue(i, assignment[variableName]);
        } else {
            String variableName = _milpEncoder->getVariableNameFromVariable(i);
            inputQuery.setSolutionValue(i, assignment[variableName]);
        }
    }
}

bool Engine::preprocessingEnabled() const {
    return _preprocessingEnabled;
}

const Preprocessor *Engine::getPreprocessor() {
    return &_preprocessor;
}

bool Engine::performDeepSoILocalSearch() {
    ENGINE_LOG("Performing local search...");
    struct timespec start = TimeUtils::sampleMicro();
    ASSERT(allVarsWithinBounds());

    // All the linear constraints have been satisfied at this point.
    // Update the cost function
    _soiManager->initializePhasePattern();

    LinearExpression initialPhasePattern =
            _soiManager->getCurrentSoIPhasePattern();

//    printf("performDeepSoILocalSearch...\n");
    if (initialPhasePattern.isZero()) {
        while (!_smtCore.needToSplit())
            _smtCore.reportRejectedPhasePatternProposal();
//        printf("Found constraint to split...\n");
        return false;
    }

//    printf("Go on local search!\n");
    minimizeHeuristicCost(initialPhasePattern);
    ASSERT(allVarsWithinBounds());
    _soiManager->updateCurrentPhasePatternForSatisfiedPLConstraints();
    // Always accept the first phase pattern.
    _soiManager->acceptCurrentPhasePattern();
    double costOfLastAcceptedPhasePattern = computeHeuristicCost
            (_soiManager->getCurrentSoIPhasePattern());

    double costOfProposedPhasePattern = FloatUtils::infinity();
    bool lastProposalAccepted = true;
    while (!_smtCore.needToSplit()) {
        struct timespec end = TimeUtils::sampleMicro();
        _statistics.incLongAttribute(Statistics::TOTAL_TIME_LOCAL_SEARCH_MICRO,
                                     TimeUtils::timePassed(start, end));
        start = end;

        if (lastProposalAccepted) {
            /*
              Check whether the optimal solution to the last accepted phase
              is a real solution. We only check this when the last proposal
              was accepted, because rejected phase pattern must have resulted in
              increase in the SoI cost.

              HW: Another option is to only do this check when
              costOfLastAcceptedPhasePattern is 0, but this might be too strict.
              The overhead is low anyway.
            */
            collectViolatedPlConstraints();
            if (allPlConstraintsHold()) {
                if (_lpSolverType == LPSolverType::NATIVE &&
                    _tableau->getBasicAssignmentStatus() !=
                    ITableau::BASIC_ASSIGNMENT_JUST_COMPUTED) {
                    if (_verbosity > 0) {
                        printf("Before declaring sat, recomputing...\n");
                    }
                    // Make sure that the assignment is precise before declaring success
                    _tableau->computeAssignment();
                    // If we actually have a real satisfying assignment,
                    return false;
                } else {
                    ENGINE_LOG("Performing local search - done");
                    return true;
                }
            } else if (FloatUtils::isZero(costOfLastAcceptedPhasePattern)) {
                // Corner case: the SoI is minimal but there are still some PL
                // constraints (those not in the SoI) unsatisfied.
                // In this case, we bump up the score of PLConstraints not in
                // the SoI with the hope to branch on them early.
//                printf("bumpUpPseudoImpactOfPLConstraintsNotInSoI...\n");
                bumpUpPseudoImpactOfPLConstraintsNotInSoI();
                while (!_smtCore.needToSplit())
                    _smtCore.reportRejectedPhasePatternProposal();
                return false;
            }
        }

        // No satisfying assignment found for the last accepted phase pattern,
        // propose an update to it.
//        printf("In finding branch process...\n");
        _soiManager->proposePhasePatternUpdate();
        minimizeHeuristicCost(_soiManager->getCurrentSoIPhasePattern());
        _soiManager->updateCurrentPhasePatternForSatisfiedPLConstraints();
        costOfProposedPhasePattern = computeHeuristicCost
                (_soiManager->getCurrentSoIPhasePattern());

        // We have the "local" effect of change the cost term of some
        // PLConstraints in the phase pattern. Use this information to influence
        // the branching decision.
        updatePseudoImpactWithSoICosts(costOfLastAcceptedPhasePattern,
                                       costOfProposedPhasePattern);

        // Decide whether to accept the last proposal.
        if (_soiManager->decideToAcceptCurrentProposal
                (costOfLastAcceptedPhasePattern, costOfProposedPhasePattern)) {
            _soiManager->acceptCurrentPhasePattern();
            costOfLastAcceptedPhasePattern = costOfProposedPhasePattern;
            lastProposalAccepted = true;
        } else {
            _smtCore.reportRejectedPhasePatternProposal();
            lastProposalAccepted = false;
        }
    }

    ENGINE_LOG("Performing local search - done");
    return false;
}

void Engine::minimizeHeuristicCost(const LinearExpression &heuristicCost) {
    ENGINE_LOG("Optimizing w.r.t. the current heuristic cost...");

    if (_lpSolverType == LPSolverType::GUROBI) {
        minimizeCostWithGurobi(heuristicCost);

        ENGINE_LOG
        (Stringf("Current heuristic cost: %f",
                 _gurobi->getOptimalCostOrObjective()).ascii());
    } else {
        _tableau->toggleOptimization(true);

        _heuristicCost = heuristicCost;

        bool localOptimumReached = false;
        while (!localOptimumReached) {
            DEBUG(_tableau->verifyInvariants());

            mainLoopStatistics();
            if (_verbosity > 1 &&
                _statistics.getLongAttribute
                        (Statistics::NUM_MAIN_LOOP_ITERATIONS) %
                _statisticsPrintingFrequency == 0)
                _statistics.print();

            if (!allVarsWithinBounds())
                throw VariableOutOfBoundDuringOptimizationException();

            if (performPrecisionRestorationIfNeeded())
                continue;

            ASSERT(allVarsWithinBounds());

            localOptimumReached = performSimplexStep();
        }
        _tableau->toggleOptimization(false);
        ENGINE_LOG
        (Stringf("Current heuristic cost: %f",
                 computeHeuristicCost(heuristicCost)).ascii());
    }

    ENGINE_LOG("Optimizing w.r.t. the current heuristic cost - done\n");
}

double Engine::computeHeuristicCost(const LinearExpression &heuristicCost) {
    return (_costFunctionManager->
            computeGivenCostFunctionDirectly(heuristicCost._addends) +
            heuristicCost._constant);
}

void Engine::updatePseudoImpactWithSoICosts(double costOfLastAcceptedPhasePattern,
                                            double costOfProposedPhasePattern) {
    ASSERT(_soiManager);

    const List<PiecewiseLinearConstraint *> &constraintsUpdated =
            _soiManager->getConstraintsUpdatedInLastProposal();
    // Score is divided by the number of updated constraints in the last
    // proposal. In the Sum of Infeasibilities paper, only one constraint
    // is updated each time. But we might consider alternative proposal
    // strategy in the future.
    double score = (fabs(costOfLastAcceptedPhasePattern -
                         costOfProposedPhasePattern)
                    / constraintsUpdated.size());

    ASSERT(constraintsUpdated.size() > 0);
    // Update the Pseudo-Impact estimation.
    for (const auto &constraint: constraintsUpdated)
        _smtCore.updatePLConstraintScore(constraint, score);
}

void Engine::bumpUpPseudoImpactOfPLConstraintsNotInSoI() {
    ASSERT(_soiManager);
    for (const auto &plConstraint: _plConstraints) {
        if (plConstraint->isActive() && !plConstraint->supportSoI() &&
            !plConstraint->phaseFixed() && !plConstraint->satisfied())
            _smtCore.updatePLConstraintScore
                    (plConstraint, GlobalConfiguration::SCORE_BUMP_FOR_PL_CONSTRAINTS_NOT_IN_SOI);
    }
}

void Engine::informLPSolverOfBounds() {
    if (_lpSolverType == LPSolverType::GUROBI) {
        struct timespec start = TimeUtils::sampleMicro();
        for (unsigned i = 0; i < _preprocessedQuery->getNumberOfVariables(); ++i) {
            String variableName = _milpEncoder->getVariableNameFromVariable(i);
            _gurobi->setLowerBound(variableName, _tableau->getLowerBound(i));
            _gurobi->setUpperBound(variableName, _tableau->getUpperBound(i));
        }
        _gurobi->updateModel();
        struct timespec end = TimeUtils::sampleMicro();
        _statistics.incLongAttribute
                (Statistics::TIME_ADDING_CONSTRAINTS_TO_MILP_SOLVER_MICRO,
                 TimeUtils::timePassed(start, end));
    } else {
        // Bounds are already up-to-date in Tableau when using native Simplex.
        return;
    }
}

bool Engine::minimizeCostWithGurobi(const LinearExpression &costFunction) {
    ASSERT(_gurobi && _milpEncoder);

    struct timespec simplexStart = TimeUtils::sampleMicro();

    _milpEncoder->encodeCostFunction(*_gurobi, costFunction);
    _gurobi->setTimeLimit(FloatUtils::infinity());
    _gurobi->solve();

    struct timespec simplexEnd = TimeUtils::sampleMicro();

    _statistics.incLongAttribute(Statistics::TIME_SIMPLEX_STEPS_MICRO,
                                 TimeUtils::timePassed(simplexStart, simplexEnd));
    _statistics.incLongAttribute(Statistics::NUM_SIMPLEX_STEPS,
                                 _gurobi->getNumberOfSimplexIterations());

    if (_gurobi->infeasible())
        throw InfeasibleQueryException();
    else if (_gurobi->optimal())
        return true;
    else
        throw CommonError(CommonError::UNEXPECTED_GUROBI_STATUS,
                          Stringf("Current status: %u",
                                  _gurobi->getStatusCode()).ascii());

    return false;
}

void Engine::checkGurobiBoundConsistency() const {
    if (_gurobi && _milpEncoder) {
        for (unsigned i = 0; i < _preprocessedQuery->getNumberOfVariables(); ++i) {
            String iName = _milpEncoder->getVariableNameFromVariable(i);
            double gurobiLowerBound = _gurobi->getLowerBound(iName);
            double lowerBound = _tableau->getLowerBound(i);
            if (!FloatUtils::areEqual(gurobiLowerBound, lowerBound)) {
                throw MarabouError
                        (MarabouError::BOUNDS_NOT_UP_TO_DATE_IN_LP_SOLVER,
                         Stringf("x%u lower bound inconsistent!"
                                 " Gurobi: %f, Tableau: %f",
                                 i, gurobiLowerBound, lowerBound).ascii());
            }
            double gurobiUpperBound = _gurobi->getUpperBound(iName);
            double upperBound = _tableau->getUpperBound(i);

            if (!FloatUtils::areEqual(gurobiUpperBound, upperBound)) {
                throw MarabouError
                        (MarabouError::BOUNDS_NOT_UP_TO_DATE_IN_LP_SOLVER,
                         Stringf("x%u upper bound inconsistent!"
                                 " Gurobi: %f, Tableau: %f",
                                 i, gurobiUpperBound, upperBound).ascii());
            }
        }
    }
}

bool Engine::consistentBounds() const {
    return _boundManager.consistentBounds();
}

InputQuery Engine::buildQueryFromCurrentState() const {
    InputQuery query = *_preprocessedQuery;
    for (unsigned i = 0; i < query.getNumberOfVariables(); ++i) {
        query.setLowerBound(i, _tableau->getLowerBound(i));
        query.setUpperBound(i, _tableau->getUpperBound(i));
    }
    return query;
}

PiecewiseLinearConstraint *
Engine::getDisjunctionConstraintBasedOnIntervalWidth(unsigned inputVariableWithLargestInterval) {
    double mid = (_tableau->getLowerBound(inputVariableWithLargestInterval)
                  + _tableau->getUpperBound(inputVariableWithLargestInterval)
                 ) / 2;

    PiecewiseLinearCaseSplit s1;
    s1.storeBoundTightening(Tightening(inputVariableWithLargestInterval,
                                       mid, Tightening::UB));
    s1.setInfo(0, inputVariableWithLargestInterval, CaseSplitType::DISJUNCTION_UPPER);
    PiecewiseLinearCaseSplit s2;
    s2.storeBoundTightening(Tightening(inputVariableWithLargestInterval,
                                       mid, Tightening::LB));
    s2.setInfo(0, inputVariableWithLargestInterval, CaseSplitType::DISJUNCTION_LOWER);

    List<PiecewiseLinearCaseSplit> splits;
    splits.append(s1);
    splits.append(s2);
    _disjunctionForSplitting = std::unique_ptr<DisjunctionConstraint>
            (new DisjunctionConstraint(splits));
    _disjunctionForSplitting->setPosition(0, inputVariableWithLargestInterval);
    return _disjunctionForSplitting.get();
}

bool Engine::checkSolve(unsigned timeoutInSeconds) {
    SignalHandler::getInstance()->initialize();
    SignalHandler::getInstance()->registerClient(this);
    // Register the boundManager with all the PL constraints
    for (auto &plConstraint: _plConstraints) {
        plConstraint->registerBoundManager(&_boundManager);
        _positionToConstraint[plConstraint->getPosition()] = plConstraint;
    }

    for (auto& plConstraint : _preprocessor.getEliminatedConstraintsList()) {
        auto pos = plConstraint->getPosition();
        CaseSplitType type = CaseSplitType::UNKNOWN;
        if (plConstraint->getPhaseStatus() == RELU_PHASE_ACTIVE) {
            type = CaseSplitType::RELU_ACTIVE;
        } else if (plConstraint->getPhaseStatus() == RELU_PHASE_INACTIVE) {
            type = CaseSplitType::RELU_INACTIVE;
        } else {
            printf("Can not handle!\n");
        }
        _smtCore._searchPath.addEliminatedConstraint(pos._layer, pos._node, RELU_ACTIVE);
    }
    if (_solveWithMILP)
        return solveWithMILPEncoding(timeoutInSeconds);


    updateDirections();
    if (_lpSolverType == LPSolverType::NATIVE)
        storeInitialEngineState();
    else if (_lpSolverType == LPSolverType::GUROBI) {
        ENGINE_LOG("Encoding convex relaxation into Gurobi...");
        _milpEncoder->encodeInputQuery(*_gurobi, *_preprocessedQuery, true);
        ENGINE_LOG("Encoding convex relaxation into Gurobi - done");
    }

    mainLoopStatistics();
    if (_verbosity > 0) {
        printf("\nEngine::solve: Initial statistics\n");
        _statistics.print();
        printf("\n---\n");
    }

    applyAllValidConstraintCaseSplits();
    struct timespec mainLoopStart = TimeUtils::sampleMicro();


    auto &searchPath = getSearchPath();
    auto &preSearchPath = getPreSearchPath();
    int pathNum = 0;
    int neetExtra = 0, needLess = 0;
    int canNotJudge = 0;
    for (auto &path: preSearchPath._paths) {
        int currentPos = 0;
        unsigned int maxDepth = 0;
        bool splitJustPerformed = true;
        std::sort(path.begin(), path.end());
        while (true) {
            struct timespec mainLoopEnd = TimeUtils::sampleMicro();
            _statistics.incLongAttribute(Statistics::TIME_MAIN_LOOP_MICRO,
                                         TimeUtils::timePassed(mainLoopStart,
                                                               mainLoopEnd));
            mainLoopStart = mainLoopEnd;

            if (shouldExitDueToTimeout(timeoutInSeconds)) {
                if (_verbosity > 0) {
                    printf("\n\nEngine: quitting due to timeout...\n\n");
                    printf("Final statistics:\n");
                    _statistics.print();
                }

                _exitCode = Engine::TIMEOUT;
                _statistics.timeout();
                return false;
            }

            if (_quitRequested) {
                if (_verbosity > 0) {
                    printf("\n\nEngine: quitting due to external request...\n\n");
                    printf("Final statistics:\n");
                    _statistics.print();
                }

                _exitCode = Engine::QUIT_REQUESTED;
                return false;
            }

            try {
                DEBUG(_tableau->verifyInvariants());

                mainLoopStatistics();
                if (_verbosity > 1 &&
                    _statistics.getLongAttribute
                            (Statistics::NUM_MAIN_LOOP_ITERATIONS) %
                    _statisticsPrintingFrequency == 0)
                    _statistics.print();

                if (_lpSolverType == LPSolverType::NATIVE) {
                    checkOverallProgress();
                    // Check whether progress has been made recently

                    if (performPrecisionRestorationIfNeeded())
                        continue;

                    if (_tableau->basisMatrixAvailable()) {
                        explicitBasisBoundTightening();
                        applyAllBoundTightenings();
                        applyAllValidConstraintCaseSplits();
                    }
                }

                // If true, we just entered a new subproblem
                if (splitJustPerformed) {
                    performBoundTighteningAfterCaseSplit();
                    informLPSolverOfBounds();
                    splitJustPerformed = false;
                    if (_lpSolverType == LPSolverType::GUROBI) {
                        LinearExpression dontCare;
                        minimizeCostWithGurobi(dontCare);
                    }
                    if (currentPos < path.size()) {
                        auto& element = path[currentPos];
                        auto constraint = getConstraintByPosition(element.getPosition());
                        printf("Set constriant: "); element.getPosition().dump();
                        printf("\n");
                        _smtCore.setConstraintForSplit(constraint, element.getType());
                    }
                }

                // Perform any SmtCore-initiated case splits
                if (_smtCore.needToSplit()) {
                    if (currentPos < path.size()) {
                        _smtCore.performCheckSplit();
                    } else {
                        _smtCore.performSplit();
                        canNotJudge ++;
                    }
                    currentPos ++;
                    printf("Current path: [%d], Stack depth: %d\n", pathNum,  _smtCore.getStackDepth());
                    maxDepth = std::max(maxDepth, _smtCore.getStackDepth());
                    splitJustPerformed = true;
                    continue;
                }

                if (!_tableau->allBoundsValid()) {
                    // Some variable bounds are invalid, so the query is unsat
                    throw InfeasibleQueryException();
                }

                if (allVarsWithinBounds()) {
                    // The linear portion of the problem has been solved.
                    // Check the status of the PL constraints
                    bool solutionFound =
                            adjustAssignmentToSatisfyNonLinearConstraints();
                    if (solutionFound) {
                        struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                        _statistics.incLongAttribute
                                (Statistics::TIME_MAIN_LOOP_MICRO,
                                 TimeUtils::timePassed(mainLoopStart,
                                                       mainLoopEnd));
                        if (_verbosity > 0) {
                            printf("\nEngine::solve: sat assignment found\n");
                            _statistics.print();
                        }
                        _exitCode = Engine::SAT;
                        assert(_smtCore.getStackDepth() == 0 && "stack depth is not 0!!");
                        printf("\nPath [%d] verified sat,\n", pathNum);
                        _smtCore.recordStackInfo();
                        preSearchPath.dumpPath(pathNum++);
                        return true;
                    } else
                        continue;
                }

                // We have out-of-bounds variables.
                if (_lpSolverType == LPSolverType::NATIVE)
                    performSimplexStep();
                else {
                    ENGINE_LOG("Checking LP feasibility with Gurobi...");
                    DEBUG({ checkGurobiBoundConsistency(); });
                    ASSERT(_lpSolverType == LPSolverType::GUROBI);
                    LinearExpression dontCare;
                    minimizeCostWithGurobi(dontCare);
                }
                continue;
            }
            catch (const MalformedBasisException &) {
                _tableau->toggleOptimization(false);
                if (!handleMalformedBasisException()) {
                    ASSERT(_lpSolverType == LPSolverType::NATIVE);
                    _exitCode = Engine::ERROR;
                    exportInputQueryWithError("Cannot restore tableau");
                    struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                    _statistics.incLongAttribute
                            (Statistics::TIME_MAIN_LOOP_MICRO,
                             TimeUtils::timePassed(mainLoopStart,
                                                   mainLoopEnd));
                    return false;
                }
            }
            catch (const InfeasibleQueryException &) {
                _tableau->toggleOptimization(false);
                // The current query is unsat, and we need to pop.
                // If we're at level 0, the whole query is unsat.
                _smtCore.recordStackInfo();
                if (!_smtCore.popSplit()) {
                    printf("Verified unsat!\n");
                    struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                    _statistics.incLongAttribute
                            (Statistics::TIME_MAIN_LOOP_MICRO,
                             TimeUtils::timePassed(mainLoopStart,
                                                   mainLoopEnd));
                    if (_verbosity > 0) {
                        printf("\nEngine::solve: unsat query\n");
                        _statistics.print();
                    }
                    break;
                } else {
                    splitJustPerformed = true;
                }
            }
            catch (const VariableOutOfBoundDuringOptimizationException &) {
                _tableau->toggleOptimization(false);
                continue;
            }
            catch (MarabouError &e) {
                String message =
                        Stringf("Caught a MarabouError. Code: %u. Message: %s ",
                                e.getCode(), e.getUserMessage());
                _exitCode = Engine::ERROR;
                exportInputQueryWithError(message);
                struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                _statistics.incLongAttribute
                        (Statistics::TIME_MAIN_LOOP_MICRO,
                         TimeUtils::timePassed(mainLoopStart,
                                               mainLoopEnd));
                return false;
            }
            catch (...) {
                _exitCode = Engine::ERROR;
                exportInputQueryWithError("Unknown error");
                struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                _statistics.incLongAttribute
                        (Statistics::TIME_MAIN_LOOP_MICRO,
                         TimeUtils::timePassed(mainLoopStart,
                                               mainLoopEnd));
                return false;
            }
        }
        assert(_smtCore.getStackDepth() == 0 && "stack depth is not 0!!");
        printf("\nPath [%d] verified unsat,\n", pathNum);
        if (maxDepth > preSearchPath._paths[pathNum].size()) {
            printf("Need %zu extra verified info!\n", maxDepth - preSearchPath._paths[pathNum].size());
            neetExtra ++;
        } else if (maxDepth < preSearchPath._paths[pathNum].size()) {
            printf("Need %zu less verified info!\n", preSearchPath._paths[pathNum].size() - maxDepth);
            needLess ++;
        }
        auto element = preSearchPath._paths[pathNum][0];
        if (element.getPosition()._layer) {
            auto constraint = getConstraintByPosition(element.getPosition());
            if (constraint and !constraint->isActive()) {
                printf("Enforce active "); constraint->getPosition().dump();
                constraint->setActiveConstraint(true);
            }
        }
        preSearchPath.dumpPath(pathNum++);
    }
    printf("Presearch tree path: [%zu], current search tree path [%zu]\n[%d] path need extra info, [%d] path need less\n",
           preSearchPath._paths.size(), searchPath._paths.size(), canNotJudge, needLess);
    return false;
}

void Engine::LearnClause() {
    SignalHandler::getInstance()->initialize();
    SignalHandler::getInstance()->registerClient(this);
    // Register the boundManager with all the PL constraints
    for (auto &plConstraint: _plConstraints) {
        plConstraint->registerBoundManager(&_boundManager);
        _positionToConstraint[plConstraint->getPosition()] = plConstraint;
    }

    for (auto& plConstraint : _preprocessor.getEliminatedConstraintsList()) {
        auto pos = plConstraint->getPosition();
        CaseSplitType type = CaseSplitType::UNKNOWN;
        if (plConstraint->getPhaseStatus() == RELU_PHASE_ACTIVE) {
            type = CaseSplitType::RELU_ACTIVE;
        } else if (plConstraint->getPhaseStatus() == RELU_PHASE_INACTIVE) {
            type = CaseSplitType::RELU_INACTIVE;
        } else {
            printf("Can not handle!\n");
        }
        _smtCore._searchPath.addEliminatedConstraint(pos._layer, pos._node, RELU_ACTIVE);
    }

    updateDirections();

    _milpEncoder->encodeInputQuery(*_gurobi, *_preprocessedQuery, true);

    mainLoopStatistics();
    if (_verbosity > 0) {
        printf("\nEngine::solve: Initial statistics\n");
        _statistics.print();
        printf("\n---\n");
    }

    applyAllValidConstraintCaseSplits();
    struct timespec mainLoopStart = TimeUtils::sampleMicro();


    auto &searchPath = getSearchPath();
    auto &preSearchPath = getPreSearchPath();
    int pathNum = 0;
    int canNotJudge = 0, needLess = 0;

    for (auto &path: preSearchPath._paths) {
        unsigned int maxDepth = 0;
        bool splitJustPerformed = true;
        std::sort(path.begin(), path.end());

        for (auto& pathElement : path) {
            try {
                auto constraint = getConstraintByPosition(pathElement.getPosition());
                printf("Set constriant: "); pathElement.getPosition().dump();
                printf("\n");
                _smtCore.setConstraintForSplit(constraint, pathElement.getType());
                _smtCore.performCheckSplit();
                maxDepth = std::max(maxDepth, _smtCore.getStackDepth());
                if (pathElement.getType() == DISJUNCTION_UPPER or pathElement.getType() == DISJUNCTION_LOWER) {
                    performBoundTighteningAfterCaseSplit();
                }
                informLPSolverOfBounds();

                if (!_tableau->allBoundsValid()) {
                    // Some variable bounds are invalid, so the query is unsat
                    throw InfeasibleQueryException();
                }
                LinearExpression dontCare;
                minimizeCostWithGurobi(dontCare);
            } catch (const InfeasibleQueryException &) {
                _tableau->toggleOptimization(false);
                // The current query is unsat, and we need to pop.
                // If we're at level 0, the whole query is unsat.
                _smtCore.recordStackInfo();
                if (!_smtCore.popSplit()) {
                    printf("Verified unsat!\n");
                    struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                }
                break;
            }
        }

        printf("\nPath [%d] verified unsat,\n", pathNum);
        if (_smtCore.getStackDepth()) {
            _smtCore.popSplit();
            canNotJudge ++;
        } else if (maxDepth < preSearchPath._paths[pathNum].size()) {
            printf("Need %zu less verified info!\n", preSearchPath._paths[pathNum].size() - maxDepth);
            needLess ++;
        }
        assert(_smtCore.getStackDepth() == 0 && "stack depth is not 0!!");
        auto element = preSearchPath._paths[pathNum][0];
        if (element.getPosition()._layer) {
            auto constraint = getConstraintByPosition(element.getPosition());
            if (!constraint->isActive()) {
                printf("Enforce active "); constraint->getPosition().dump();
                constraint->setActiveConstraint(true);
            }
        }
        preSearchPath.dumpPath(pathNum++);
    }
    printf("Presearch tree path: [%zu], current search tree path [%zu]\n[%d] path need extra info, [%d] path need less\n",
           preSearchPath._paths.size(), searchPath._paths.size(), canNotJudge, needLess);
}

bool Engine::checkSolve2(unsigned int timeoutInSeconds) {
    SignalHandler::getInstance()->initialize();
    SignalHandler::getInstance()->registerClient(this);
    // Register the boundManager with all the PL constraints
    for (auto &plConstraint: _plConstraints) {
        plConstraint->registerBoundManager(&_boundManager);
        _positionToConstraint[plConstraint->getPosition()] = plConstraint;
    }

    for (auto& plConstraint : _preprocessor.getEliminatedConstraintsList()) {
        auto pos = plConstraint->getPosition();
        CaseSplitType type = CaseSplitType::UNKNOWN;
        if (plConstraint->getPhaseStatus() == RELU_PHASE_ACTIVE) {
            type = CaseSplitType::RELU_ACTIVE;
        } else if (plConstraint->getPhaseStatus() == RELU_PHASE_INACTIVE) {
            type = CaseSplitType::RELU_INACTIVE;
        } else {
            printf("Can not handle!\n");
        }
        _smtCore._searchPath.addEliminatedConstraint(pos._layer, pos._node, RELU_ACTIVE);
    }

    updateDirections();

    _milpEncoder->encodeInputQuery(*_gurobi, *_preprocessedQuery, true);

    mainLoopStatistics();
    if (_verbosity > 0) {
        printf("\nEngine::solve: Initial statistics\n");
        _statistics.print();
        printf("\n---\n");
    }

    applyAllValidConstraintCaseSplits();
    struct timespec mainLoopStart = TimeUtils::sampleMicro();


    auto &searchPath = getSearchPath();
    auto &preSearchPath = getPreSearchPath();
    int pathNum = 0;
    int canNotJudge = 0, needLess = 0;

    for (auto &path: preSearchPath._paths) {

        bool splitJustPerformed = true;
        std::sort(path.begin(), path.end());
        size_t oSize = path.size();

        std::vector<PathElement> newPath;
        for (auto& pathElement : path) {
            try {
                auto constraint = getConstraintByPosition(pathElement.getPosition());
                if (pathElement.getType() == DISJUNCTION_UPPER or pathElement.getType() == DISJUNCTION_LOWER) {
                    _smtCore.setConstraintForSplit(constraint, pathElement.getType());
                    _smtCore.performCheckSplit();
                    performBoundTighteningAfterCaseSplit();
                    informLPSolverOfBounds();
                    newPath.push_back(pathElement);
                } else {
                    break;
                }
            }
            catch (const InfeasibleQueryException &) {
                _tableau->toggleOptimization(false);
                // The current query is unsat, and we need to pop.
                // If we're at level 0, the whole query is unsat.
                _smtCore.recordStackInfo();
                if (!_smtCore.popSplit()) {
                    printf("Verified unsat!\n");
                    struct timespec mainLoopEnd = TimeUtils::sampleMicro();
                }
                break;
            }
        }
        std::vector<double> lower(_tableau->getN()), upper(_tableau->getN());
        for (unsigned int var = 0; var < _tableau->getN(); ++ var) {
            lower[var] = _tableau->getLowerBound(var);
            upper[var] = _tableau->getUpperBound(var);
        }
        _milpEncoder->storeInitialBounds(lower, upper);
        bool ok = conflictClauseLearning(path, lower, upper, newPath);
        if (_smtCore.getStackDepth()) {
            _smtCore.popSplit();
        }
        if (ok) {
            printf("Path[%d]: Conflict clause learned! From [%zu] to [%zu]\n", pathNum, path.size(), newPath.size());
            bool allDis = true;
            for (auto& e : newPath) {
                if (e.getType() != DISJUNCTION_UPPER and e.getType() != DISJUNCTION_LOWER) {
                    allDis = false;
                    break;
                }
            }
            if (!allDis)
                searchPath._paths.push_back(std::move(newPath));
        } else {
            canNotJudge += 1;
        }
        assert(_smtCore.getStackDepth() == 0 && "stack depth is not 0!!");
        auto element = preSearchPath._paths[pathNum][0];
        if (element.getPosition()._layer) {
            auto constraint = getConstraintByPosition(element.getPosition());
            if (!constraint)
                continue;
            if (!constraint->isActive()) {
                printf("Enforce active "); constraint->getPosition().dump();
                constraint->setActiveConstraint(true);
            }
        }
        pathNum ++;
//        preSearchPath.dumpPath(pathNum++);
    }

    printf("Presearch tree path: [%zu], current search tree path [%zu]\n[%d] path can not judge, [%d] path need less\n",
           preSearchPath._paths.size(), searchPath._paths.size(), canNotJudge, needLess);
    size_t oLength = 0;
    size_t nLength = 0;
    for (auto& path : searchPath._paths) {
        nLength += path.size();
        if (path.size() == 1) {
            path[0].dump();
            printf("\n");
        }
    }
    for (auto& path : preSearchPath._paths) {
        oLength += path.size();
    }
    printf("Average clause length:[%lf] -> [%lf]\n", 1.0 * oLength / preSearchPath._paths.size(), 1.0*nLength/searchPath._paths.size());

    return false;
}

bool Engine::ClauseLearning() {
    SignalHandler::getInstance()->initialize();
    SignalHandler::getInstance()->registerClient(this);
    // Register the boundManager with all the PL constraints
    for (auto &plConstraint: _plConstraints) {
        plConstraint->registerBoundManager(&_boundManager);
        _positionToConstraint[plConstraint->getPosition()] = plConstraint;
    }
    for (auto& plConstraint : _preprocessor.getEliminatedConstraintsList()) {
        auto pos = plConstraint->getPosition();
        CaseSplitType type = CaseSplitType::UNKNOWN;
        if (plConstraint->getPhaseStatus() == RELU_PHASE_ACTIVE) {
            type = CaseSplitType::RELU_ACTIVE;
        } else if (plConstraint->getPhaseStatus() == RELU_PHASE_INACTIVE) {
            type = CaseSplitType::RELU_INACTIVE;
        } else {
            printf("Can not handle!\n");
        }
        _smtCore._searchPath.addEliminatedConstraint(pos._layer, pos._node, RELU_ACTIVE);
    }
    updateDirections();
    applyAllValidConstraintCaseSplits();

    std::vector<double> lower(_tableau->getN()), upper(_tableau->getN());
    for (unsigned int var = 0; var < _tableau->getN(); ++ var) {
        lower[var] = _tableau->getLowerBound(var);
        upper[var] = _tableau->getUpperBound(var);
    }

    auto &searchPath = getSearchPath();
    auto &preSearchPath = getPreSearchPath();
    int pathNum = 0;
    int length = 0,minLength = 0, verified_num = 0, slack_verified_num = 0;
    bool verified = false;
    int oTotal = 0;

    _milpEncoder->storeInitialBounds();
    for (auto &path: preSearchPath._paths) {
        //trivial encode
        verified = verifyPath(path);
        // clause learning verify
        if (verified) {
            verified_num ++;
            printf("Old path: [%d]\n", pathNum);
//            for (auto& pathElement : path) {
//                dumpConstraintBoundInfo(pathElement);
//            }
            printf("-------------------------------------------------------\n");
            oTotal += path.size();
//            size_t minSize = minClauseLearning(path);
//            minLength += minSize;
            // slack
            std::vector<PathElement> newPath;
            bool ok = conflictClauseLearning(path, lower, upper, newPath);
            if (ok) {
                length += newPath.size();
                slack_verified_num ++;
                searchPath._paths.push_back(newPath);

                printf("New path:\n");
                for (auto& pathElement : newPath) {
                    dumpConstraintBoundInfo(pathElement);
                }
            }
            printf("\n");
        }

        pathNum ++;
    }

    for (auto& path : preSearchPath._paths) {
        oTotal += path.size();
    }
    printf("Verified path: %d [%lf]\n", verified_num, 1.0 * verified_num / preSearchPath._paths.size() * 100);
    printf("Ave length : [%lf] -> [%lf], min: [%lf]\n", 1.0 * oTotal / slack_verified_num,
           1.0 * length / slack_verified_num, 1.0 * minLength / slack_verified_num);
}

bool Engine::conflictClauseLearning(std::vector<PathElement> &origin_path, std::vector<double> lowerBounds,
                                    std::vector<double> upperBounds, std::vector<PathElement> &newPath) {// record state before
    // add implied
    std::vector<PathElement> path;
    path = origin_path;

//    for (auto& element : origin_path) {
//        PathElement new_e;
//        new_e._caseSplit = element._caseSplit;
//        path.push_back(std::move(new_e));
//        for (auto& info : element._impliedSplits) {
//            PathElement imply_e;
//            imply_e._caseSplit = info;
//            path.push_back(std::move(imply_e));
//        }
//    }

    _milpEncoder->storeInitialBounds(lowerBounds, upperBounds);
    auto gurobi = GurobiWrapper();
    _milpEncoder->encodeInitialInputQuery(gurobi, *_preprocessedQuery, true);
    Map<PiecewiseLinearConstraint*, String> slackNames;
    Map<String, CaseSplitTypeInfo> slackNameToSplitInfo;
    // encode disjunction
    if (path.size() > 10) {
        {
            for (auto& element : path) {
                if (element._caseSplit._type == CaseSplitType::DISJUNCTION_LOWER ||
                    element._caseSplit._type == CaseSplitType::DISJUNCTION_UPPER) {
                    newPath.push_back(element);
                    auto position = element._caseSplit._position;
                    auto type = element._caseSplit._type;
                    auto constraint = getDisjunctionConstraintBasedOnIntervalWidth(position._node);
                    auto splits = constraint->getCaseSplits();
                    PiecewiseLinearCaseSplit split;
                    for (auto& s : splits) {
                        if (type == s.getType()) {
                            split = s;
                            break;
                        }
                    }
                    applySplit(split);
                    performSymbolicBoundTightening();
                    for ( unsigned var = 0; var < _tableau->getN(); var++ )
                    {
                        double lb = _boundManager.getLowerBound(var);
                        double ub = _boundManager.getUpperBound(var);
                        String varName = Stringf( "x%u", var );
                        gurobi.setLowerBound(varName, lb);
                        gurobi.setUpperBound(varName, ub);
                    }

                    LinearExpression costFunction;
                    _milpEncoder->encodeCostFunction(gurobi, costFunction);
                    gurobi.updateModel();
                    gurobi.setTimeLimit(FloatUtils::infinity());
                    gurobi.solve();
                    if (gurobi.infeasible()) {
                        printf("Slack: [input enough] Length from [%zu] to [%d]\n", origin_path.size(), newPath.size());
                        return true;
                    }
                }
            }
        }
    }

    //Encode the problem
    {
        unsigned int slackNum = 0;
        for (auto &element : path) {

            PiecewiseLinearCaseSplit split;
            auto pos = element.getPosition();
            auto type = element.getType();
            auto constraint = getConstraintByPosition(pos);

            if (!constraint) {
//                printf("Position not exists!\n");
                continue;
            }
            if (constraint->getPhaseStatus() != PHASE_NOT_FIXED) {
//                printf("Already fixed!\n");
                continue;
            }
            auto splits = constraint->getCaseSplits();
            for (auto& sp : splits) {
                if (sp.getType() == type) {
                    split = sp;
                    break;
                }
            }
            if (type == RELU_ACTIVE or type == RELU_INACTIVE) {
                auto* relu = dynamic_cast<ReluConstraint *>(constraint);
                unsigned int b = _tableau->getVariableAfterMerging(relu->getB());
                unsigned int f = _tableau->getVariableAfterMerging(relu->getF());
                String bName = _milpEncoder->getVariableNameFromVariable(b);
                String fName = _milpEncoder->getVariableNameFromVariable(f);
                String slackName = Stringf("s%u", slackNum ++);
                if (type == RELU_INACTIVE) {
                    {
                        List<GurobiWrapper::Term> terms;
                        gurobi.addVariable(slackName, 0, FloatUtils::infinity());
                        terms.append(GurobiWrapper::Term(1, bName));
                        terms.append(GurobiWrapper::Term(-1, slackName));
                        gurobi.addLeqConstraint(terms, 0);
                    }
                    {
                        List<GurobiWrapper::Term> terms;
                        terms.append(GurobiWrapper::Term(1, fName));
                        terms.append(GurobiWrapper::Term(-1, slackName));
                        gurobi.addLeqConstraint(terms, 0);
                    }
                } else {
                    unsigned variable = _tableau->getVariableAfterMerging(relu->getAux());
                    slackName = _milpEncoder->getVariableNameFromVariable(variable);
                }
                slackNames[constraint] = slackName;
                slackNameToSplitInfo[slackName] = split.getInfo();
            }
        }
        gurobi.updateModel();
    }
    //encode cost function
    {
        List<GurobiWrapper::Term> terms;
        for (auto& item : slackNames) {
            terms.append(GurobiWrapper::Term(1, item.second));
        }
        gurobi.setCost(terms);
    }

    gurobi.updateModel();
    int num = 0;
    while(true) {
        gurobi.setTimeLimit(FloatUtils::infinity());
        gurobi.solve();
        if (gurobi.infeasible()) {
            printf("Slack: Length from [%zu] to [%d]\n", origin_path.size(), newPath.size());
            return true;
        } else if (gurobi.optimal()) {
            Map<String, double> assignment;
            double costOrObjective;
            gurobi.extractSolution(assignment, costOrObjective);
            double maxScore = 0;
            String maxName = "";
            for (auto& item : slackNames) {
                auto name = item.second;
                if (assignment.exists(name)) {
                    if (FloatUtils::lt(maxScore,assignment.at(name))) {
                        maxScore = assignment.at(name);
                        maxName = name;
                    }
                }
            }
            if (FloatUtils::isZero(maxScore)) {
                printf("Slack: Can not judge!\n");
                return false;
            }
            num ++;
//            printf("Slack: Set %s from %lf to 0\n", maxName.ascii(), maxScore);
            gurobi.setLowerBound(maxName, 0);
            gurobi.setUpperBound(maxName, 0);
            gurobi.updateModel();
            {
                CaseSplitTypeInfo info = slackNameToSplitInfo.at(maxName);
                PathElement element;
                element.setSplit(info);
                newPath.push_back(std::move(element));
            }
        }
    }
    return false;
}

PiecewiseLinearConstraint *Engine::getConstraintByPosition(Position position) {
    if (position._layer) {
        if (_positionToConstraint.exists(position))
            return _positionToConstraint[position];
        else {
            printf("Position not exists!\n");
            return (PiecewiseLinearConstraint*) nullptr;
        }
    } else {
        return getDisjunctionConstraintBasedOnIntervalWidth(position._node);
    }
}

bool Engine::verifyPath(std::vector<PathElement> &path) {
    bool verified = false;
    {
        _gurobi = std::unique_ptr<GurobiWrapper>(new GurobiWrapper());
        auto& gurobi = *_gurobi;
        _milpEncoder->encodeInitialInputQuery(gurobi, *_preprocessedQuery, true);
        for (auto& element: path) {
            PiecewiseLinearCaseSplit split;
            auto pos = element.getPosition();
            auto type = element.getType();
            auto constraint = getConstraintByPosition(pos);
            if (!constraint) {
                continue;
            }
            auto p = constraint->getPosition();
            if (constraint->phaseFixed())
                continue;
            auto splits = constraint->getCaseSplits();
            for (auto& sp : splits) {
                if (sp.getType() == type) {
                    split = sp;
                    break;
                }
            }
            List<Tightening> bounds = split.getBoundTightenings();
            for (auto &bound: bounds) {
                unsigned variable = _tableau->getVariableAfterMerging(bound._variable);
                String varName = Stringf( "x%u", variable );

                if (bound._type == Tightening::LB) {
//                    printf("%s\n", Stringf("x%u: lower bound set to %.3lf", variable, bound._value).ascii());
                    gurobi.setLowerBound(varName, bound._value);
                } else {
//                    printf("%s\n",Stringf("x%u: upper bound set to %.3lf", variable, bound._value).ascii());
                    gurobi.setUpperBound(varName, bound._value);
                }
            }
        }
        gurobi.updateModel();
        try {
            LinearExpression dontCare;
            minimizeCostWithGurobi(dontCare);
        } catch (InfeasibleQueryException) {
            verified = true;
        }
        if (verified) {
            printf("Trivial: Verified UnSat\n");
        } else {
            printf("Trivial: Not verified!\n");
        }
    }
    return verified;
}

size_t Engine::minClauseLearning(std::vector<PathElement> &path) {
    std::vector<std::vector<PathElement>> learnedPaths;
    for (size_t i = 1; i <= path.size(); ++ i) {
        checkClauseOfLength(path, i, learnedPaths, 0);
        if (!learnedPaths.empty()) {
            printf("minClauseLearning: From [%zu] to [%zu]\n", path.size(), i);
            int leaned_num = 0;
            for (auto& learnedPath : learnedPaths) {
                // dump learned clause
                printf("Dump learned path [%d]\n", ++ leaned_num);
                for (auto& pathElement : learnedPath) {
                    dumpConstraintBoundInfo(pathElement);
                }
                printf("\n");
            }
            return i;
        }
    }
}

bool Engine::checkClauseOfLength(std::vector<PathElement> &path, size_t length,
                                 std::vector<std::vector<PathElement>> &learnedPaths, size_t startPos) {
    static std::vector<PathElement> newPath;
    if (startPos + length > path.size()) return false;

    if (!length) {
        bool verified = verifyPath(newPath);
        if (verified) {
            learnedPaths.push_back(newPath);
        }
        return verified;
    }
    bool verified = (checkClauseOfLength(path, length, learnedPaths, startPos + 1));
    newPath.push_back(path[startPos]);
    verified |= checkClauseOfLength(path, length - 1, learnedPaths, startPos + 1);
    newPath.pop_back();
    return verified;
}

void Engine::dumpConstraintBoundInfo(PathElement &pathElement) {
    auto pos = pathElement.getPosition();
    String pos_s;
    pos.dump(pos_s);
    auto constraint = (ReluConstraint *) (getConstraintByPosition(pos));
    unsigned b = constraint->getB(), f = constraint->getF(), aux = constraint->getAux();
    auto type = CaseSplitTypeInfo::getStringCaseSplitType(pathElement.getType());
    printf("%s\n", pos_s.ascii());
    String names[3] = {"back", "forward", "aux"};
    printf("Shrink: %lf ", pathElement.getType() == RELU_ACTIVE ? _tableau->getUpperBound(b): -_tableau->getLowerBound(b));
    unsigned vars[3] = {b, f, aux};
    for (size_t j = 0; j < 3; ++j) {
        unsigned var = vars[j];
        printf("{%s %s: [%lf, %lf], %lf} ", names[j].ascii(), type.ascii(),
               _tableau->getLowerBound(var), _tableau->getUpperBound(var),
               _tableau->getUpperBound(var) - _tableau->getLowerBound(var));
    }
    printf("\n");
}

void Engine::enforcePushHook() {
    tmpState._plConstraintToState.clear();
    _boundManager.storeLocalBounds();
    for (const auto &constraint: _plConstraints) {
        tmpState._plConstraintToState[constraint] = constraint->duplicateConstraint();
    }
    for (auto& constraint : _plConstraints) {
//        String s; constraint->getPosition().dump(s);
//        printf("origin: %lld position %s\n", &constraint, s.ascii());
        if (!_initial._plConstraintToState.exists(constraint))
            throw MarabouError(MarabouError::MISSING_PL_CONSTRAINT_STATE);

        constraint->restoreState(_initial._plConstraintToState[constraint]);
    }
}

void Engine::enforcePopHook() {
    _boundManager.restoreLocalBounds();
    _tableau->postContextPopHook();
}

void Engine::setSolver(Minisat::Solver* solver_ptr) {
    _solver = solver_ptr;
}

void Engine::initEngine() {
    SignalHandler::getInstance()->initialize();
    SignalHandler::getInstance()->registerClient(this);


    // Register the boundManager with all the PL constraints
    for (auto &plConstraint: _plConstraints) {
        plConstraint->registerBoundManager(&_boundManager);
        auto position = plConstraint->getPosition();
        auto var = Minisat::mkLit(_solver->newVar());
        _positionToLit[position] = var;
        _litToPosition[var] = position;
        _litToPosition[~var] = position;
        _positionToConstraint[plConstraint->getPosition()] = plConstraint;
        if (plConstraint->isActive() and plConstraint->phaseFixed()) {
            if (plConstraint->getPhaseStatus() == PhaseStatus::RELU_PHASE_ACTIVE) {
                _solver->addClause(var);
            } else if (plConstraint->getPhaseStatus() == PhaseStatus::RELU_PHASE_INACTIVE) {
                _solver->addClause(~var);
            }
        }
    }
    printf("Init done!\n");
    for (auto& plConstraint : _preprocessor.getEliminatedConstraintsList()) {
        auto pos = plConstraint->getPosition();
        CaseSplitType type = CaseSplitType::UNKNOWN;
        if (plConstraint->getPhaseStatus() == RELU_PHASE_ACTIVE) {
            type = CaseSplitType::RELU_ACTIVE;
        } else if (plConstraint->getPhaseStatus() == RELU_PHASE_INACTIVE) {
            type = CaseSplitType::RELU_INACTIVE;
        } else {
            printf("Can not handle!\n");
        }
        _smtCore._searchPath.addEliminatedConstraint(pos._layer, pos._node, RELU_ACTIVE);
    }


    updateDirections();

    applyAllValidConstraintCaseSplits();
    storeState(_initial, TableauStateStorageLevel::STORE_ENTIRE_TABLEAU_STATE );

    initial_lower.resize(_tableau->getN());
    initial_upper.resize(_tableau->getN());
    for (size_t i = 0; i < _tableau->getN(); ++ i) {
        initial_lower[i] = _tableau->getLowerBound(i);
        initial_upper[i] = _tableau->getUpperBound(i);
    }

    ENGINE_LOG("Encoding convex relaxation into Gurobi...");
    _milpEncoder->encodeInputQuery(*_gurobi, *_preprocessedQuery, true);
    ENGINE_LOG("Encoding convex relaxation into Gurobi - done");
}

void Engine::restart() {
    printf("Need restart!\n");
}

bool Engine::gurobiSolve( unsigned timeoutInSeconds ) {
    initEngine();
    applyAllValidConstraintCaseSplits();
    while (true) {
        // Perform any SmtCore-initiated case splits
        if (_smtCore.needToSplit()) {
            _smtCore.performSplit();
            performBoundTighteningWithoutEnqueue();
            continue;
        }

        if (!checkFeasible()) {
            if (!processUnSat()) {
                return false;
            }
        }

        if (allVarsWithinBounds()) {
            // The linear portion of the problem has been solved.
            // Check the status of the PL constraints
            bool solutionFound = gurobiBranch();
            if (solutionFound) {
                _smtCore.recordStackInfo();
                _exitCode = Engine::SAT;
                return true;
            } else
                continue;
        }
        continue;
    }
    return false;
}

bool Engine::processUnSat() {
    _tableau->toggleOptimization(false);
    // The current query is unsat, and we need to pop.
    _smtCore.recordStackInfo();
    auto& searchPath = getSearchPath();
    auto& back = searchPath._paths.back();
//    printf("Unsat path!\n");
//    dumpSearchPath(back);
    std::vector<PathElement> new_path;
//    printf("search path\n");
//    searchPath.dumpPath(searchPath._paths.size() - 1);
    bool learn = false;

    if (_smtCore.getStackDepth() < 18) {
        backToOriginState();
        learn = conflictClauseLearning(back, initial_lower, initial_upper, new_path);
        backToCurrentState();
    }

    int level = _smtCore.getStackDepth() - 1;
    CaseSplitTypeInfo info;
    if (learn) {
        info._position = new_path.back().getPosition();
        info._type = _smtCore.reverseCaseSplitType(new_path.back().getType());
        level = analysisBacktrackLevelMarabou(back, new_path);
        printf("Should backtrack form level [%d] to level: [%d]\n", _smtCore.getStackDepth(), level);
        searchPath._learnt.push_back(std::move(new_path));
    }

    if (level < _smtCore.getStackDepth() - 1) {
            auto back_level = _smtCore.AtLeastBackTrackTo(level);
//        auto back_level = _smtCore.At(level, info);
        printf("Back track to level: %d\n", back_level);
        if (back_level) {
            performBoundTighteningWithoutEnqueue();
        } else {
            if (_verbosity > 0) {
                printf("\nEngine::solve: unsat query\n");
                _statistics.print();
            }
            _exitCode = Engine::UNSAT;
            return false;
        }
    } else {
        if (!_smtCore.popSplit()) {
            if (_verbosity > 0) {
                printf("\nEngine::solve: unsat query\n");
                _statistics.print();
            }
            _exitCode = Engine::UNSAT;
            return false;
        } else {
            performBoundTighteningWithoutEnqueue();
        }
    }
    return true;
}


void Engine::performBoundTightening() {
    CenterStatics time("performBoundTightening");
    performMILPSolverBoundedTighteningForSingleLayer(1);
    do {
        performSymbolicBoundTightening();
    } while (applyValidConstraintCaseSplitsWithSat());

    // Tighten bounds of an output layer with MILP solver
    if (_networkLevelReasoner)    // to avoid failing of system test.
        performMILPSolverBoundedTighteningForSingleLayer
                (_networkLevelReasoner->getLayerIndexToLayer().size() - 1);
}

bool Engine::checkFeasible() {
    CenterStatics time("checkFeasible");
    LinearExpression costFunction;

    informLPSolverOfBounds();
    _gurobi->updateModel();

    _milpEncoder->encodeCostFunction(*_gurobi, costFunction);
    _gurobi->setTimeLimit(FloatUtils::infinity());
    _gurobi->solve();

    if (_gurobi->infeasible()) {
        return false;
    }
    else if (_gurobi->optimal())
        return true;
    else
        throw CommonError(CommonError::UNEXPECTED_GUROBI_STATUS,
                          Stringf("Current status: %u",
                                  _gurobi->getStatusCode()).ascii());
}

bool Engine::gurobiBranch() {
    CenterStatics time("gurobiBranch");
    collectViolatedPlConstraints();
    // If all constraints are satisfied, we are possibly done
    if (allPlConstraintsHold()) {
        return true;
    } else {
        _smtCore.clearConstraintForSplit();
        return performDeepSoILocalSearch();
    }
}

void Engine::learnClauseByGurobi() {

}

Minisat::Lit Engine::getLitByPosition(Position &position, int split_num) {
    if (_positionToLit.exists(position)) {
        return _positionToLit.at(position);
    } else if (position._layer == 0) {
        printf("Trying to get Disjunction!\n");
        return getLitByInputPosition(position, split_num);
//        if (_inputPositionToLit[position].size() < split_num) {
//            printf("Invalid input split num! current max is [%d], request [%d]\n",
//                   _inputPositionToLit[position].size(), split_num);
//            exit(-1);
//        }
//        return _inputPositionToLit[position].at(split_num - 1);
    }
    printf("Position doesn't have target Lit!\n");
    exit(-1);
}

void Engine::encodePathToLit(std::vector<PathElement> &path, Minisat::vec<Minisat::Lit> &vec) {
    for( auto& element : path) {
        vec.push(~getLitByCaseSplitTypeInfo(element._caseSplit));
        for (auto& implied_element : element._impliedSplits) {
            vec.push(~getLitByCaseSplitTypeInfo(implied_element));
        }
    }
}

Position Engine::getPositionByLit(Minisat::Lit lit) {
    if (_litToPosition.exists(lit)) {
        return _litToPosition.at(lit);
    }
    printf("Unknown lit!\n");
    exit(-1);
}

void Engine::performPropagatedSplit(Minisat::vec<Minisat::Lit> &vec) {
    for (int i = 0; i < vec.size(); ++ i) {
        auto lit = vec[i];
        auto pos = getPositionByLit(lit);
        auto constraint = getConstraintByPosition(pos);
        CaseSplitType type = getCaseSplitTypeByLit(lit);
        assert(pos._layer != 0);
        performTargetSplit(constraint, type, 2);
    }
    performBoundTightening();
}

void Engine::performTargetSplit(PiecewiseLinearConstraint *constraint, PhaseStatus type) {
    if (constraint->isActive() && !constraint->phaseFixed()) {
        String constraintString;
        constraint->dump(constraintString);
//        ENGINE_LOG(Stringf("A constraint has become valid. Dumping constraint: %s",
//                           constraintString.ascii()).ascii());
        constraint->setActiveConstraint(false);
        PiecewiseLinearCaseSplit validSplit = constraint->getCaseSplit(type);
        _smtCore.recordImpliedValidSplit(validSplit);
        validSplit.dump();
        applySplit(validSplit);
        if (_soiManager)
            _soiManager->removeCostComponentFromHeuristicCost(constraint);
    }
}

bool Engine::applyValidConstraintCaseSplitsWithSat() {
    bool appliedSplit = false;
    for (auto &constraint: _plConstraints)
        if (applyValidConstraintCaseSplit(constraint)) {
            appliedSplit = true;
            PhaseStatus status = constraint->getPhaseStatus();
            auto pos = constraint->getPosition();
            Minisat::Lit lit = getLitByPosition(pos, 0);
            if (status == PhaseStatus::RELU_PHASE_INACTIVE) {
                lit = ~lit;
            } else if (status == PhaseStatus::RELU_PHASE_ACTIVE) {
            } else {
                printf("Unsupported!\n");
                exit(-1);
            }
            printf("applyValidConstraintCaseSplitsWithSat::position: ");
            pos.dump();
            printf(" Lit val: %d\n", Minisat::var(lit));
            _solver->encodeGurobiImply(lit);
        }
    return appliedSplit;
}

unsigned int Engine::backtrackAndPerformLearntSplit(unsigned int level, Minisat::Lit lit) {
    CaseSplitTypeInfo info;
    info._position = getPositionByLit(lit);
    info._type = getCaseSplitTypeByLit(lit);
    printf("Engine::backtrackAndPerformLearntSplit: ");
    info.dump();
    printf("\n");
    return _smtCore.backTrackToGivenLevelAndPerformSplit(level, info);
}

PhaseStatus Engine::getPhaseStatusByLit(Minisat::Lit lit) {
    PhaseStatus type;
    if (Minisat::sign(lit)) {
        type = PhaseStatus::RELU_PHASE_INACTIVE;
    } else {
        type = PhaseStatus::RELU_PHASE_ACTIVE;
    }
    return type;
}

CaseSplitType Engine::getCaseSplitTypeByLit(Minisat::Lit lit) {
    CaseSplitType type;
    if (_litToInputSplitNum.exists(lit)) {
        if (Minisat::sign(lit)) {
            type = CaseSplitType::DISJUNCTION_LOWER;
        } else {
            type = CaseSplitType::DISJUNCTION_UPPER;
        }
    } else {
        if (Minisat::sign(lit)) {
            type = CaseSplitType::RELU_INACTIVE;
        } else {
            type = CaseSplitType::RELU_ACTIVE;
        }
    }
    return type;
}

Minisat::Lit Engine::getLitByCaseSplitTypeInfo(CaseSplitTypeInfo &type_info) {
    auto pos = type_info._position;
    auto type = type_info._type;
    printf("Get split type: ");
    type_info.dump();
    printf("\n");
    auto lit = getLitByPosition(pos, type_info._splitNum);
    switch (type) {
        case CaseSplitType::RELU_ACTIVE: {
            return lit;
        }
        case CaseSplitType::RELU_INACTIVE: {
            return ~lit;
        }
        case CaseSplitType::DISJUNCTION_LOWER : {
            return getLitByInputPosition(type_info._position, type_info._splitNum);
        }
        case CaseSplitType::DISJUNCTION_UPPER : {
            return ~getLitByInputPosition(type_info._position, type_info._splitNum);
        }
        default: {
            printf("Unsupported casesplit type %s\n",
                   CaseSplitTypeInfo::getStringCaseSplitType(type).ascii());
            exit(-1);
        }
    }
}

int Engine::learnClauseAndGetBackLevel(Minisat::vec<Minisat::Lit> &vec) {
    _tableau->toggleOptimization(false);
    // The current query is unsat, and we need to pop.
    _smtCore.recordStackInfo();
    auto& searchPath = getSearchPath();
    auto& back = searchPath._paths.back();
    std::vector<PathElement> new_path;
    bool learn = false;
    int level = back.size() - 1;
    if (_smtCore.getStackDepth() < 20) {
        backToOriginState();
        learn = conflictClauseLearning(back, initial_lower, initial_upper, new_path);
        backToCurrentState();
    }
    if (learn) {
//        searchPath._learnt.push_back(std::move(new_path));
        level = analysisBacktrackLevel(back, new_path);
        std::reverse(new_path.begin(), new_path.end());
        encodePathToLit(new_path, vec);
    } else {
        new_path = back;
        std::reverse(new_path.begin(), new_path.end());
        new_path[0].dump();
        encodePathToLit(new_path, vec);
    }
    return level;
}

void Engine::performBoundTighteningWithoutEnqueue() {
    performMILPSolverBoundedTighteningForSingleLayer(1);
    do {
        performSymbolicBoundTightening();
    } while (applyAllValidConstraintCaseSplits());

    // Tighten bounds of an output layer with MILP solver
    if (_networkLevelReasoner)    // to avoid failing of system test.
        performMILPSolverBoundedTighteningForSingleLayer
                (_networkLevelReasoner->getLayerIndexToLayer().size() - 1);
    informLPSolverOfBounds();
}

unsigned int Engine::getStackDepth() {
    return _smtCore.getStackDepth();
}

unsigned int Engine::analysisBacktrackLevel(std::vector<PathElement> &path,
                                            std::vector<PathElement> &learned) {
    std::map<Position, int> cnt;
    int counter = 0;

    printf("Origin path:\n");
    dumpSearchPath(path);
    printf("Learnt path:\n");
    dumpSearchPath(learned);

    for (auto& element : learned) {
        auto pos = element.getPosition();
        cnt[pos] ++;
        counter ++;
    }

    learned.clear();
    unsigned int index = 0;
    for (size_t i = 0; i < path.size(); ++ i) {
        auto& pos = path[i]._caseSplit._position;
        if (cnt.count(pos)) {
            cnt[pos] --;
            counter --;
            if (!cnt[pos]) {
                cnt.erase(pos);
            }
            learned.push_back(path[i]);
        }
        if (counter == 1) {
            index = i + 1;
            counter --;
        }
        for (auto& element : path[i]._impliedSplits) {
            auto& pos = element._position;
            if (cnt.count(pos)) {
                cnt[pos] --;
                counter --;
                if (!cnt[pos]) {
                    cnt.erase(pos);
                }
                learned.push_back(path[i]);
            }
            if (counter == 1) {
                index = i + 1;
                counter --;
            }
        }
    }
    return index;
}

void Engine::backToCurrentState() {
    ASSERT(_smtCore.needToSplit() == false);
    enforcePopHook();
    _smtCore.popContext();
    restoreState(_tmpState);
//    printf("After learn bound:\n");
//    _boundManager.dump();
//    printf("constraint state:\n");
//    for (auto& constraint : _plConstraints) {
//        String s;
//        constraint->dump(s);
//        printf("%s", s.ascii());
//    }
}

void Engine::backToOriginState() {
    ASSERT(_smtCore.needToSplit() == false);
    storeState( _tmpState, TableauStateStorageLevel::STORE_BOUNDS_ONLY );
    _smtCore.pushContext();
    enforcePushHook();
    restoreState(_initial);
    for (int i = 0; i < initial_lower.size(); ++ i) {
        _boundManager.enforceUpperBound(i, initial_upper[i]);
        _boundManager.enforceLowerBound(i, initial_lower[i]);
    }
//    printf("Before learn bound:\n");
//    _boundManager.dump();
//    printf("constraint state:\n");
//    for (auto& constraint : _plConstraints) {
//        String s;
//        constraint->dump(s);
//        printf("%s", s.ascii());
//    }
}

void Engine::performTargetSplit(PiecewiseLinearConstraint *constraint, CaseSplitType type, int record) {
    String constraintString;
    constraint->dump(constraintString);
//    printf("Perform target split!\n");
//    printf("Is active: %d, phase fixed: %d\n", constraint->isActive(), constraint->phaseFixed());
//    printf("Constraint: \n%s\nIs active: %d, phase fixed: %d\n", constraintString.ascii(), constraint->isActive(), constraint->phaseFixed());
    if (constraint->isActive() && !constraint->phaseFixed()) {
        constraint->setActiveConstraint(false);
        auto splits = constraint->getCaseSplits();
        PiecewiseLinearCaseSplit validSplit;
        for (auto& split : splits) {
            if (split.getType() == type) {
                validSplit = split;
                break;
            }
        }
        if (record == 1) {
            _smtCore.recordImpliedValidSplit(validSplit);
        } else if (record == 2) {
            _smtCore.recordSatImpliedValidSplit(validSplit);
        }
//        printf("Apply split type: %d\n", record);
//        validSplit.dump();
//        printf("\n");
        applySplit(validSplit);
        if (_soiManager)
            _soiManager->removeCostComponentFromHeuristicCost(constraint);
    }
//    printf("After: Constraint: \n%s\nIs active: %d, phase fixed: %d\n", constraintString.ascii(), constraint->isActive(), constraint->phaseFixed());
}

Minisat::Lit Engine::getBranchLit() {
    ASSERT(_smtCore.needToSplit());
    auto constraint = _smtCore.getConstraintForSplit();
    ASSERT(constraint);
    String s; constraint->dump(s);
    if (constraint->getType() == PiecewiseLinearFunctionType::DISJUNCTION) {
        auto inputIndex = constraint->getPosition()._node;
        int num = getInputSplitNum(inputIndex) + 1;
        auto split = constraint->getCaseSplits().front();
        split.setSplitNum(num);
        printf("Get split num %d\n", num);
        return getLitByCaseSplitTypeInfo(split.getInfo());
    }
    auto split = constraint->getCaseSplits().front();
    return getLitByCaseSplitTypeInfo(split.getInfo());
}

void Engine::performSplit() {
    ASSERT(_smtCore.needToSplit());
    auto constraint = _smtCore.getConstraintForSplit();
    printf("perform constraint split: ");
    auto pos = constraint->getPosition(); pos.dump();
    auto lit = getLitByPosition(pos, 0);
    printf("Lit -%d", Minisat::var(lit));
    printf("\n");
    _smtCore.performSplit();
}

unsigned int Engine::analysisBacktrackLevelMarabou(std::vector<PathElement> &path, std::vector<PathElement> &learned) {
    std::map<int, int> level_lit_count;
    std::map<Position, int> cnt;
    int total = 0;
    for (auto& element : learned) {
        cnt[element.getPosition()] ++;
        total ++;
    }

//    printf("Origin path:\n");
//    dumpSearchPath(path);
//    printf("Learnt path:\n");
//    dumpSearchPath(learned);

    learned.clear();

    for (size_t i = 0; i < path.size(); ++ i) {
        auto& info = path[i]._caseSplit;
        if (cnt.count(info._position)) {
            cnt[info._position] --;
            total --;
            level_lit_count[i + 1] ++;

            PathElement tmp; tmp._caseSplit = info;
            learned.push_back(std::move(tmp));
        }
        for (auto& imply : path[i]._impliedSplits) {
            if (cnt.count(imply._position)) {
                cnt[info._position] --;
                total --;
                level_lit_count[i + 1] ++;

                PathElement tmp; tmp._caseSplit = imply;
                learned.push_back(std::move(tmp));
            }
            if (!total) break;
        }
        if (!total) break;
    }
    for (auto& item : level_lit_count) {
        printf("Level %d: %d\n", item.first, item.second);
    }
    auto back = level_lit_count.rbegin();
    if (back->second == 1) {
        printf("Perfect!\n");
        back ++;
        return back->first;
    } else {
        printf("GG!\n");
        for (int i = 0; i < back->second; ++ i) {learned.pop_back();}
        learned.push_back(path.back());
        return back->first - 1;
    }
    return 0;
}

void Engine::performGivenSplit(PiecewiseLinearConstraint *constraint, CaseSplitType type) {
    _smtCore.performGivenSplit(constraint, type);
}

bool Engine::gurobiCheckSolve(unsigned int timeoutInSeconds) {
    initEngine();
    bool goOn = true;
    auto pre_search_tree = getPreSearchPath();
    int success = 0, total = pre_search_tree._paths.size();
//    _smtCore.pushEmptyStack();
    for (auto& path : pre_search_tree._paths) {
//        _smtCore.emptyStackClear();
        backToInitial();
        for (auto& element : path) {
            auto position = element._caseSplit._position;
            auto type = element._caseSplit._type;
            auto constraint = getConstraintByPosition(position);
            if (!constraint->phaseFixed()) {
                auto splits = constraint->getCaseSplits();
                PiecewiseLinearCaseSplit split;
                for (auto& s : splits) {
                    if (type == s.getType()) {
                        split = s;
                        break;
                    }
                }
                constraint->setActiveConstraint(false);;
                applySplit(split);
            }
        }
        performBoundTighteningAfterCaseSplit();
        informLPSolverOfBounds();

        printf("===================\n");
        dumpSearchPath(path);
//        printf("-------------------\n");
//        _smtCore.printSimpleStackInfo();

        if (checkFeasible()) {
            printf("Can not judge!\n");
            collectViolatedPlConstraints();
            // If all constraints are satisfied, we are possibly done
            if (allPlConstraintsHold()) {
                printf("Satisfy!!!\n");
            }
            if (goOn) {
                while (true) {
                    // Perform any SmtCore-initiated case splits
                    if (_smtCore.needToSplit()) {
                        _smtCore.performSplit();
                        performBoundTighteningWithoutEnqueue();
                        continue;
                    }

                    if (!checkFeasible()) {
                        if (!processUnSat()) {
                            return false;
                        }
                    }

                    if (allVarsWithinBounds()) {
                        // The linear portion of the problem has been solved.
                        // Check the status of the PL constraints
                        bool solutionFound = gurobiBranch();
                        if (solutionFound) {
                            _smtCore.recordStackInfo();
                            _exitCode = Engine::SAT;
                            return true;
                        } else
                            continue;
                    }
                    continue;
                }
            }
        } else {
            success ++;
            printf("Success judge!\n");
        }
        std::vector<PathElement> new_path;
        backToOriginState();
        conflictClauseLearning(path, new_path);
    }
    printf("Total %d, success: %d, success rate: %f\n", total, success, 1.0 * success / total);
    return false;
}

void Engine::backToInitial() {
//    restoreState(_initial);
    for (int i = 0; i < initial_lower.size(); ++ i) {
        _boundManager.enforceUpperBound(i, initial_upper[i]);
        _boundManager.enforceLowerBound(i, initial_lower[i]);
    }
}

void Engine::performSplitByLit(Minisat::Lit lit, bool record) {
    auto info = getCaseSplitTypeInfoByLit(lit);
    auto constraint = getConstraintByPosition(info._position);

    if (constraint->getType() == DISJUNCTION) {
        auto pos = constraint->getPosition();
        printf("Input [%d], input split num: %d, lit to input split num: %d\n",
               pos._node, getInputSplitNum(pos._node), _litToInputSplitNum[lit]);
        if (getInputSplitNum(pos._node) >= _litToInputSplitNum[lit]) {
            return;
        } else {
            auto split = getCaseSplit(info);
            applySplit(split);
            if (record)
                recordSplit(split);
        }
        return;
    }

    if (!constraint->phaseFixed()) {
        auto split = getCaseSplit(info);
        if (constraint->getType() != DISJUNCTION)
            constraint->setActiveConstraint(false);
        applySplit(split);
        if (record)
            recordSplit(split);
    }
}

CaseSplitTypeInfo Engine::getCaseSplitTypeInfoByLit(Minisat::Lit lit) {
    CenterStatics time("getCaseSplitTypeInfoByLit");
    auto type = getCaseSplitTypeByLit(lit);
    auto pos = getPositionByLit(lit);
    return CaseSplitTypeInfo(pos, type);
}

PiecewiseLinearCaseSplit Engine::getCaseSplit(CaseSplitTypeInfo info) {
    auto constraint = getConstraintByPosition(info._position);
    printf("Constraint is nullptr: %d\n", constraint == nullptr);
    auto splits = constraint->getCaseSplits();
    for (auto& split : splits) {
        if (split.getType() == info._type) {
            return split;
        }
    }
    exit(-1);
}

bool Engine::gurobiCheck(Minisat::vec<Minisat::Lit> &vec, int last) {
    CenterStatics time("gurobiCheck");
//    enforcePushHook();
////    backToInitial();
//    printf("Gurobi check: ");
//    for (int i = 0; i < last; ++ i) {
//        performSplitByLit(vec[i], false);
//        printf("%s%d ", Minisat::sign(vec[i]) ? "-" : " ", Minisat::var(vec[i]));
//    }
//    printf("\n");
//    performBoundTighteningAfterCaseSplit();
//    informLPSolverOfBounds();
    bool feasible = checkFeasible();
//    _gurobi->getConflict();
//    if (!feasible) {
//        _gurobi->getIIS();
//    }
//    enforcePopHook();
    return feasible;
}

void Engine::gurobiPropagate(Minisat::vec<Minisat::Lit> &vec) {
    CenterStatics time("gurobiPropagate");
//    backToInitial();
    for (int i = 0; i < vec.size(); ++ i) {
        performSplitByLit(vec[i], true);
    }
    performBoundTightening();
}

bool Engine::conflictClauseLearning(std::vector<PathElement> &path, std::vector<PathElement> &newPath) {
    _milpEncoder->storeInitialBounds(initial_lower, initial_upper);
    auto gurobi = GurobiWrapper();
    _milpEncoder->encodeInitialInputQuery(gurobi, *_preprocessedQuery, true);
    Map<PiecewiseLinearConstraint*, String> slackNames;
    Map<String, CaseSplitTypeInfo> slackNameToSplitInfo;
    // encode disjunction
    {
        for (auto& element : path) {
            if (element._caseSplit._type == CaseSplitType::DISJUNCTION_LOWER ||
                element._caseSplit._type == CaseSplitType::DISJUNCTION_UPPER) {
                newPath.push_back(element);
                auto position = element._caseSplit._position;
                auto type = element._caseSplit._type;
                auto constraint = getDisjunctionConstraintBasedOnIntervalWidth(position._node);
                auto splits = constraint->getCaseSplits();
                PiecewiseLinearCaseSplit split;
                for (auto& s : splits) {
                    if (type == s.getType()) {
                        split = s;
                        break;
                    }
                }
                applySplit(split);
                performSymbolicBoundTightening();
                for ( unsigned var = 0; var < _tableau->getN(); var++ )
                {
                    double lb = _boundManager.getLowerBound(var);
                    double ub = _boundManager.getUpperBound(var);
                    String varName = Stringf( "x%u", var );
                    gurobi.setLowerBound(varName, lb);
                    gurobi.setUpperBound(varName, ub);
                }

                LinearExpression costFunction;
                _milpEncoder->encodeCostFunction(gurobi, costFunction);
                gurobi.updateModel();
                gurobi.setTimeLimit(FloatUtils::infinity());
                gurobi.solve();
                if (gurobi.infeasible()) {
                    printf("Slack: [input enough] Length from [%zu] to [%d]\n", path.size(), newPath.size());
                    return true;
                }
            }
        }
    }
    //Encode the problem
    {
        unsigned int slackNum = 0;
        for (auto &element : path) {
            PiecewiseLinearCaseSplit split;
            auto pos = element.getPosition();
            auto type = element.getType();
            auto constraint = getConstraintByPosition(pos);

            if (!constraint) {
                continue;
            }
            if (constraint->getPhaseStatus() != PHASE_NOT_FIXED) {
                continue;
            }
            auto splits = constraint->getCaseSplits();
            for (auto& sp : splits) {
                if (sp.getType() == type) {
                    split = sp;
                    break;
                }
            }
            if (type == RELU_ACTIVE or type == RELU_INACTIVE) {
                auto* relu = dynamic_cast<ReluConstraint *>(constraint);
                unsigned int b = _tableau->getVariableAfterMerging(relu->getB());
                unsigned int f = _tableau->getVariableAfterMerging(relu->getF());
                String bName = _milpEncoder->getVariableNameFromVariable(b);
                String fName = _milpEncoder->getVariableNameFromVariable(f);
                String slackName = Stringf("s%u", slackNum ++);
                if (type == RELU_INACTIVE) {
                    {
                        List<GurobiWrapper::Term> terms;
                        gurobi.addVariable(slackName, 0, FloatUtils::infinity());
                        terms.append(GurobiWrapper::Term(1, bName));
                        terms.append(GurobiWrapper::Term(-1, slackName));
                        gurobi.addLeqConstraint(terms, 0);
                    }
                    {
                        List<GurobiWrapper::Term> terms;
                        terms.append(GurobiWrapper::Term(1, fName));
                        terms.append(GurobiWrapper::Term(-1, slackName));
                        gurobi.addLeqConstraint(terms, 0);
                    }
                } else {
                    unsigned variable = _tableau->getVariableAfterMerging(relu->getAux());
                    slackName = _milpEncoder->getVariableNameFromVariable(variable);
                }
                slackNames[constraint] = slackName;
                slackNameToSplitInfo[slackName] = split.getInfo();
            }
        }
        gurobi.updateModel();
    }
    //encode cost function
    {
        List<GurobiWrapper::Term> terms;
        for (auto& item : slackNames) {
            terms.append(GurobiWrapper::Term(1, item.second));
        }
        gurobi.setCost(terms);
    }

    gurobi.updateModel();
    int num = 0;
    while(true) {
        gurobi.setTimeLimit(FloatUtils::infinity());
        gurobi.solve();
        if (gurobi.infeasible()) {
            printf("Slack: Length from [%zu] to [%d]\n", path.size(), newPath.size());
            return true;
        } else if (gurobi.optimal()) {
            Map<String, double> assignment;
            double costOrObjective;
            gurobi.extractSolution(assignment, costOrObjective);
            double maxScore = 0;
            String maxName = "";
            for (auto& item : slackNames) {
                auto name = item.second;
                if (assignment.exists(name)) {
                    if (FloatUtils::lt(maxScore,assignment.at(name))) {
                        maxScore = assignment.at(name);
                        maxName = name;
                    }
                }
            }
            if (FloatUtils::isZero(maxScore)) {
                printf("Slack: Can not judge!\n");
                return false;
            }
            num ++;
            gurobi.setLowerBound(maxName, 0);
            gurobi.setUpperBound(maxName, 0);
            gurobi.updateModel();
            {
                CaseSplitTypeInfo info = slackNameToSplitInfo.at(maxName);
                PathElement element;
                element.setSplit(info);
                newPath.push_back(std::move(element));
            }
        }
    }
    return false;
}

bool Engine::conflictClauseLearning(Minisat::vec<Minisat::Lit> &trail,
                                    Minisat::vec<Minisat::Lit> &learnt) {
    std::vector<PathElement> path, learnt_clause;
    for (int i = 0; i < trail.size(); ++ i) {
        PathElement pathElement;
        pathElement._caseSplit = getCaseSplitTypeInfoByLit(trail[i]);
        path.push_back(std::move(pathElement));
    }
    _total_num ++;
    enforcePushHook();
    auto learn = conflictClauseLearning(path, learnt_clause);
    enforcePopHook();
    if (learn) {
        _learnt_num ++;
        std::map<Position, int> counter;
        for (auto& info : learnt_clause) {
            counter[info.getPosition()] ++;
        }
        for (auto it = path.rbegin(); it != path.rend(); ++ it) {
            if (counter.count(it->getPosition())) {
                learnt.push(~getLitByCaseSplitTypeInfo(it->_caseSplit));
            }
        }
    }
    _smtCore._searchPath._paths.push_back(std::move(path));
    printf("Record stack: [%d]\n", _smtCore._searchPath._paths.size());
    return learn;
}

int Engine::analyzeBackTrackLevel( Minisat::vec<int>& trail_lim,
                                   Minisat::vec<Minisat::Lit>& trail,
                                   Minisat::vec<Minisat::Lit>& learnt) {
    CenterStatics time("analyzeBackTrackLevel");
    bool success = false;
    _smtCore._searchPath._paths.emplace_back();
//    success = conflictClauseLearning(trail, learnt);
    int backtrack_level = trail_lim.size() - 1;
    if (!success) {
        for (int i = trail_lim.size() - 1; i >= 0; -- i) {
            learnt.push(~trail[trail_lim[i]]);
        }
    } else {
        if (learnt.size() == 1) {
            if (trail_lim.size()) {
                backtrack_level = 0;
            } else {
                backtrack_level = -1;
            }
        } else {
            if(_solver->level(Minisat::var(learnt[0])) ==
                    _solver->level(Minisat::var(learnt[1])) ) {

                printf("Learnt clause: ");
                for (int i = 0; i < learnt.size(); ++ i) {
                    printf("%s%d ", Minisat::sign(learnt[i]) ? "-" : "", Minisat::var(learnt[i]));
                }
                printf("\n");
                _solver->addLearntClauseWithoutEnqueue(learnt);
                learnt.clear();
                for (int i = trail_lim.size() - 1; i >= 0; -- i) {
                    learnt.push(~trail[trail_lim[i]]);
                }
            } else {
                backtrack_level = _solver->level(Minisat::var(learnt[1]));
            }
        }
    }
    dumpLearntSuccessRate();
    return backtrack_level;
}

void Engine::setExitCode(Engine::ExitCode exitCode) {
    _exitCode = exitCode;
}

double Engine::dumpLearntSuccessRate() {
    printf("Total path: %d, learnt num: %d, success rate: %f\n", _total_num, _learnt_num, 1.0 * _learnt_num / _total_num);
    return 0;
}

Minisat::Lit Engine::addInputSplitLit() {
    unsigned inputVariableWithLargestInterval = 0;
    double largestIntervalSoFar = 0;
    for (const auto &variable: _preprocessedQuery->getInputVariables()) {
        double interval = _tableau->getUpperBound(variable) -
                          _tableau->getLowerBound(variable);
        if (interval > largestIntervalSoFar) {
            inputVariableWithLargestInterval = variable;
            largestIntervalSoFar = interval;
        }
    }
    Position pos(0, inputVariableWithLargestInterval);

    return Minisat::Lit();
}

void Engine::updateCenterStack() {
    CenterStatics time("updateCenterStack");
    std::vector<double> lower(_tableau->getN()),upper(_tableau->getN());
    for (size_t i = 0; i < _tableau->getN(); ++ i) {
        lower[i] = _tableau->getLowerBound(i);
        upper[i] = _tableau->getUpperBound(i);
    }
    _centerStack.back().updateBound(lower, upper);
    _centerStack.back().updateConstraintState(_plConstraints);
}

void Engine::recordSplit(PiecewiseLinearCaseSplit &split) {
    auto pos = split.getPosition();
    if (pos._layer == 0) {
        _inputSplitNum[pos._node] ++;
        split.setSplitNum(_inputSplitNum[pos._node]);
    }
    _centerStack.back().recordSplit(split);
    updateCenterStack();
}

void Engine::backTrack(int level) {
    while(getDecisionLevel() > (unsigned int)level) {
        auto& vec = _centerStack.back().returnSplits();
        for (auto& split: vec) {
            auto pos = split.getPosition();
            if (!pos._layer)
                _inputSplitNum[pos._node] --;
        }
        _centerStack.pop_back();
    }

    std::vector<double> lower, upper;
    _centerStack.back().restoreConstraintState(_plConstraints);
    _centerStack.back().restoreBounds(lower, upper);
    for (int i = 0; i < lower.size(); ++ i) {
        _boundManager.enforceUpperBound(i, upper[i]);
        _boundManager.enforceLowerBound(i, lower[i]);
    }

}

bool Engine::centerSolve(unsigned int timeoutInSeconds) {
    initEngine();
    updateCenterStack();
    while (true) {
        dumpCenterStack();
        _smtCore.printSimpleStackInfo();
        // Perform any SmtCore-initiated case splits
        if (_smtCore.needToSplit()) {
            auto constraint = _smtCore.getConstraintForSplit();
            _smtCore.setNeedToSplit(false);
            auto split = constraint->getCaseSplits().front();
            constraint->setActiveConstraint(false);
            applySplit(split);
            newDecisionLevel();
            recordSplit(split);
            performBoundTighteningWithoutEnqueue();
            continue;
        }

        if (!checkFeasible()) {
            printf("Check infeasible!\n");
            if (!centerUnSat()) {
                _exitCode = ExitCode::UNSAT;
                return false;
            }
            continue;
        }
        if (allVarsWithinBounds()) {
            // The linear portion of the problem has been solved.
            // Check the status of the PL constraints
            bool solutionFound = gurobiBranch();
            if (solutionFound) {
                recordStackInfo();
                _exitCode = Engine::SAT;
                return true;
            } else
                continue;
        }
        continue;
    }
    return false;
}

void Engine::newDecisionLevel() {
    printf("newDecisionLevel!\n");
    _centerStack.emplace_back(CenterStackEntry());
    printf("Hi!\n");
}

void Engine::dumpCenterStack() {
    printf("======dumping center stack======\n");
    for (int i = 0; i < _centerStack.size(); ++ i) {
        printf("Level: %d: ",  i);
        _centerStack[i].dump();
    }
}

bool Engine::centerUnSat() {
    _tableau->toggleOptimization(false);
    recordStackInfo();
    auto& searchPath = getSearchPath();
    auto& back = searchPath._paths.back();

    std::vector<PathElement> new_path;
    bool learn = false;
    int level = getDecisionLevel() - 1;
    CaseSplitTypeInfo info;
    {
        enforcePushHook();
//        learn = conflictClauseLearning(back, initial_lower, initial_upper, new_path);
        if (!back.empty())
            learn = binaryConflictClauseLearning(back, new_path);
        enforcePopHook();

        if (learn) {
            info._position = new_path.back().getPosition();
            info._type = _smtCore.reverseCaseSplitType(new_path.back().getType());
            level = analysisBacktrackLevelMarabou(back, new_path);
            searchPath._learnt.push_back(std::move(new_path));
        } else {
            if (back.empty())
                return false;
            info._position = back.back().getPosition();
            info._type = _smtCore.reverseCaseSplitType(back.back().getType());
        }
    }

    printf("Should backtrack form level [%d] to level: [%d]\n", getDecisionLevel(), level);

    if (level == -1) {
        return false;
    }

    backTrack(level);

    auto constraint = getConstraintByPosition(info._position);
    PiecewiseLinearCaseSplit split;
    auto splits = constraint->getCaseSplits();
    for (auto& s : splits) {
        if (s.getType() == info._type) {
            split = s;
        }
    }
    constraint->setActiveConstraint(false);
    applySplit(split);
    recordSplit(split);
    performBoundTighteningAfterCaseSplit();
    return true;
}

void Engine::recordStackInfo() {
    std::vector<PathElement> path;
    bool skip = true;
    for (auto& stack : _centerStack) {
        if (skip) {
            skip = false;
            continue;
        }
        auto vec = stack.returnSplits();
        PathElement element;
        element.setSplit(vec[0].getInfo());
        for (size_t i = 1; i < vec.size(); ++ i) {
            element.appendImpliedSplit(vec[i].getInfo());
        }
        path.push_back(std::move(element));
    }
    auto& searchPath = getSearchPath();
    searchPath.appendPath(path);
}

size_t Engine::getDecisionLevel() {
    return _centerStack.size() - 1;
}

bool Engine::binaryConflictClauseLearning(std::vector<PathElement> &path, std::vector<PathElement> &newPath) {
    printf("Path size: %d\n", path.size());
    CaseSplitTypeInfo back_split;
    if (path.back()._impliedSplits.empty()) {
        back_split = path.back()._caseSplit;
    } else {
        back_split = path.back()._impliedSplits.back();
    }

    int l = 0, r = path.size() - 1;
    PathElement pathElement;
    pathElement._caseSplit = back_split;
    while (l < r) {
        int mid = (l + r) >> 1;
        printf("l: %d r: %d mid:%d\n", l, r, mid);
        std::vector<PathElement> for_check;
        {
            for_check.insert(for_check.begin(), path.begin(), path.begin() + mid + 1);
            for_check.push_back(pathElement);
        }

        if (gurobiCheck(for_check)) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    if (l == path.size() - 1)
        return false;

    for (int i = 0; i <= l; ++ i) {
        newPath.push_back(path[i]);
    }
    newPath.push_back(pathElement);

    return true;
}

bool Engine::gurobiCheck(std::vector<PathElement> &path) {
    CenterStatics time("gurobiCheck");
    backToInitial();
    printf("gurobi check: ");
    for (auto& element : path) {
        auto split = getCaseSplitByInfo(element._caseSplit);
        applySplit(split);
        split.getInfo().dump();
        printf(" ");
        for (auto& imply : element._impliedSplits) {
            auto imply_split = getCaseSplitByInfo(imply);
            applySplit(imply_split);
            imply_split.getInfo().dump();
            printf(" ");
        }
        performBoundTighteningAfterCaseSplit();
    }
    printf("\n");
    informLPSolverOfBounds();
    bool feasible = checkFeasible();
    return feasible;
}

PiecewiseLinearCaseSplit Engine::getCaseSplitByInfo(CaseSplitTypeInfo& info) {
    auto constraint = getConstraintByPosition(info._position);
    auto splits = constraint->getCaseSplits();
    for (auto& s : splits) {
        if (s.getType() == info._type) {
            return s;
        }
    }
}

void Engine::syncStack(Minisat::vec<Minisat::Lit> &vec) {
    CenterStatics time("syncStack");
    for (int i = 0; i < vec.size(); ++ i) {
        performSplitByLit(vec[i], true);
    }
    updateCenterStack();
}

int Engine::getInputSplitNum(int inputIndex) {
    return _inputSplitNum[inputIndex];
}

Minisat::Lit Engine::getLitByInputPosition(Position &pos, int split_num) {
    printf("Split input %d, split num: %d\n", pos._node, split_num);
    if (_inputPositionToLit.count(pos)) {
        if (_inputPositionToLit[pos].size() >= split_num) {
//            printf("Total size: %d\n",_inputPositionToLit[pos].size());
            return _inputPositionToLit[pos][split_num - 1];
        }
    }
    printf("Creating new lit for the %d split of input %d\n", split_num, pos._node);
    Minisat::Lit lit = Minisat::mkLit(_solver->newVar());
    _inputPositionToLit[pos].push_back(lit);
    int num = _inputPositionToLit[pos].size();
    _litToPosition[lit] = pos;
    _litToPosition[~lit] = pos;
    _litToInputSplitNum[lit] = num;
    _litToInputSplitNum[~lit] = num;
    return lit;
}

