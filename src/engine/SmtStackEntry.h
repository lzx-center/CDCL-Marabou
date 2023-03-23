/*********************                                                        */
/*! \file SmtStackEntry.h
** \verbatim
** Top contributors (to current version):
**   Guy Katz, Haoze Wu
** This file is part of the Marabou project.
** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
** in the top-level source directory) and their institutional affiliations.
** All rights reserved. See the file COPYING in the top-level source
** directory for licensing information.\endverbatim
**
** [[ Add lengthier description here ]]

**/

#ifndef __SmtStackEntry_h__
#define __SmtStackEntry_h__

#include "EngineState.h"
#include "PiecewiseLinearCaseSplit.h"

/*
  A stack entry consists of the engine state before the split,
  the active split, the alternative splits (in case of backtrack),
  and also any implied splits that were discovered subsequently.
*/
struct SmtStackEntry
{
public:
    PiecewiseLinearCaseSplit _activeSplit;
    List<PiecewiseLinearCaseSplit> _impliedValidSplits, _satImpliedValidSplits;
    List<PiecewiseLinearCaseSplit> _alternativeSplits;
    EngineState *_engineState;

    /*
      Create a copy of the SmtStackEntry on the stack and returns a pointer to
      the copy.
      We do not copy the engineState for now, since where this method is called,
      we recreate the engineState by replaying the caseSplits.
    */
    SmtStackEntry *duplicateSmtStackEntry()
    {
        SmtStackEntry *copy = new SmtStackEntry();

        copy->_activeSplit = _activeSplit;
        copy->_impliedValidSplits = _impliedValidSplits;
        copy->_alternativeSplits = _alternativeSplits;
        copy->_engineState = NULL;

        return copy;
    }
};

class CenterStackEntry {
private:
    std::vector<double> _lower, _upper;
    Map<PiecewiseLinearConstraint *, PiecewiseLinearConstraint *> _plConstraintToState;
    std::vector<PiecewiseLinearCaseSplit> _caseSplits;
public:
    CenterStackEntry() = default;
    void updateBound(std::vector<double>& lower, std::vector<double>& upper);
    void updateConstraintState(List<PiecewiseLinearConstraint *>& list);
    void recordSplit(PiecewiseLinearCaseSplit& split);
    void restoreConstraintState(List<PiecewiseLinearConstraint *>& list);
    void restoreBounds(std::vector<double>& lower, std::vector<double>& upper);
    void dump();
    std::vector<PiecewiseLinearCaseSplit>& returnSplits();
};
#endif // __SmtStackEntry_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
