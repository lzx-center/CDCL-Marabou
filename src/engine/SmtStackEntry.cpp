//
// Created by z8701 on 2023/3/22.
//
#include "SmtStackEntry.h"

void CenterStackEntry::updateBound(std::vector<double>& lower, std::vector<double>& upper) {
    _lower = lower;
    _upper = upper;
}

void CenterStackEntry::updateConstraintState(List<PiecewiseLinearConstraint *> &list) {
    for (auto& constraint : list) {
        _plConstraintToState[constraint] = constraint->duplicateConstraint();
    }
}

void CenterStackEntry::recordSplit(PiecewiseLinearCaseSplit& split) {
    _caseSplits.push_back(split);
}

void CenterStackEntry::restoreConstraintState(List<PiecewiseLinearConstraint *>& list) {
    for (auto &constraint: list) {
        if (!_plConstraintToState.exists(constraint))
            throw MarabouError(MarabouError::MISSING_PL_CONSTRAINT_STATE);
        constraint->restoreState(_plConstraintToState[constraint]);
    }
}

void CenterStackEntry::restoreBounds(std::vector<double> &lower, std::vector<double> &upper) {
    lower = _lower;
    upper = _upper;
}

void CenterStackEntry::dump() {
    if (!_caseSplits.empty()) {
        _caseSplits[0].getInfo().dump();
        if (_caseSplits.size() > 1) {
            printf(" Implied: ");
            for (size_t i = 1; i < _caseSplits.size(); ++ i) {
                _caseSplits[i].getInfo().dump();
                printf(" ");
            }
        }
    }
    printf("\n");
}

std::vector<PiecewiseLinearCaseSplit> CenterStackEntry::returnSplits() {
    return _caseSplits;
}
