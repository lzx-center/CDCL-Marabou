/*********************                                                        */
/*! \file RowBoundTightener.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Duligur Ibeling
 ** This file is part of the Marabou project.
 ** Copyright (c) 2016-2017 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **/

#include "RowBoundTightener.h"
#include "Debug.h"
#include "ReluplexError.h"
#include "Statistics.h"

RowBoundTightener::RowBoundTightener()
    : _lowerBounds( NULL )
    , _upperBounds( NULL )
    , _tightenedLower( NULL )
    , _tightenedUpper( NULL )
    , _statistics( NULL )
{
}

void RowBoundTightener::initialize( const ITableau &tableau )
{
    freeMemoryIfNeeded();

    _n = tableau.getN();

    _lowerBounds = new double[_n];
    if ( !_lowerBounds )
        throw ReluplexError( ReluplexError::ALLOCATION_FAILED, "RowBoundTightener::lowerBounds" );

    _upperBounds = new double[_n];
    if ( !_upperBounds )
        throw ReluplexError( ReluplexError::ALLOCATION_FAILED, "RowBoundTightener::upperBounds" );

    _tightenedLower = new bool[_n];
    if ( !_tightenedLower )
        throw ReluplexError( ReluplexError::ALLOCATION_FAILED, "RowBoundTightener::tightenedLower" );

    _tightenedUpper = new bool[_n];
    if ( !_tightenedUpper )
        throw ReluplexError( ReluplexError::ALLOCATION_FAILED, "RowBoundTightener::tightenedUpper" );
}

void RowBoundTightener::reset( const ITableau &tableau )
{
    std::fill( _tightenedLower, _tightenedLower + _n, false );
    std::fill( _tightenedUpper, _tightenedUpper + _n, false );

    for ( unsigned i = 0; i < _n; ++i )
    {
        _lowerBounds[i] = tableau.getLowerBound( i );
        _upperBounds[i] = tableau.getUpperBound( i );
    }
}

RowBoundTightener::~RowBoundTightener()
{
    freeMemoryIfNeeded();
}

void RowBoundTightener::freeMemoryIfNeeded()
{
    if ( _lowerBounds )
    {
        delete[] _lowerBounds;
        _lowerBounds = NULL;
    }

    if ( _upperBounds )
    {
        delete[] _upperBounds;
        _upperBounds = NULL;
    }

    if ( _tightenedLower )
    {
        delete[] _tightenedLower;
        _tightenedLower = NULL;
    }

    if ( _tightenedUpper )
    {
        delete[] _tightenedUpper;
        _tightenedUpper = NULL;
    }
}

void RowBoundTightener::examineConstraintMatrix( const ITableau &tableau, bool untilSaturation )
{
    bool newBoundsLearned;

    /*
      If working until saturation, do single passes over the matrix until no new bounds
      are learned. Otherwise, just do a single pass.
    */
    do
    {
        newBoundsLearned = onePassOverConstraintMatrix( tableau );
    }
    while ( untilSaturation && newBoundsLearned );
}

bool RowBoundTightener::onePassOverConstraintMatrix( const ITableau &tableau )
{
    bool result = false;

    unsigned n = tableau.getN();
    unsigned m = tableau.getM();

    for ( unsigned i = 0; i < m; ++i )
        for ( unsigned j = 0; j < n; ++j )
            if ( tightenOnSingleRow( tableau, i, j ) )
                result = true;

    return result;
}

bool RowBoundTightener::tightenOnSingleRow( const ITableau &tableau, unsigned row, unsigned varBeingTightened )
{
    const double *A = tableau.getA();
    const double *b = tableau.getRightHandSide();
    unsigned n = tableau.getN();
    unsigned m = tableau.getM();

    double tightenedCoefficient = A[varBeingTightened * m + row];
    if ( FloatUtils::isZero( tightenedCoefficient ) )
        return false;

    bool foundNewBound = false;

    // Initialize both bounds using the right hand side of the row
    double tightenedLowerBound = b[row] / tightenedCoefficient;
	double tightenedUpperBound = tightenedLowerBound;

	for ( unsigned i = 0; i < n; ++i )
	{
        if ( i == varBeingTightened )
            continue;

		double coefficient = -A[i * m + row] / tightenedCoefficient;

		if ( FloatUtils::isPositive( coefficient ) )
		{
			tightenedLowerBound += coefficient * _lowerBounds[i];
			tightenedUpperBound += coefficient * _upperBounds[i];
		}
		else if ( FloatUtils::isNegative( coefficient ) )
		{
			tightenedLowerBound += coefficient * _upperBounds[i];
			tightenedUpperBound += coefficient * _lowerBounds[i];
		}
    }

    // Tighten lower bound if needed
	if ( FloatUtils::lt( _lowerBounds[varBeingTightened], tightenedLowerBound ) )
    {
        _lowerBounds[varBeingTightened] = tightenedLowerBound;
        _tightenedLower[varBeingTightened] = true;
        foundNewBound = true;
    }

    // Tighten upper bound if needed
	if ( FloatUtils::gt( _upperBounds[varBeingTightened], tightenedUpperBound ) )
    {
        _upperBounds[varBeingTightened] = tightenedUpperBound;
        _tightenedUpper[varBeingTightened] = true;
        foundNewBound = true;
    }

    return foundNewBound;
}

void RowBoundTightener::examinePivotRow( ITableau &tableau )
{
	if ( _statistics )
        _statistics->incNumRowsExaminedByRowTightener();

	// The entering/leaving assignments are reversed because these are called post-pivot.
	unsigned enteringVariable = tableau.getLeavingVariable();
	unsigned leavingVariable = tableau.getEnteringVariable();

	const TableauRow &row = *tableau.getPivotRow();

    unsigned enteringIndex = tableau.getEnteringVariableIndex();
	double enteringCoef = row[enteringIndex];

	/*
      The pre-pivot row says:

         leaving = enteringCoef * entering + sum ci xi + b

      where sum runs over nonbasic vars (that are not entering).
      Rearrange to

         entering = leaving / enteringCoef - sum ci/enteringCoef xi - b / enteringCoef
    */

	// Get right hand side
    double constCoef = -row._scalar / enteringCoef;

    // Compute the lower and upper bounds from this row
	double tightenedLowerBound = constCoef;
	double tightenedUpperBound = constCoef;

    unsigned numNonBasic = tableau.getN() - tableau.getM();
	for ( unsigned i = 0; i < numNonBasic; ++i )
	{
		const TableauRow::Entry &entry( row._row[i] );
		unsigned var = entry._var;
		double coef = -entry._coefficient / enteringCoef;
		// Reuse the pass of this loop on the entering index
		// to account for the leaving / enteringCoef term above.
		if ( i == enteringIndex )
		{
			var = leavingVariable;
			coef = 1.0 / enteringCoef;
		}

		double currentLowerBound = tableau.getLowerBound( var );
		double currentUpperBound = tableau.getUpperBound( var );

		if ( FloatUtils::isPositive( coef ) )
		{
			tightenedLowerBound += coef * currentLowerBound;
			tightenedUpperBound += coef * currentUpperBound;
		}
		else if ( FloatUtils::isNegative( coef ) )
		{
			tightenedLowerBound += coef * currentUpperBound;
			tightenedUpperBound += coef * currentLowerBound;
		}
	}

    // Tighten lower bound if needed
	if ( FloatUtils::lt( _lowerBounds[enteringVariable], tightenedLowerBound ) )
    {
        _lowerBounds[enteringVariable] = tightenedLowerBound;
        _tightenedLower[enteringVariable] = true;
    }

    // Tighten upper bound if needed
	if ( FloatUtils::gt( _upperBounds[enteringVariable], tightenedUpperBound ) )
    {
        _upperBounds[enteringVariable] = tightenedUpperBound;
        _tightenedUpper[enteringVariable] = true;
    }
}

void RowBoundTightener::getRowTightenings( List<Tightening> &tightenings ) const
{
    for ( unsigned i = 0; i < _n; ++i )
    {
        if ( _tightenedLower[i] )
        {
            tightenings.append( Tightening( i, _lowerBounds[i], Tightening::LB ) );
            _tightenedLower[i] = false;
        }

        if ( _tightenedUpper[i] )
        {
            tightenings.append( Tightening( i, _upperBounds[i], Tightening::UB ) );
            _tightenedUpper[i] = false;
        }
    }
}

void RowBoundTightener::setStatistics( Statistics *statistics )
{
    _statistics = statistics;
}

void RowBoundTightener::notifyLowerBound( unsigned variable, double bound )
{
    if ( FloatUtils::gt( bound, _lowerBounds[variable] ) )
    {
        _lowerBounds[variable] = bound;
        _tightenedLower[variable] = false;
    }
}

void RowBoundTightener::notifyUpperBound( unsigned variable, double bound )
{
    if ( FloatUtils::lt( bound, _upperBounds[variable] ) )
    {
        _upperBounds[variable] = bound;
        _tightenedUpper[variable] = false;
    }
}

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//