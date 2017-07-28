/*********************                                                        */
/*! \file BoundTightener.cpp
 ** \verbatim
 ** Top contributors (to current version):
 **   Duligur Ibeling
 ** This file is part of the Marabou project.
 ** Copyright (c) 2016-2017 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **/

#include "BoundTightener.h"

Tightening::Tightening( unsigned variable, double value, BoundType type )
    : _variable( variable )
    , _value( value )
    , _type( type )
{
}

bool Tightening::tighten( ITableau &tableau ) const
{
	switch ( _type )
    {
    case Tightening::BoundType::LB:
        tableau.tightenLowerBound( _variable, _value );
        break;

    case Tightening::BoundType::UB:
        tableau.tightenUpperBound( _variable, _value );
        break;
	}

    // Guy: Lets move this logic back to the engine - i.e., let the tightener
    // tighten, and let the engine ask if the bounds are valid or not.
	return tableau.boundsValid( _variable );
}

void BoundTightener::deriveTightenings( ITableau &tableau, unsigned variable )
{
    // Extract the variable's row from the tableau
	unsigned numNonBasic = tableau.getN() - tableau.getM();
	TableauRow row( numNonBasic );
	tableau.getTableauRow( variable, &row ); // ???

    // Compute the lower and upper bounds from this row
	double tightenedLowerBound = 0.0;
	double tightenedUpperBound = 0.0;
	for ( unsigned i = 0; i < numNonBasic; ++i )
	{
		const TableauRow::Entry &entry( row._row[i] );
		unsigned var = entry._var;
		double coef = entry._coefficient;
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
	if ( FloatUtils::lt( tableau.getLowerBound( variable ), tightenedLowerBound ) )
		enqueueTightening( Tightening( variable, tightenedLowerBound, Tightening::LB ) );

    // Tighten upper bound if needed
	if ( FloatUtils::gt( tableau.getUpperBound( variable ), tightenedUpperBound ) )
		enqueueTightening( Tightening( variable, tightenedLowerBound, Tightening::UB ) );
}

void BoundTightener::enqueueTightening( const Tightening& tightening )
{
	_tighteningRequests.push( tightening );
}

bool BoundTightener::tighten( ITableau &tableau )
{
	while ( !_tighteningRequests.empty() )
    {
		const Tightening &request = _tighteningRequests.peak();
		bool valid = request.tighten( tableau );
		_tighteningRequests.pop();
		if ( !valid )
		{
			_tighteningRequests.clear();
			return false;
		}
	}
	return true;
}

//
// Local Variables:
// compile-command: "make -C .. "
// tags-file-name: "../TAGS"
// c-basic-offset: 4
// End:
//