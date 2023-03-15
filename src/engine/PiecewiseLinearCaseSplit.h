/*********************                                                        */
/*! \file PiecewiseLinearCaseSplit.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz, Duligur Ibeling
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

**/
#pragma once
#ifndef __PiecewiseLinearCaseSplit_h__
#define __PiecewiseLinearCaseSplit_h__

#include "Equation.h"
#include "IEngine.h"
#include "MString.h"
#include "Pair.h"
#include "Tightening.h"
#include "PiecewiseLinearFunctionType.h"

enum CaseSplitType {
    RELU_INACTIVE = 0,
    RELU_ACTIVE,
    DISJUNCTION_LOWER,
    DISJUNCTION_UPPER,
    UNKNOWN
};

struct CaseSplitTypeInfo {
    Position _position;
    CaseSplitType _type;
    explicit CaseSplitTypeInfo(int layer=-1, int node=-1, CaseSplitType type=UNKNOWN) : _position(layer, node), _type(type) {}
    explicit CaseSplitTypeInfo(Position position, CaseSplitType type) {
        _position = position;
        _type = type;
    }

    void inline setPosition(int layer, int node) {
        _position = Position(layer, node);
    }

    void inline setPosition(Position& position) {
        _position = position;
    }

    void inline setType(CaseSplitType type) {
        _type = type;
    }

    void setInfo(int layer, int node, CaseSplitType type) {
        setPosition(layer, node);
        setType(type);
    }

    void setInfo(Position& position, CaseSplitType type) {
        setPosition(position);
        setType(type);
    }

    void dump(String& s) const {
        s += Stringf("(%d, %d, %s)", _position._layer, _position._node, getStringCaseSplitType(_type).ascii());
    }

    void dump() const {
        String out;
        dump(out);
        printf(" %s", out.ascii());
    }

    static Stringf getStringCaseSplitType(CaseSplitType type) {
        switch (type) {
            case RELU_ACTIVE:
                return {"Relu active"};
            case RELU_INACTIVE:
                return {"Relu inactive"};
            case DISJUNCTION_LOWER:
                return {"Disjunction lower"};
            case DISJUNCTION_UPPER:
                return {"Disjunction upper"};
            case UNKNOWN:
                return {"Unknown"};
        }
    }

    static CaseSplitType getCaseSplitTypeByString(String& s);

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & _position & _type;
    }

    bool operator < (const CaseSplitTypeInfo& split) const {
        if (_position == split._position) {
            return _type < split._type;
        }
        return _position < split._position;
    }
};

class PiecewiseLinearCaseSplit
{
public:
    /*
      Store information regarding a bound tightening.
    */
    void storeBoundTightening( const Tightening &tightening );
    const List<Tightening> &getBoundTightenings() const;

    /*
      Store information regarding a new equation to be added.
    */
    void addEquation( const Equation &equation );
  	const List<Equation> &getEquations() const;

    /*
      Dump the case split - for debugging purposes.
    */
    void dump() const;
    void dump( String &output ) const;

    /*
      Equality operator.
    */
    bool operator==( const PiecewiseLinearCaseSplit &other ) const;

    /*
      Change the index of a variable that appears in this case split
    */
    void updateVariableIndex( unsigned oldIndex, unsigned newIndex );

    void setInfo(int layer, int node, CaseSplitType type) {
        _info = CaseSplitTypeInfo(layer, node, type);
    }

    void setInfo(Position position, CaseSplitType type) {
        _info = CaseSplitTypeInfo(position, type);
    }

    CaseSplitTypeInfo& getInfo() {
        return _info;
    }

    CaseSplitType getType() {
        return _info._type;
    }

    Position getPosition() {
        return _info._position;
    }

private:
    /*
      Bound tightening information.
    */
    List<Tightening> _bounds;

    /*
      The equation that needs to be added.
    */
    List<Equation> _equations;

    CaseSplitTypeInfo _info;
};

#endif // __PiecewiseLinearCaseSplit_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
