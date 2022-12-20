/*********************                                                        */
/*! \file Engine.h
 ** \verbatim
 ** Top contributors (to current version):
 **   Guy Katz
 ** This file is part of the Marabou project.
 ** Copyright (c) 2017-2019 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 ** [[ Add lengthier description here ]]

 **/
#include "MStringf.h"
#include <boost/serialization/access.hpp>
#ifndef __PiecewiseLinearFunctionType_h__
#define __PiecewiseLinearFunctionType_h__

enum PiecewiseLinearFunctionType {
    RELU = 0,
    ABSOLUTE_VALUE = 1,
    MAX = 2,
    DISJUNCTION = 3,
    SIGN = 4,
};

struct Position {
    int _layer, _node;
    explicit Position(int layer=-1, int node=-1) : _layer(layer), _node(node) {}
    void dump(String& s) const {
        s += Stringf("Position: (%d, %d)", _layer, _node);
    }

    void dump() const {
        printf("Position: (%d, %d)", _layer, _node);
    }

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
            ar & _layer & _node;
    }
};
#endif // __PiecewiseLinearFunctionType_h__

//
// Local Variables:
// compile-command: "make -C ../.. "
// tags-file-name: "../../TAGS"
// c-basic-offset: 4
// End:
//
