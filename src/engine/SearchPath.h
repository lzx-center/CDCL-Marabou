//
// Created by z8701 on 2022/12/19.
//

#ifndef MARABOU_SEARCHPATH_H
#define MARABOU_SEARCHPATH_H
#include "PiecewiseLinearCaseSplit.h"
#include "boost/serialization/vector.hpp"

class PathElement {
public:
    enum ElementType {
        PATH = 0,
        UNSAT,
        SAT,
        LAZY
    } _type;
    CaseSplitTypeInfo _caseSplit;
    std::vector<CaseSplitTypeInfo> _impliedSplits;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        // serialize base class information
        ar & _caseSplit & _impliedSplits & _type;
    }

    void setSplit(CaseSplitTypeInfo& info);

    void appendImpliedSplit(CaseSplitTypeInfo& info);

    void dump(String& output);

    void dump();
};

class SearchPath {
public:
    std::vector<std::vector<PathElement>> _paths;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        // serialize base class information
        ar & _paths;
    }

    void appendPath(std::vector<PathElement>& path);

    void dump(String& output);

    void dump();

    void saveToFile(const String& path) const;
    void loadFromFile(const String& path);
};


#endif //MARABOU_SEARCHPATH_H
