//
// Created by z8701 on 2022/12/19.
//

#ifndef MARABOU_SEARCHPATH_H
#define MARABOU_SEARCHPATH_H
#include "PiecewiseLinearCaseSplit.h"
#include "boost/serialization/vector.hpp"
#include "boost/property_tree/ptree.hpp"
#include "boost/property_tree/json_parser.hpp"
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

    Position getPosition() const {
        return _caseSplit._position;
    }

    CaseSplitType getType() const {
        return _caseSplit._type;
    }

    void dumpJson(String& output);

    bool operator < (const PathElement& path) const {
        return _caseSplit._position < path._caseSplit._position;
    }
};

class SearchPath {
public:
    std::vector<std::vector<PathElement>> _paths, _learnt;
    std::vector<PathElement> _eliminatedConstraint;

    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        // serialize base class information
        ar & _paths;
    }

    void appendPath(std::vector<PathElement>& path);

    void dump(String& output);

    void addEliminatedConstraint(int layer, int node, CaseSplitType type);
    void simpleDump(String& out);
    void simpleDump();

    void dump();

    void saveToFile(const String& path) const;
    void loadFromFile(const String& path);

    void dumpPath(int i);

    void dumpJson(String& output);

    void loadJson(const String& jsonPath);

    void saveJson(const String& path);

    void calc();
};


#endif //MARABOU_SEARCHPATH_H
