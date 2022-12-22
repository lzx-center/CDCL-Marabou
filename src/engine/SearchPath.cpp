//
// Created by z8701 on 2022/12/19.
//

#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "SearchPath.h"

void SearchPath::appendPath(std::vector<PathElement>& path) {
    _paths.push_back(std::move(path));
}

void SearchPath::dump(String &output) {
    int pathNum = 1;
    for (auto& path : _paths) {
        output += "***************************Start**************************\n";
        output += Stringf("Path [%d], total depth: [%d]\n", pathNum ++, path.size());
        int level = 1;
        for (auto& element: path) {
            output += Stringf("Level %d\n", level ++);
            element.dump(output);
        }
        output += "****************************End***************************\n";
    }
}

void SearchPath::dump() {
    String output;
    dump(output);
    printf("%s\n", output.ascii());
}

void SearchPath::saveToFile(const String& filePath) const {
    if (filePath.length()) {
        std::ofstream ofs(filePath.ascii());
        {
            boost::archive::text_oarchive oa(ofs);
            oa << *this;
        }
    } else {
        printf("Empty file path!\n");
    }
}

void SearchPath::loadFromFile(const String &filePath)  {
    std::ifstream ifs(filePath.ascii());
    {
        boost::archive::text_iarchive ia(ifs);
        // read class state from archive
        ia >> (*this);
        // archive and stream closed when destructors are called
    }
}

void SearchPath::simpleDump(String &out) {
    int pathNum = 1;
    for (auto& path : _paths) {
        out += Stringf("Path [%d], total depth: [%d]\n", pathNum ++, path.size());
        for (auto& element: path) {
            element._caseSplit.dump(out);
            out += "\n";
        }
        out += "\n";
    }
}

void SearchPath::simpleDump() {
    String output;
    simpleDump(output);
    printf("%s\n", output.ascii());
}

void SearchPath::dumpPath(int i) {
    String out;
    for (auto& element: _paths[i]) {
        element._caseSplit.dump(out);
        out += "\n";
    }
    printf("%s\n", out.ascii());
}

void SearchPath::dumpJson(String &output) {
    output += "{\n\"data\":[";
    for (size_t pathNum = 0; pathNum < _paths.size(); ++ pathNum) {
        auto& path = _paths[pathNum];
        output += "[";
        for (size_t i = 0; i < path.size(); ++ i) {
            String element;
            path[i].dumpJson(element);
            if (i == path.size() - 1) {
                output += element;
            } else {
                output += element + ",";
            }
        }
        output += "]";
        if (pathNum < _paths.size() - 1) {
            output += ",";
        }
    }
    output += "]\n}";
}

void PathElement::setSplit(CaseSplitTypeInfo &info) {
    _caseSplit = info;
}

void PathElement::appendImpliedSplit(CaseSplitTypeInfo &info) {
    _impliedSplits.push_back(info);
}

void PathElement::dump(String& output) {
    output += Stringf("==================================================\n");
    output += Stringf("Current split:\n");
    _caseSplit.dump(output);
    output += Stringf("\n");
    int impliedNum = 1;
    if (!_impliedSplits.empty()) {
        output += Stringf("------------------------\n");
        output += Stringf("Implied splits:\n");
        for (auto& info : _impliedSplits) {
            output += Stringf("Implied path [%d]\n", impliedNum ++);
            info.dump(output);
            output += Stringf("\n");
        }
        output += Stringf("------------------------\n");
    }
    output += Stringf("\n==================================================\n");
}

void PathElement::dump() {
    String s;
    dump(s);
    printf("%s", s.ascii());
}

void PathElement::dumpJson(String &output) {
    output = "{";
    String type;
    _caseSplit.dump(type);
    output += Stringf(R"("split": "%s",)", type.ascii());
    output += "\"implied\":[";
    for (size_t i = 0; i < _impliedSplits.size(); ++ i) {
        String out;
        _impliedSplits[i].dump(out);
        if (i == _impliedSplits.size() - 1) {
            output += Stringf("\"%s\"", out.ascii());
        } else {
            output += Stringf("\"%s\",", out.ascii());
        }
    }
    output += "]";
    output += "}";
}
