//
// Created by z8701 on 2022/12/19.
//

#include "SearchPath.h"

void SearchPath::appendPath(std::vector<PathElement>& path) {
    _paths.push_back(std::move(path));
}

void SearchPath::dump(String &output) {
    int pathNum = 1;
    for (auto& path : _paths) {
        output += "**************************************************\n";
        output += Stringf("Path [%d]\n", pathNum ++);
        int level = 1;
        for (auto& element: path) {
            output += Stringf("Level %d\n", level ++);
            element.dump(output);
        }
        output += "**************************************************\n";
    }
}

void SearchPath::dump() {
    String output;
    dump(output);
    printf("%s\n", output.ascii());
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
