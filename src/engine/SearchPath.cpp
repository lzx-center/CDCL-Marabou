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
        if (element._impliedSplits.size()) {
            out += "->";
            for (auto& imply: element._impliedSplits) {
                imply.dump(out);
                out += " ";
            }
        }
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
    output += "]\n";
    output += ", \"eliminated\" : [";

    for (size_t i = 0; i < _eliminatedConstraint.size(); ++ i) {
        String element;
        _eliminatedConstraint[i].dumpJson(element);
        if (i == _eliminatedConstraint.size() - 1) {
            output += element;
        } else {
            output += element + ",";
        }
    }

    output += "]\n";
    output += "}";
}


void SearchPath::loadJson(const String &jsonPath) {
    std::ifstream ifStream(jsonPath.ascii(), std::ios_base::in);
    using namespace boost::property_tree;
    ptree root;
    read_json(ifStream, root);
    ptree data = root.get_child("data");
    for (auto asserts = data.begin(); asserts != data.end(); ++ asserts) {
        std::vector<PathElement> path;
        for (auto assert = asserts->second.begin(); assert != asserts->second.end(); ++ assert) {
            String str = assert->second.get<std::string>("split");

            int layer = 0, node = 0;
            String type;
            int num = 0;
            //parse string
            for (int i = 0; i < str.length(); ++ i) {
                if (str[i] == ')' or str[i] == '(')
                    continue;
                if (str[i] == ',') {
                    num ++; i ++;
                    continue;
                }

                if (num == 0) {
                    assert('0' < str[i] and str[i] < '9');
                    layer = layer * 10 + (str[i] - '0');
                } else if (num == 1) {
                    assert('0' < str[i] and str[i] < '9');
                    node = node * 10 + (str[i] - '0');
                } else {
                    type = str.substring(i, str.length() - i - 1);
                    break;
                }
            }

            auto caseSplitType = CaseSplitTypeInfo::getCaseSplitTypeByString(type);
            PathElement e;
            e._caseSplit.setPosition(layer, node);
            e._caseSplit.setType(caseSplitType);
            path.push_back(std::move(e));
        }
        _paths.push_back(std::move(path));
    }
}

void SearchPath::addEliminatedConstraint(int layer, int node, CaseSplitType type) {
    PathElement element;
    element._caseSplit.setPosition(layer, node);
    element._caseSplit.setType(type);
    _eliminatedConstraint.push_back(std::move(element));
}

void SearchPath::saveJson(const String &jsonPath) {
    std::ofstream os(jsonPath.ascii(), std::ios::out);

    os << "{\n\"data\":[";
    for (size_t pathNum = 0; pathNum < _paths.size(); ++ pathNum) {
        auto& path = _paths[pathNum];
        os << "[";
        for (size_t i = 0; i < path.size(); ++ i) {
            String element;
            path[i].dumpJson(element);
            if (i == path.size() - 1) {
                os << element.ascii();
            } else {
                os << element.ascii() << ",";
            }
        }
        os << "]";
        if (pathNum < _paths.size() - 1) {
            os << ",";
        }
    }
    os << "]\n";
    os << ", \"eliminated\" : [";

    for (size_t i = 0; i < _eliminatedConstraint.size(); ++ i) {
        String element;
        _eliminatedConstraint[i].dumpJson(element);
        if (i == _eliminatedConstraint.size() - 1) {
            os << element.ascii();
        } else {
            os << element.ascii() << ",";
        }
    }

    os << "]\n";
    os << "}";
    os.close();
}

void SearchPath::calc() {
    double success_rate = 1.0 * _learnt.size() / _paths.size();
    printf("Learnt success rate: %f", success_rate);
    for (auto& learnt : _learnt) {
        int use_time = 0;
        if (!learnt.size())
            continue;
        for (auto& path : _paths) {
            std::set<CaseSplitTypeInfo> st;
            for (auto& element : path) {
                st.insert(element._caseSplit);
            }

            bool contain = true;
            for (auto& element : learnt) {
                if (!st.count(element._caseSplit)) {
                    contain = false;
                }
            }
            if (contain) {
                use_time ++;
            }
        }
        printf("Learnt size %zu, can use time: %d, rate: %f\n", learnt.size(), use_time, 1.0 * use_time / _paths.size());
    }
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

void dumpSearchPath(std::vector<PathElement>& vec) {
    for (size_t i = 0; i < vec.size(); ++ i) {
        printf("Level %zu: ", i + 1);
        vec[i]._caseSplit.dump();
        if (vec[i]._impliedSplits.size()) {
            printf(" implied: ");
            for (auto& im : vec[i]._impliedSplits) {
                im.dump(); printf(" ");
            }
        }
        printf("\n");
    }
}