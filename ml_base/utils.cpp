//
// Created by g8y3e on 6/14/16.
//
#include "utils.h"

#include <fstream>

namespace ml_base {
    std::list<std::string> ReadFileFromPath(const std::string file_path) {
        std::ifstream in_file(file_path.c_str());

        std::list<std::string> result_str;
        std::string line;
        while(std::getline(in_file, line)) {
            result_str.push_back(line + "\n");
        }

        return result_str;
    }
}
