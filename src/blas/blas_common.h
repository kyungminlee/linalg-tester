#pragma once
#include <cctype>

inline char to_upper(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}
