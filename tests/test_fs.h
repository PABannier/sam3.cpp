#pragma once

#include <cerrno>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(_WIN32)
#include <direct.h>
#define SAM3_MKDIR(p) _mkdir(p)
#else
#define SAM3_MKDIR(p) mkdir(p, 0755)
#endif

static inline bool ensure_dir(const std::string & path) {
    if (path.empty()) {
        return true;
    }

    std::string cur;
    size_t pos = 0;
    if (path[0] == '/') {
        cur = "/";
        pos = 1;
    }

    while (pos <= path.size()) {
        size_t next = path.find('/', pos);
        if (next == std::string::npos) {
            next = path.size();
        }

        const std::string part = path.substr(pos, next - pos);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') {
                cur.push_back('/');
            }
            cur += part;
            if (SAM3_MKDIR(cur.c_str()) != 0 && errno != EEXIST) {
                return false;
            }
        }

        pos = next + 1;
    }

    return true;
}
