#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <dlfcn.h>

inline std::vector<void *> preload_libs(const std::vector<std::string> &paths)
{
    std::vector<void *> handles;
    for (const auto &p : paths) {
        void *h = dlopen(p.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!h) {
            std::fprintf(stderr, "dlopen preload(%s): %s\n", p.c_str(), dlerror());
            std::exit(EXIT_FAILURE);
        }
        handles.push_back(h);
    }
    return handles;
}

inline void close_libs(std::vector<void *> &handles)
{
    for (void *h : handles) dlclose(h);
    handles.clear();
}

inline void *load_sym(void *lib, const char *name)
{
    dlerror(); /* clear */
    void *sym = dlsym(lib, name);
    const char *err = dlerror();
    if (err) {
        std::fprintf(stderr, "dlsym(%s): %s\n", name, err);
        std::exit(EXIT_FAILURE);
    }
    return sym;
}
