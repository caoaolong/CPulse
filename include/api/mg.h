/*
 * api/mg.h - Mongoose include with Cygwin-safe defaults
 */

#ifndef API_MG_H
#define API_MG_H

#define MG_TLS MG_TLS_NONE
#define MG_ENABLE_EPOLL 0
#include "mongoose.h"

#endif /* API_MG_H */
