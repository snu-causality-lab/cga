#ifndef CGA_STATUS_H
#define CGA_STATUS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CGA_ABI_VERSION UINT32_C(1)

/* Numeric values are part of ABI version 1. */
typedef enum {
    CGA_STATUS_OK = 0,
    CGA_STATUS_INVALID_ARGUMENT = 1,
    CGA_STATUS_INVALID_GRAPH = 2,
    CGA_STATUS_UNSUPPORTED_SCOPE = 3,
    CGA_STATUS_RESOURCE_LIMIT = 4,
    CGA_STATUS_MALFORMED_INPUT = 5,
    CGA_STATUS_BUFFER_TOO_SMALL = 6,
    CGA_STATUS_ALLOCATION_FAILED = 7,
    CGA_STATUS_INTERNAL_ERROR = 8
} cga_status;

uint32_t cga_abi_version(void);
const char *cga_status_name(cga_status status);

#ifdef __cplusplus
}
#endif

#endif
