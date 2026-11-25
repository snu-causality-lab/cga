#include "cga/status.h"

uint32_t cga_abi_version(void)
{
    return CGA_ABI_VERSION;
}

const char *cga_status_name(cga_status status)
{
    switch (status) {
    case CGA_STATUS_OK:
        return "ok";
    case CGA_STATUS_INVALID_ARGUMENT:
        return "invalid argument";
    case CGA_STATUS_INVALID_GRAPH:
        return "invalid graph";
    case CGA_STATUS_UNSUPPORTED_SCOPE:
        return "unsupported scope";
    case CGA_STATUS_RESOURCE_LIMIT:
        return "resource limit";
    case CGA_STATUS_MALFORMED_INPUT:
        return "malformed input";
    case CGA_STATUS_BUFFER_TOO_SMALL:
        return "buffer too small";
    case CGA_STATUS_ALLOCATION_FAILED:
        return "allocation failed";
    case CGA_STATUS_INTERNAL_ERROR:
        return "internal error";
    default:
        return "unknown status";
    }
}
