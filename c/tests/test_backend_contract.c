#include "cga/backend.h"
#include "cga/status.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "contract failure at line %d: %s\n", __LINE__,   \
                    #condition);                                               \
            return 1;                                                          \
        }                                                                      \
    } while (false)

_Static_assert(CGA_STATUS_OK == 0, "CGA_STATUS_OK ABI value changed");
_Static_assert(CGA_STATUS_INVALID_ARGUMENT == 1,
               "CGA_STATUS_INVALID_ARGUMENT ABI value changed");
_Static_assert(CGA_STATUS_INVALID_GRAPH == 2,
               "CGA_STATUS_INVALID_GRAPH ABI value changed");
_Static_assert(CGA_STATUS_UNSUPPORTED_SCOPE == 3,
               "CGA_STATUS_UNSUPPORTED_SCOPE ABI value changed");
_Static_assert(CGA_STATUS_RESOURCE_LIMIT == 4,
               "CGA_STATUS_RESOURCE_LIMIT ABI value changed");
_Static_assert(CGA_STATUS_MALFORMED_INPUT == 5,
               "CGA_STATUS_MALFORMED_INPUT ABI value changed");
_Static_assert(CGA_STATUS_BUFFER_TOO_SMALL == 6,
               "CGA_STATUS_BUFFER_TOO_SMALL ABI value changed");
_Static_assert(CGA_STATUS_ALLOCATION_FAILED == 7,
               "CGA_STATUS_ALLOCATION_FAILED ABI value changed");
_Static_assert(CGA_STATUS_INTERNAL_ERROR == 8,
               "CGA_STATUS_INTERNAL_ERROR ABI value changed");
_Static_assert(CGA_BACKEND_PMG16 == 1,
               "CGA_BACKEND_PMG16 ABI value changed");
_Static_assert(CGA_BACKEND_DIRECTED_BIDIRECTED16 == 2,
               "CGA_BACKEND_DIRECTED_BIDIRECTED16 ABI value changed");

static int check_status_contract(void)
{
    const char *unknown;

    CHECK(CGA_ABI_VERSION == 1u);
    CHECK(cga_abi_version() == CGA_ABI_VERSION);
    CHECK(strcmp(cga_status_name(CGA_STATUS_OK), "ok") == 0);
    CHECK(strcmp(cga_status_name(CGA_STATUS_INVALID_ARGUMENT),
                 "invalid argument") == 0);
    CHECK(strcmp(cga_status_name(CGA_STATUS_INVALID_GRAPH),
                 "invalid graph") == 0);
    CHECK(strcmp(cga_status_name(CGA_STATUS_UNSUPPORTED_SCOPE),
                 "unsupported scope") == 0);
    CHECK(strcmp(cga_status_name(CGA_STATUS_RESOURCE_LIMIT),
                 "resource limit") == 0);
    CHECK(strcmp(cga_status_name(CGA_STATUS_MALFORMED_INPUT),
                 "malformed input") == 0);
    CHECK(strcmp(cga_status_name(CGA_STATUS_BUFFER_TOO_SMALL),
                 "buffer too small") == 0);
    CHECK(strcmp(cga_status_name(CGA_STATUS_ALLOCATION_FAILED),
                 "allocation failed") == 0);
    CHECK(strcmp(cga_status_name(CGA_STATUS_INTERNAL_ERROR),
                 "internal error") == 0);
    unknown = cga_status_name((cga_status)999);
    CHECK(unknown != NULL);
    CHECK(strcmp(unknown, "unknown status") == 0);
    return 0;
}

static int check_capabilities(void)
{
    const cga_backend_operation leg_operations[] = {
        CGA_OPERATION_REPRESENTATIVE_LEG, CGA_OPERATION_LEG_ENUMERATION
    };
    size_t operation_index;
    cga_backend_capabilities pmg = {
        UINT32_MAX, SIZE_MAX, false, true
    };
    cga_backend_capabilities directed_bidirected = {
        UINT32_MAX, SIZE_MAX, true, false
    };
    cga_backend_capabilities before;
    cga_backend_capabilities invalid;

    CHECK(cga_backend_get_capabilities(CGA_BACKEND_PMG16, &pmg) ==
          CGA_STATUS_OK);
    CHECK(pmg.abi_version == 1u);
    CHECK(pmg.max_topology_vertices == 16u);
    CHECK(pmg.endpoint_marked);
    CHECK(!pmg.supports_bows);

    CHECK(cga_backend_get_capabilities(CGA_BACKEND_DIRECTED_BIDIRECTED16,
                                       &directed_bidirected) == CGA_STATUS_OK);
    CHECK(directed_bidirected.abi_version == 1u);
    CHECK(directed_bidirected.max_topology_vertices == 16u);
    CHECK(!directed_bidirected.endpoint_marked);
    CHECK(directed_bidirected.supports_bows);

    memset(&invalid, 0xa5, sizeof(invalid));
    memcpy(&before, &invalid, sizeof(before));
    CHECK(cga_backend_get_capabilities((cga_backend_family)999, &invalid) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&invalid, &before, sizeof(invalid)) == 0);
    CHECK(cga_backend_get_capabilities(CGA_BACKEND_PMG16, NULL) ==
          CGA_STATUS_INVALID_ARGUMENT);
    /* Unassigned operation IDs leave caller-owned outputs unchanged. */
    {
        bool supported = false;
        size_t max_vertices = 99u;

        CHECK(cga_backend_query_operation(CGA_BACKEND_DIRECTED_BIDIRECTED16,
                  (cga_backend_operation)10, &supported, &max_vertices) ==
              CGA_STATUS_INVALID_ARGUMENT);
        CHECK(!supported && max_vertices == 99u);
    }
    for (operation_index = 0u; operation_index < 2u; ++operation_index) {
        bool supported = false;
        size_t max_vertices = 99u;
        CHECK(cga_backend_query_operation(CGA_BACKEND_PMG16,
                  leg_operations[operation_index], &supported,
                  &max_vertices) == CGA_STATUS_OK);
        CHECK(supported && max_vertices == 16u);
        CHECK(cga_backend_query_operation(CGA_BACKEND_DIRECTED_BIDIRECTED16,
                  leg_operations[operation_index], &supported,
                  &max_vertices) == CGA_STATUS_OK);
        CHECK(!supported && max_vertices == 0u);
    }
    return 0;
}

static int check_pmg_capacity_and_atomicity(void)
{
    cga_pmg16 *graph = (cga_pmg16 *)(uintptr_t)1u;
    size_t vertices = 999u;

    CHECK(cga_pmg16_create(17u, &graph) == CGA_STATUS_UNSUPPORTED_SCOPE);
    CHECK(graph == (cga_pmg16 *)(uintptr_t)1u);
    CHECK(cga_pmg16_create(0u, &graph) == CGA_STATUS_OK);
    CHECK(graph != NULL);
    CHECK(cga_pmg16_vertex_count(graph, &vertices) == CGA_STATUS_OK);
    CHECK(vertices == 0u);
    cga_pmg16_destroy(graph);

    graph = NULL;
    CHECK(cga_pmg16_create(16u, &graph) == CGA_STATUS_OK);
    CHECK(graph != NULL);
    CHECK(cga_pmg16_vertex_count(graph, &vertices) == CGA_STATUS_OK);
    CHECK(vertices == 16u);
    CHECK(cga_pmg16_vertex_count(graph, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    cga_pmg16_destroy(graph);

    vertices = 999u;
    CHECK(cga_pmg16_vertex_count(NULL, &vertices) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(vertices == 999u);

    CHECK(cga_pmg16_create(1u, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    cga_pmg16_destroy(NULL);
    return 0;
}

static int check_directed_bidirected_capacity_and_atomicity(void)
{
    cga_directed_bidirected16 *graph =
        (cga_directed_bidirected16 *)(uintptr_t)1u;
    size_t vertices = 999u;

    CHECK(cga_directed_bidirected16_create(17u, &graph) ==
          CGA_STATUS_UNSUPPORTED_SCOPE);
    CHECK(graph == (cga_directed_bidirected16 *)(uintptr_t)1u);
    CHECK(cga_directed_bidirected16_create(0u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_vertex_count(graph, &vertices) ==
          CGA_STATUS_OK);
    CHECK(vertices == 0u);
    cga_directed_bidirected16_destroy(graph);

    graph = NULL;
    CHECK(cga_directed_bidirected16_create(16u, &graph) == CGA_STATUS_OK);
    CHECK(graph != NULL);
    CHECK(cga_directed_bidirected16_vertex_count(graph, &vertices) ==
          CGA_STATUS_OK);
    CHECK(vertices == 16u);
    vertices = 999u;
    CHECK(cga_directed_bidirected16_vertex_count(NULL, &vertices) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(vertices == 999u);
    CHECK(cga_directed_bidirected16_vertex_count(graph, NULL) ==
          CGA_STATUS_INVALID_ARGUMENT);
    cga_directed_bidirected16_destroy(graph);

    CHECK(cga_directed_bidirected16_create(1u, NULL) ==
          CGA_STATUS_INVALID_ARGUMENT);
    cga_directed_bidirected16_destroy(NULL);
    return 0;
}

int main(void)
{
    CHECK(check_status_contract() == 0);
    CHECK(check_capabilities() == 0);
    CHECK(check_pmg_capacity_and_atomicity() == 0);
    CHECK(check_directed_bidirected_capacity_and_atomicity() == 0);
    puts("backend contract: passed");
    return 0;
}
