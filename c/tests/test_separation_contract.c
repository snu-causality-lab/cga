#include "cga/backend.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "separation contract failure at line %d: %s\n", \
                    __LINE__, #condition);                                     \
            return 1;                                                          \
        }                                                                      \
    } while (false)

_Static_assert(CGA_ENDPOINT_NONE == 0,
               "CGA_ENDPOINT_NONE ABI value changed");
_Static_assert(CGA_ENDPOINT_TAIL == 1,
               "CGA_ENDPOINT_TAIL ABI value changed");
_Static_assert(CGA_ENDPOINT_ARROW == 2,
               "CGA_ENDPOINT_ARROW ABI value changed");
_Static_assert(CGA_ENDPOINT_CIRCLE == 3,
               "CGA_ENDPOINT_CIRCLE ABI value changed");
_Static_assert(CGA_OPERATION_M_SEPARATION == 1,
               "CGA_OPERATION_M_SEPARATION ABI value changed");
_Static_assert(CGA_OPERATION_ANCESTRAL_CLASSIFICATION == 2,
               "CGA_OPERATION_ANCESTRAL_CLASSIFICATION ABI value changed");

static uint16_t vertex_bit(size_t vertex)
{
    return (uint16_t)(UINT16_C(1) << vertex);
}

static int check_operation_capabilities(void)
{
    bool supported = false;
    size_t max_vertices = 0u;
    bool before_supported;
    size_t before_max_vertices;

    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16, CGA_OPERATION_M_SEPARATION,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(supported);
    CHECK(max_vertices == 16u);

    supported = false;
    max_vertices = 0u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_DIRECTED_BIDIRECTED16,
              CGA_OPERATION_M_SEPARATION,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(supported);
    CHECK(max_vertices == 16u);

    supported = false;
    max_vertices = 0u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_ANCESTRAL_CLASSIFICATION,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(supported);
    CHECK(max_vertices == 16u);

    supported = true;
    max_vertices = 99u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_DIRECTED_BIDIRECTED16,
              CGA_OPERATION_ANCESTRAL_CLASSIFICATION,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(!supported);
    CHECK(max_vertices == 0u);

    supported = true;
    max_vertices = 99u;
    before_supported = supported;
    before_max_vertices = max_vertices;
    CHECK(cga_backend_query_operation(
              (cga_backend_family)999,
              CGA_OPERATION_M_SEPARATION,
              &supported, &max_vertices) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(supported == before_supported);
    CHECK(max_vertices == before_max_vertices);

    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              (cga_backend_operation)999,
              &supported, &max_vertices) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(supported == before_supported);
    CHECK(max_vertices == before_max_vertices);

    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_M_SEPARATION,
              NULL, &max_vertices) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(max_vertices == before_max_vertices);
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_M_SEPARATION,
              &supported, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(supported == before_supported);
    return 0;
}

static int check_pmg_edge_contract(void)
{
    cga_pmg16 *graph = NULL;
    cga_endpoint_mark first = CGA_ENDPOINT_CIRCLE;
    cga_endpoint_mark second = CGA_ENDPOINT_CIRCLE;
    cga_endpoint_mark before_first;
    cga_endpoint_mark before_second;

    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(cga_pmg16_get_edge(graph, 0u, 1u, &first, &second) ==
          CGA_STATUS_OK);
    CHECK(first == CGA_ENDPOINT_NONE);
    CHECK(second == CGA_ENDPOINT_NONE);

    CHECK(cga_pmg16_set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_get_edge(graph, 0u, 1u, &first, &second) ==
          CGA_STATUS_OK);
    CHECK(first == CGA_ENDPOINT_TAIL);
    CHECK(second == CGA_ENDPOINT_ARROW);
    CHECK(cga_pmg16_get_edge(graph, 1u, 0u, &first, &second) ==
          CGA_STATUS_OK);
    CHECK(first == CGA_ENDPOINT_ARROW);
    CHECK(second == CGA_ENDPOINT_TAIL);

    CHECK(cga_pmg16_set_edge(
              graph, 1u, CGA_ENDPOINT_CIRCLE,
              0u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_get_edge(graph, 0u, 1u, &first, &second) ==
          CGA_STATUS_OK);
    CHECK(first == CGA_ENDPOINT_ARROW);
    CHECK(second == CGA_ENDPOINT_CIRCLE);

    CHECK(cga_pmg16_set_edge(
              graph, 0u, CGA_ENDPOINT_NONE,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_pmg16_get_edge(graph, 0u, 1u, &first, &second) ==
          CGA_STATUS_OK);
    CHECK(first == CGA_ENDPOINT_ARROW);
    CHECK(second == CGA_ENDPOINT_CIRCLE);
    CHECK(cga_pmg16_set_edge(
              graph, 0u, (cga_endpoint_mark)99,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_pmg16_set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              0u, CGA_ENDPOINT_ARROW) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_pmg16_set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_ARROW) == CGA_STATUS_INVALID_ARGUMENT);

    first = CGA_ENDPOINT_TAIL;
    second = CGA_ENDPOINT_ARROW;
    before_first = first;
    before_second = second;
    CHECK(cga_pmg16_get_edge(graph, 0u, 3u, &first, &second) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(first == before_first);
    CHECK(second == before_second);
    CHECK(cga_pmg16_get_edge(graph, 0u, 1u, NULL, &second) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(second == before_second);
    CHECK(cga_pmg16_get_edge(graph, 0u, 1u, &first, NULL) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(first == before_first);

    CHECK(cga_pmg16_remove_edge(graph, 1u, 0u) == CGA_STATUS_OK);
    CHECK(cga_pmg16_get_edge(graph, 0u, 1u, &first, &second) ==
          CGA_STATUS_OK);
    CHECK(first == CGA_ENDPOINT_NONE);
    CHECK(second == CGA_ENDPOINT_NONE);
    CHECK(cga_pmg16_remove_edge(graph, 0u, 1u) == CGA_STATUS_OK);
    CHECK(cga_pmg16_remove_edge(graph, 0u, 0u) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_pmg16_remove_edge(graph, 0u, 3u) ==
          CGA_STATUS_INVALID_ARGUMENT);

    cga_pmg16_destroy(graph);
    return 0;
}

static int check_directed_bidirected_edge_contract(void)
{
    cga_directed_bidirected16 *graph = NULL;
    bool present = true;
    bool before;

    CHECK(cga_directed_bidirected16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_has_directed(
              graph, 0u, 1u, &present) == CGA_STATUS_OK);
    CHECK(!present);
    CHECK(cga_directed_bidirected16_has_bidirected(
              graph, 0u, 1u, &present) == CGA_STATUS_OK);
    CHECK(!present);

    CHECK(cga_directed_bidirected16_set_directed(
              graph, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              graph, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_has_directed(
              graph, 0u, 1u, &present) == CGA_STATUS_OK);
    CHECK(present);
    CHECK(cga_directed_bidirected16_has_directed(
              graph, 1u, 0u, &present) == CGA_STATUS_OK);
    CHECK(!present);
    CHECK(cga_directed_bidirected16_has_bidirected(
              graph, 1u, 0u, &present) == CGA_STATUS_OK);
    CHECK(present);

    CHECK(cga_directed_bidirected16_set_directed(
              graph, 0u, 1u, false) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              graph, 1u, 0u, false) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_has_directed(
              graph, 0u, 1u, &present) == CGA_STATUS_OK);
    CHECK(!present);
    CHECK(cga_directed_bidirected16_has_bidirected(
              graph, 0u, 1u, &present) == CGA_STATUS_OK);
    CHECK(!present);

    CHECK(cga_directed_bidirected16_set_directed(
              graph, 0u, 0u, true) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_directed_bidirected16_set_bidirected(
              graph, 0u, 3u, true) == CGA_STATUS_INVALID_ARGUMENT);
    present = true;
    before = present;
    CHECK(cga_directed_bidirected16_has_directed(
              graph, 0u, 3u, &present) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(present == before);
    CHECK(cga_directed_bidirected16_has_bidirected(
              graph, 0u, 1u, NULL) == CGA_STATUS_INVALID_ARGUMENT);

    cga_directed_bidirected16_destroy(graph);
    return 0;
}

static int check_ancestral_classification(void)
{
    cga_pmg16 *graph = NULL;
    bool ancestral = false;
    bool before;

    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(cga_pmg16_is_ancestral(graph, &ancestral) == CGA_STATUS_OK);
    CHECK(ancestral);

    CHECK(cga_pmg16_set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              graph, 2u, CGA_ENDPOINT_TAIL,
              0u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_is_ancestral(graph, &ancestral) == CGA_STATUS_OK);
    CHECK(!ancestral);
    cga_pmg16_destroy(graph);

    graph = NULL;
    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              graph, 0u, CGA_ENDPOINT_ARROW,
              2u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_is_ancestral(graph, &ancestral) == CGA_STATUS_OK);
    CHECK(!ancestral);
    cga_pmg16_destroy(graph);

    graph = NULL;
    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_TAIL) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              graph, 2u, CGA_ENDPOINT_TAIL,
              0u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_is_ancestral(graph, &ancestral) == CGA_STATUS_OK);
    CHECK(!ancestral);
    cga_pmg16_destroy(graph);

    graph = NULL;
    CHECK(cga_pmg16_create(2u, &graph) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              graph, 0u, CGA_ENDPOINT_CIRCLE,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_is_ancestral(graph, &ancestral) == CGA_STATUS_OK);
    CHECK(!ancestral);

    ancestral = true;
    before = ancestral;
    CHECK(cga_pmg16_is_ancestral(NULL, &ancestral) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(ancestral == before);
    CHECK(cga_pmg16_is_ancestral(graph, NULL) ==
          CGA_STATUS_INVALID_ARGUMENT);
    cga_pmg16_destroy(graph);
    return 0;
}

static int check_m_separation_argument_contract(void)
{
    cga_directed_bidirected16 *directed_bidirected = NULL;
    cga_pmg16 *pmg = NULL;
    bool separated = true;
    bool before;
    const uint16_t zero = UINT16_C(0);
    const uint16_t a = vertex_bit(0u);
    const uint16_t b = vertex_bit(1u);
    const uint16_t outside = vertex_bit(3u);

    CHECK(cga_directed_bidirected16_create(3u, &directed_bidirected) ==
          CGA_STATUS_OK);
    before = separated;
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, zero, b, zero, &separated) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(separated == before);
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, a, zero, zero, &separated) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(separated == before);
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, a, a, zero, &separated) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(separated == before);
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, a, b, a, &separated) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(separated == before);
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, a, b, b, &separated) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(separated == before);
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, a | outside, b, zero, &separated) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(separated == before);
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, a, b, zero, NULL) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_directed_bidirected16_m_separated(
              NULL, a, b, zero, &separated) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(separated == before);

    CHECK(cga_directed_bidirected16_set_directed(
              directed_bidirected, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              directed_bidirected, 1u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              directed_bidirected, 2u, 0u, true) == CGA_STATUS_OK);
    separated = false;
    before = separated;
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, a, b, zero, &separated) ==
          CGA_STATUS_INVALID_GRAPH);
    CHECK(separated == before);
    cga_directed_bidirected16_destroy(directed_bidirected);

    directed_bidirected = NULL;
    CHECK(cga_directed_bidirected16_create(2u, &directed_bidirected) ==
          CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              directed_bidirected, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              directed_bidirected, 1u, 0u, true) == CGA_STATUS_OK);
    separated = true;
    before = separated;
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, a, b, zero, &separated) ==
          CGA_STATUS_INVALID_GRAPH);
    CHECK(separated == before);
    cga_directed_bidirected16_destroy(directed_bidirected);

    CHECK(cga_pmg16_create(2u, &pmg) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 0u, CGA_ENDPOINT_CIRCLE,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    separated = true;
    before = separated;
    CHECK(cga_pmg16_m_separated(
              pmg, a, b, zero, &separated) == CGA_STATUS_INVALID_GRAPH);
    CHECK(separated == before);
    cga_pmg16_destroy(pmg);

    CHECK(cga_pmg16_create(3u, &pmg) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 2u, CGA_ENDPOINT_TAIL,
              0u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    separated = false;
    before = separated;
    CHECK(cga_pmg16_m_separated(
              pmg, a, b, zero, &separated) == CGA_STATUS_INVALID_GRAPH);
    CHECK(separated == before);
    cga_pmg16_destroy(pmg);
    return 0;
}

static int check_chain_collider_and_bow(void)
{
    cga_directed_bidirected16 *graph = NULL;
    bool separated = true;
    const uint16_t x = vertex_bit(0u);
    const uint16_t middle = vertex_bit(1u);
    const uint16_t y = vertex_bit(2u);
    const uint16_t descendant = vertex_bit(3u);

    CHECK(cga_directed_bidirected16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 1u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, y, UINT16_C(0), &separated) == CGA_STATUS_OK);
    CHECK(!separated);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, y, middle, &separated) == CGA_STATUS_OK);
    CHECK(separated);
    cga_directed_bidirected16_destroy(graph);

    graph = NULL;
    CHECK(cga_directed_bidirected16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 2u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, y, UINT16_C(0), &separated) == CGA_STATUS_OK);
    CHECK(separated);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, y, middle, &separated) == CGA_STATUS_OK);
    CHECK(!separated);
    cga_directed_bidirected16_destroy(graph);

    graph = NULL;
    CHECK(cga_directed_bidirected16_create(4u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 2u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 1u, 3u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, y, UINT16_C(0), &separated) == CGA_STATUS_OK);
    CHECK(separated);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, y, descendant, &separated) == CGA_STATUS_OK);
    CHECK(!separated);
    cga_directed_bidirected16_destroy(graph);

    graph = NULL;
    CHECK(cga_directed_bidirected16_create(4u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 1u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 2u, 3u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, descendant, UINT16_C(0), &separated) ==
          CGA_STATUS_OK);
    CHECK(!separated);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, descendant, middle, &separated) == CGA_STATUS_OK);
    CHECK(separated);
    cga_directed_bidirected16_destroy(graph);

    /*
     * Evans--Richardson bow fixture.  The vertex sequence X-M-Y denotes
     * both X <- M <- Y (blocked by M) and X <-> M <- Y (opened by M).
     * A traversal that collapses the two X-M edge atoms gets this wrong.
     */
    graph = NULL;
    CHECK(cga_directed_bidirected16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 1u, 0u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              graph, 2u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              graph, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, y, middle, &separated) == CGA_STATUS_OK);
    CHECK(!separated);
    CHECK(cga_directed_bidirected16_set_bidirected(
              graph, 0u, 1u, false) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_m_separated(
              graph, x, y, middle, &separated) == CGA_STATUS_OK);
    CHECK(separated);
    cga_directed_bidirected16_destroy(graph);
    return 0;
}

static int check_richardson_spirtes_figure9(void)
{
    cga_pmg16 *pmg = NULL;
    cga_directed_bidirected16 *directed_bidirected = NULL;
    bool ancestral = false;
    bool separated = true;
    size_t given_index;
    const uint16_t gamma = vertex_bit(2u);
    const uint16_t delta = vertex_bit(3u);
    const uint16_t givens[4] = {
        UINT16_C(0), vertex_bit(0u), vertex_bit(1u),
        (uint16_t)(vertex_bit(0u) | vertex_bit(1u))
    };

    /* Richardson--Spirtes (2002), Figure 9(a), p. 979. */
    CHECK(cga_pmg16_create(4u, &pmg) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 0u, CGA_ENDPOINT_ARROW,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 1u, CGA_ENDPOINT_ARROW,
              2u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 0u, CGA_ENDPOINT_ARROW,
              3u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 0u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 1u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_is_ancestral(pmg, &ancestral) == CGA_STATUS_OK);
    CHECK(ancestral);
    for (given_index = 0u; given_index < 4u; ++given_index) {
        separated = true;
        CHECK(cga_pmg16_m_separated(
                  pmg, gamma, delta, givens[given_index], &separated) ==
              CGA_STATUS_OK);
        CHECK(!separated);
    }
    cga_pmg16_destroy(pmg);

    CHECK(cga_directed_bidirected16_create(4u, &directed_bidirected) ==
          CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              directed_bidirected, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              directed_bidirected, 1u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              directed_bidirected, 0u, 3u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              directed_bidirected, 0u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              directed_bidirected, 1u, 3u, true) == CGA_STATUS_OK);
    for (given_index = 0u; given_index < 4u; ++given_index) {
        separated = true;
        CHECK(cga_directed_bidirected16_m_separated(
                  directed_bidirected, gamma, delta,
                  givens[given_index], &separated) == CGA_STATUS_OK);
        CHECK(!separated);
    }
    cga_directed_bidirected16_destroy(directed_bidirected);
    return 0;
}

static int check_zero_and_sixteen_vertex_boundaries(void)
{
    cga_pmg16 *pmg = NULL;
    cga_directed_bidirected16 *directed_bidirected = NULL;
    cga_endpoint_mark low_mark = CGA_ENDPOINT_NONE;
    cga_endpoint_mark high_mark = CGA_ENDPOINT_NONE;
    bool present = false;
    bool separated = true;
    bool before;
    const uint16_t low = vertex_bit(0u);
    const uint16_t high = vertex_bit(15u);

    CHECK(cga_pmg16_create(16u, &pmg) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 15u, CGA_ENDPOINT_TAIL,
              0u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_get_edge(
              pmg, 0u, 15u, &low_mark, &high_mark) == CGA_STATUS_OK);
    CHECK(low_mark == CGA_ENDPOINT_ARROW);
    CHECK(high_mark == CGA_ENDPOINT_TAIL);
    CHECK(cga_pmg16_m_separated(
              pmg, high, low, UINT16_C(0), &separated) == CGA_STATUS_OK);
    CHECK(!separated);
    cga_pmg16_destroy(pmg);

    CHECK(cga_directed_bidirected16_create(16u, &directed_bidirected) ==
          CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              directed_bidirected, 15u, 0u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              directed_bidirected, 0u, 15u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_has_directed(
              directed_bidirected, 15u, 0u, &present) == CGA_STATUS_OK);
    CHECK(present);
    CHECK(cga_directed_bidirected16_has_bidirected(
              directed_bidirected, 15u, 0u, &present) == CGA_STATUS_OK);
    CHECK(present);
    CHECK(cga_directed_bidirected16_m_separated(
              directed_bidirected, high, low, UINT16_C(0), &separated) ==
          CGA_STATUS_OK);
    CHECK(!separated);
    cga_directed_bidirected16_destroy(directed_bidirected);

    CHECK(cga_pmg16_create(0u, &pmg) == CGA_STATUS_OK);
    separated = false;
    before = separated;
    CHECK(cga_pmg16_m_separated(
              pmg, low, high, UINT16_C(0), &separated) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(separated == before);
    cga_pmg16_destroy(pmg);
    return 0;
}

int main(void)
{
    CHECK(check_operation_capabilities() == 0);
    CHECK(check_pmg_edge_contract() == 0);
    CHECK(check_directed_bidirected_edge_contract() == 0);
    CHECK(check_ancestral_classification() == 0);
    CHECK(check_m_separation_argument_contract() == 0);
    CHECK(check_chain_collider_and_bow() == 0);
    CHECK(check_richardson_spirtes_figure9() == 0);
    CHECK(check_zero_and_sixteen_vertex_boundaries() == 0);
    puts("C separation contract: passed");
    return 0;
}
