#include "cga/backend.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define MAX_VERTICES ((size_t)16u)

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "projection contract failure at line %d: %s\n", \
                    __LINE__, #condition);                                     \
            return 1;                                                          \
        }                                                                      \
    } while (false)

_Static_assert(CGA_OPERATION_INDUCED_SUBGRAPH == 3,
               "CGA_OPERATION_INDUCED_SUBGRAPH ABI value changed");
_Static_assert(CGA_OPERATION_LATENT_PROJECTION == 4,
               "CGA_OPERATION_LATENT_PROJECTION ABI value changed");

static uint16_t vertex_bit(size_t vertex)
{
    return (uint16_t)(UINT32_C(1) << vertex);
}

static void expect_directed(
    uint16_t rows[MAX_VERTICES], size_t source, size_t target)
{
    rows[source] |= vertex_bit(target);
}

static void expect_bidirected(
    uint16_t rows[MAX_VERTICES], size_t first, size_t second)
{
    rows[first] |= vertex_bit(second);
    rows[second] |= vertex_bit(first);
}

static int check_db_exact(
    const cga_directed_bidirected16 *graph,
    size_t expected_vertex_count,
    const uint16_t expected_directed[MAX_VERTICES],
    const uint16_t expected_bidirected[MAX_VERTICES])
{
    size_t actual_vertex_count = SIZE_MAX;
    size_t first;
    size_t second;

    CHECK(cga_directed_bidirected16_vertex_count(
              graph, &actual_vertex_count) == CGA_STATUS_OK);
    CHECK(actual_vertex_count == expected_vertex_count);
    for (first = 0u; first < expected_vertex_count; ++first) {
        for (second = 0u; second < expected_vertex_count; ++second) {
            bool present = false;
            const bool expected =
                (expected_directed[first] & vertex_bit(second)) !=
                UINT16_C(0);

            if (first == second) {
                continue;
            }
            CHECK(cga_directed_bidirected16_has_directed(
                      graph, first, second, &present) == CGA_STATUS_OK);
            CHECK(present == expected);
        }
    }
    for (first = 0u; first < expected_vertex_count; ++first) {
        for (second = first + 1u;
             second < expected_vertex_count;
             ++second) {
            bool present = false;
            const bool expected =
                (expected_bidirected[first] & vertex_bit(second)) !=
                UINT16_C(0);

            CHECK(cga_directed_bidirected16_has_bidirected(
                      graph, first, second, &present) == CGA_STATUS_OK);
            CHECK(present == expected);
        }
    }
    return 0;
}

static int check_pmg_pair(
    const cga_pmg16 *graph,
    size_t first,
    cga_endpoint_mark expected_first,
    size_t second,
    cga_endpoint_mark expected_second)
{
    cga_endpoint_mark actual_first = CGA_ENDPOINT_CIRCLE;
    cga_endpoint_mark actual_second = CGA_ENDPOINT_CIRCLE;

    CHECK(cga_pmg16_get_edge(
              graph, first, second,
              &actual_first, &actual_second) == CGA_STATUS_OK);
    CHECK(actual_first == expected_first);
    CHECK(actual_second == expected_second);
    return 0;
}

static int check_operation_capabilities(void)
{
    bool supported = false;
    size_t max_vertices = 0u;
    bool before_supported;
    size_t before_max_vertices;

    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_INDUCED_SUBGRAPH,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(supported);
    CHECK(max_vertices == 16u);

    supported = false;
    max_vertices = 0u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_DIRECTED_BIDIRECTED16,
              CGA_OPERATION_INDUCED_SUBGRAPH,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(supported);
    CHECK(max_vertices == 16u);

    supported = false;
    max_vertices = 0u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_DIRECTED_BIDIRECTED16,
              CGA_OPERATION_LATENT_PROJECTION,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(supported);
    CHECK(max_vertices == 16u);

    supported = true;
    max_vertices = 99u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_LATENT_PROJECTION,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(!supported);
    CHECK(max_vertices == 0u);

    supported = true;
    max_vertices = 99u;
    before_supported = supported;
    before_max_vertices = max_vertices;
    CHECK(cga_backend_query_operation(
              (cga_backend_family)999,
              CGA_OPERATION_LATENT_PROJECTION,
              &supported, &max_vertices) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(supported == before_supported);
    CHECK(max_vertices == before_max_vertices);

    CHECK(cga_backend_query_operation(
              CGA_BACKEND_DIRECTED_BIDIRECTED16,
              (cga_backend_operation)999,
              &supported, &max_vertices) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(supported == before_supported);
    CHECK(max_vertices == before_max_vertices);

    CHECK(cga_backend_query_operation(
              CGA_BACKEND_DIRECTED_BIDIRECTED16,
              CGA_OPERATION_LATENT_PROJECTION,
              NULL, &max_vertices) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(max_vertices == before_max_vertices);
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_DIRECTED_BIDIRECTED16,
              CGA_OPERATION_LATENT_PROJECTION,
              &supported, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(supported == before_supported);
    return 0;
}

static int check_empty_graphs_and_invalid_arguments(void)
{
    cga_directed_bidirected16 *db = NULL;
    cga_directed_bidirected16 *db_result =
        (cga_directed_bidirected16 *)(uintptr_t)1u;
    cga_pmg16 *pmg = NULL;
    cga_pmg16 *pmg_result = (cga_pmg16 *)(uintptr_t)1u;
    const uint16_t no_edges[MAX_VERTICES] = {0};
    size_t pmg_vertices = SIZE_MAX;

    CHECK(cga_directed_bidirected16_create(0u, &db) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_induced_subgraph(
              db, UINT16_C(0), &db_result) == CGA_STATUS_OK);
    CHECK(db_result != NULL);
    CHECK(db_result != db);
    CHECK(check_db_exact(db_result, 0u, no_edges, no_edges) == 0);
    cga_directed_bidirected16_destroy(db_result);

    db_result = (cga_directed_bidirected16 *)(uintptr_t)1u;
    CHECK(cga_directed_bidirected16_latent_projection(
              db, UINT16_C(0), &db_result) == CGA_STATUS_OK);
    CHECK(db_result != NULL);
    CHECK(db_result != db);
    CHECK(check_db_exact(db_result, 0u, no_edges, no_edges) == 0);
    cga_directed_bidirected16_destroy(db_result);

    db_result = (cga_directed_bidirected16 *)(uintptr_t)1u;
    CHECK(cga_directed_bidirected16_induced_subgraph(
              db, UINT16_C(1), &db_result) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(db_result == (cga_directed_bidirected16 *)(uintptr_t)1u);
    CHECK(cga_directed_bidirected16_latent_projection(
              db, UINT16_C(1), &db_result) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(db_result == (cga_directed_bidirected16 *)(uintptr_t)1u);
    CHECK(cga_directed_bidirected16_induced_subgraph(
              NULL, UINT16_C(0), &db_result) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(db_result == (cga_directed_bidirected16 *)(uintptr_t)1u);
    CHECK(cga_directed_bidirected16_latent_projection(
              NULL, UINT16_C(0), &db_result) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(db_result == (cga_directed_bidirected16 *)(uintptr_t)1u);
    CHECK(cga_directed_bidirected16_induced_subgraph(
              db, UINT16_C(0), NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(cga_directed_bidirected16_latent_projection(
              db, UINT16_C(0), NULL) == CGA_STATUS_INVALID_ARGUMENT);
    cga_directed_bidirected16_destroy(db);

    CHECK(cga_pmg16_create(0u, &pmg) == CGA_STATUS_OK);
    CHECK(cga_pmg16_induced_subgraph(
              pmg, UINT16_C(0), &pmg_result) == CGA_STATUS_OK);
    CHECK(pmg_result != NULL);
    CHECK(pmg_result != pmg);
    CHECK(cga_pmg16_vertex_count(
              pmg_result, &pmg_vertices) == CGA_STATUS_OK);
    CHECK(pmg_vertices == 0u);
    cga_pmg16_destroy(pmg_result);

    pmg_result = (cga_pmg16 *)(uintptr_t)1u;
    CHECK(cga_pmg16_induced_subgraph(
              pmg, UINT16_C(1), &pmg_result) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(pmg_result == (cga_pmg16 *)(uintptr_t)1u);
    CHECK(cga_pmg16_induced_subgraph(
              NULL, UINT16_C(0), &pmg_result) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(pmg_result == (cga_pmg16 *)(uintptr_t)1u);
    CHECK(cga_pmg16_induced_subgraph(
              pmg, UINT16_C(0), NULL) == CGA_STATUS_INVALID_ARGUMENT);
    cga_pmg16_destroy(pmg);
    return 0;
}

static int check_raw_induced_subgraphs_and_cycle_rejection(void)
{
    cga_directed_bidirected16 *db = NULL;
    cga_directed_bidirected16 *db_result = NULL;
    cga_directed_bidirected16 *sentinel =
        (cga_directed_bidirected16 *)(uintptr_t)1u;
    cga_pmg16 *pmg = NULL;
    cga_pmg16 *pmg_result = NULL;
    uint16_t directed[MAX_VERTICES] = {0};
    uint16_t bidirected[MAX_VERTICES] = {0};

    CHECK(cga_directed_bidirected16_create(2u, &db) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              db, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              db, 1u, 0u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              db, 0u, 1u, true) == CGA_STATUS_OK);
    expect_directed(directed, 0u, 1u);
    expect_directed(directed, 1u, 0u);
    expect_bidirected(bidirected, 0u, 1u);

    db_result = sentinel;
    CHECK(cga_directed_bidirected16_latent_projection(
              db, UINT16_C(3), &db_result) == CGA_STATUS_INVALID_GRAPH);
    CHECK(db_result == sentinel);
    CHECK(check_db_exact(db, 2u, directed, bidirected) == 0);

    db_result = sentinel;
    CHECK(cga_directed_bidirected16_latent_projection(
              db, UINT16_C(0), &db_result) == CGA_STATUS_INVALID_GRAPH);
    CHECK(db_result == sentinel);
    CHECK(check_db_exact(db, 2u, directed, bidirected) == 0);

    CHECK(cga_directed_bidirected16_induced_subgraph(
              db, UINT16_C(3), &db_result) == CGA_STATUS_OK);
    CHECK(check_db_exact(db_result, 2u, directed, bidirected) == 0);
    CHECK(check_db_exact(db, 2u, directed, bidirected) == 0);
    cga_directed_bidirected16_destroy(db_result);
    cga_directed_bidirected16_destroy(db);

    CHECK(cga_pmg16_create(3u, &pmg) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 0u, CGA_ENDPOINT_CIRCLE,
              1u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_TAIL) == CGA_STATUS_OK);
    CHECK(cga_pmg16_induced_subgraph(
              pmg, UINT16_C(7), &pmg_result) == CGA_STATUS_OK);
    CHECK(check_pmg_pair(
              pmg_result, 0u, CGA_ENDPOINT_CIRCLE,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(check_pmg_pair(
              pmg_result, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_TAIL) == 0);
    CHECK(check_pmg_pair(
              pmg_result, 0u, CGA_ENDPOINT_NONE,
              2u, CGA_ENDPOINT_NONE) == 0);
    CHECK(check_pmg_pair(
              pmg, 0u, CGA_ENDPOINT_CIRCLE,
              1u, CGA_ENDPOINT_ARROW) == 0);
    cga_pmg16_destroy(pmg_result);

    pmg_result = (cga_pmg16 *)(uintptr_t)1u;
    CHECK(cga_pmg16_induced_subgraph(
              pmg, UINT16_C(8), &pmg_result) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(pmg_result == (cga_pmg16 *)(uintptr_t)1u);
    cga_pmg16_destroy(pmg);
    return 0;
}

static int check_high_bit_reindexing(void)
{
    cga_directed_bidirected16 *db = NULL;
    cga_directed_bidirected16 *db_result = NULL;
    cga_pmg16 *pmg = NULL;
    cga_pmg16 *pmg_result = NULL;
    uint16_t directed[MAX_VERTICES] = {0};
    uint16_t bidirected[MAX_VERTICES] = {0};
    const uint16_t retained =
        (uint16_t)(vertex_bit(0u) | vertex_bit(15u));

    CHECK(cga_directed_bidirected16_create(16u, &db) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              db, 15u, 0u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              db, 0u, 15u, true) == CGA_STATUS_OK);
    expect_directed(directed, 1u, 0u);
    expect_bidirected(bidirected, 0u, 1u);

    CHECK(cga_directed_bidirected16_induced_subgraph(
              db, retained, &db_result) == CGA_STATUS_OK);
    CHECK(check_db_exact(db_result, 2u, directed, bidirected) == 0);
    cga_directed_bidirected16_destroy(db_result);

    CHECK(cga_directed_bidirected16_latent_projection(
              db, retained, &db_result) == CGA_STATUS_OK);
    CHECK(check_db_exact(db_result, 2u, directed, bidirected) == 0);
    cga_directed_bidirected16_destroy(db_result);
    cga_directed_bidirected16_destroy(db);

    CHECK(cga_pmg16_create(16u, &pmg) == CGA_STATUS_OK);
    CHECK(cga_pmg16_set_edge(
              pmg, 15u, CGA_ENDPOINT_CIRCLE,
              0u, CGA_ENDPOINT_ARROW) == CGA_STATUS_OK);
    CHECK(cga_pmg16_induced_subgraph(
              pmg, retained, &pmg_result) == CGA_STATUS_OK);
    CHECK(check_pmg_pair(
              pmg_result, 0u, CGA_ENDPOINT_ARROW,
              1u, CGA_ENDPOINT_CIRCLE) == 0);
    cga_pmg16_destroy(pmg_result);
    cga_pmg16_destroy(pmg);
    return 0;
}

static int check_rers_2023_figure_1_to_figure_3(void)
{
    cga_directed_bidirected16 *source = NULL;
    cga_directed_bidirected16 *projected = NULL;
    uint16_t source_directed[MAX_VERTICES] = {0};
    uint16_t source_bidirected[MAX_VERTICES] = {0};
    uint16_t expected_directed[MAX_VERTICES] = {0};
    uint16_t expected_bidirected[MAX_VERTICES] = {0};
    static const size_t edges[][2] = {
        {1u, 2u}, {1u, 3u}, {2u, 3u},
        {3u, 4u}, {0u, 2u}, {0u, 4u}
    };
    size_t edge_index;

    /*
     * Richardson, Evans, Robins, and Shpitser (2023), "Nested Markov
     * Properties for Acyclic Directed Mixed Graphs," AoS 51(1),
     * doi:10.1214/22-AOS2253, main Figures 1(i) and 3(i).  Figure 3(i)
     * reports 2<->4 after vertex 0 is hidden.  The literal-path differential
     * oracle independently gives the same edge; no fixture discrepancy is
     * known at adoption.
     */
    CHECK(cga_directed_bidirected16_create(5u, &source) == CGA_STATUS_OK);
    for (edge_index = 0u;
         edge_index < sizeof(edges) / sizeof(edges[0]);
         ++edge_index) {
        CHECK(cga_directed_bidirected16_set_directed(
                  source, edges[edge_index][0], edges[edge_index][1],
                  true) == CGA_STATUS_OK);
        expect_directed(
            source_directed, edges[edge_index][0], edges[edge_index][1]);
    }

    CHECK(cga_directed_bidirected16_latent_projection(
              source, UINT16_C(0x1e), &projected) == CGA_STATUS_OK);
    expect_directed(expected_directed, 0u, 1u);
    expect_directed(expected_directed, 0u, 2u);
    expect_directed(expected_directed, 1u, 2u);
    expect_directed(expected_directed, 2u, 3u);
    expect_bidirected(expected_bidirected, 1u, 3u);
    CHECK(check_db_exact(
              projected, 4u,
              expected_directed, expected_bidirected) == 0);
    CHECK(check_db_exact(
              source, 5u,
              source_directed, source_bidirected) == 0);

    cga_directed_bidirected16_destroy(projected);
    cga_directed_bidirected16_destroy(source);
    return 0;
}

static int check_evans_2021_figure_1_bow(void)
{
    cga_directed_bidirected16 *source = NULL;
    cga_directed_bidirected16 *projected = NULL;
    uint16_t expected_directed[MAX_VERTICES] = {0};
    uint16_t expected_bidirected[MAX_VERTICES] = {0};

    /*
     * Evans (2021), "Dependency in DAG Models with Hidden Variables,"
     * UAI, PMLR 161:813-822, p. 813 Figure 1.  Figure 1(a) is a->b->c,
     * h->b, h->c (h is vertex 3); Figure 1(b) reports both b->c and b<->c.
     * Literal path enumeration independently yields this bow; no fixture
     * discrepancy is known at adoption.
     */
    CHECK(cga_directed_bidirected16_create(4u, &source) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 1u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 3u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 3u, 2u, true) == CGA_STATUS_OK);

    CHECK(cga_directed_bidirected16_latent_projection(
              source, UINT16_C(7), &projected) == CGA_STATUS_OK);
    expect_directed(expected_directed, 0u, 1u);
    expect_directed(expected_directed, 1u, 2u);
    expect_bidirected(expected_bidirected, 1u, 2u);
    CHECK(check_db_exact(
              projected, 3u,
              expected_directed, expected_bidirected) == 0);

    cga_directed_bidirected16_destroy(projected);
    cga_directed_bidirected16_destroy(source);
    return 0;
}

static int check_definition_level_primitives(void)
{
    cga_directed_bidirected16 *source = NULL;
    cga_directed_bidirected16 *projected = NULL;
    uint16_t directed[MAX_VERTICES] = {0};
    uint16_t bidirected[MAX_VERTICES] = {0};

    /* A directed latent chain contracts to a directed edge. */
    CHECK(cga_directed_bidirected16_create(4u, &source) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 1u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 2u, 3u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_latent_projection(
              source, UINT16_C(9), &projected) == CGA_STATUS_OK);
    expect_directed(directed, 0u, 1u);
    CHECK(check_db_exact(projected, 2u, directed, bidirected) == 0);
    cga_directed_bidirected16_destroy(projected);
    cga_directed_bidirected16_destroy(source);

    /* A latent fork contracts to a bidirected edge. */
    source = NULL;
    projected = NULL;
    directed[0] = UINT16_C(0);
    directed[1] = UINT16_C(0);
    bidirected[0] = UINT16_C(0);
    bidirected[1] = UINT16_C(0);
    CHECK(cga_directed_bidirected16_create(3u, &source) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 1u, 0u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 1u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_latent_projection(
              source, UINT16_C(5), &projected) == CGA_STATUS_OK);
    expect_bidirected(bidirected, 0u, 1u);
    CHECK(check_db_exact(projected, 2u, directed, bidirected) == 0);
    cga_directed_bidirected16_destroy(projected);
    cga_directed_bidirected16_destroy(source);

    /* A hidden collider does not induce adjacency. */
    source = NULL;
    projected = NULL;
    bidirected[0] = UINT16_C(0);
    bidirected[1] = UINT16_C(0);
    CHECK(cga_directed_bidirected16_create(3u, &source) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 2u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_latent_projection(
              source, UINT16_C(5), &projected) == CGA_STATUS_OK);
    CHECK(check_db_exact(projected, 2u, directed, bidirected) == 0);
    cga_directed_bidirected16_destroy(projected);
    cga_directed_bidirected16_destroy(source);

    /* A<->L->B is a noncollider path with arrowheads at both endpoints. */
    source = NULL;
    projected = NULL;
    CHECK(cga_directed_bidirected16_create(3u, &source) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_bidirected(
              source, 0u, 1u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_set_directed(
              source, 1u, 2u, true) == CGA_STATUS_OK);
    CHECK(cga_directed_bidirected16_latent_projection(
              source, UINT16_C(5), &projected) == CGA_STATUS_OK);
    expect_bidirected(bidirected, 0u, 1u);
    CHECK(check_db_exact(projected, 2u, directed, bidirected) == 0);
    cga_directed_bidirected16_destroy(projected);
    cga_directed_bidirected16_destroy(source);
    return 0;
}

int main(void)
{
    CHECK(check_operation_capabilities() == 0);
    CHECK(check_empty_graphs_and_invalid_arguments() == 0);
    CHECK(check_raw_induced_subgraphs_and_cycle_rejection() == 0);
    CHECK(check_high_bit_reindexing() == 0);
    CHECK(check_rers_2023_figure_1_to_figure_3() == 0);
    CHECK(check_evans_2021_figure_1_bow() == 0);
    CHECK(check_definition_level_primitives() == 0);
    puts("projection contract: passed");
    return 0;
}
