#include "cga/backend.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_VERTICES ((size_t)16u)

#define CHECK(condition)                                                       \
    do {                                                                       \
        if (!(condition)) {                                                    \
            fprintf(stderr, "maximality contract failure at line %d: %s\n", \
                    __LINE__, #condition);                                     \
            return 1;                                                          \
        }                                                                      \
    } while (false)

_Static_assert(CGA_OPERATION_MAXIMAL_ANCESTRAL_CLASSIFICATION == 5,
               "CGA maximal-ancestral operation ABI value changed");

typedef struct {
    size_t vertex_count;
    unsigned char marks[MAX_VERTICES][MAX_VERTICES];
} pmg_snapshot;

static int snapshot_pmg(const cga_pmg16 *graph, pmg_snapshot *out_snapshot)
{
    size_t first;
    size_t second;

    memset(out_snapshot, 0, sizeof(*out_snapshot));
    CHECK(cga_pmg16_vertex_count(
              graph, &out_snapshot->vertex_count) == CGA_STATUS_OK);
    for (first = 0u; first < out_snapshot->vertex_count; ++first) {
        for (second = first + 1u;
             second < out_snapshot->vertex_count;
             ++second) {
            cga_endpoint_mark first_mark = CGA_ENDPOINT_NONE;
            cga_endpoint_mark second_mark = CGA_ENDPOINT_NONE;

            CHECK(cga_pmg16_get_edge(
                      graph, first, second,
                      &first_mark, &second_mark) == CGA_STATUS_OK);
            out_snapshot->marks[first][second] =
                (unsigned char)first_mark;
            out_snapshot->marks[second][first] =
                (unsigned char)second_mark;
        }
    }
    return 0;
}

static int classify_without_mutation(
    const cga_pmg16 *graph, bool expected_value)
{
    pmg_snapshot before;
    pmg_snapshot after;
    bool actual_value = !expected_value;

    CHECK(snapshot_pmg(graph, &before) == 0);
    CHECK(cga_pmg16_is_maximal_ancestral(graph, &actual_value) ==
          CGA_STATUS_OK);
    CHECK(actual_value == expected_value);
    CHECK(snapshot_pmg(graph, &after) == 0);
    CHECK(memcmp(&before, &after, sizeof(before)) == 0);
    return 0;
}

static int set_edge(
    cga_pmg16 *graph,
    size_t first,
    cga_endpoint_mark first_mark,
    size_t second,
    cga_endpoint_mark second_mark)
{
    CHECK(cga_pmg16_set_edge(
              graph, first, first_mark,
              second, second_mark) == CGA_STATUS_OK);
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
              CGA_OPERATION_MAXIMAL_ANCESTRAL_CLASSIFICATION,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(supported);
    CHECK(max_vertices == 16u);

    supported = true;
    max_vertices = 99u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_DIRECTED_BIDIRECTED16,
              CGA_OPERATION_MAXIMAL_ANCESTRAL_CLASSIFICATION,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(!supported);
    CHECK(max_vertices == 0u);

    supported = true;
    max_vertices = 99u;
    before_supported = supported;
    before_max_vertices = max_vertices;
    CHECK(cga_backend_query_operation(
              (cga_backend_family)999,
              CGA_OPERATION_MAXIMAL_ANCESTRAL_CLASSIFICATION,
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
              CGA_OPERATION_MAXIMAL_ANCESTRAL_CLASSIFICATION,
              NULL, &max_vertices) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(max_vertices == before_max_vertices);
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_MAXIMAL_ANCESTRAL_CLASSIFICATION,
              &supported, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(supported == before_supported);
    return 0;
}

static int check_vacuous_and_argument_contract(void)
{
    cga_pmg16 *empty = NULL;
    cga_pmg16 *singleton = NULL;
    bool value = false;
    bool before;

    CHECK(cga_pmg16_create(0u, &empty) == CGA_STATUS_OK);
    CHECK(classify_without_mutation(empty, true) == 0);

    CHECK(cga_pmg16_create(1u, &singleton) == CGA_STATUS_OK);
    CHECK(classify_without_mutation(singleton, true) == 0);

    value = true;
    before = value;
    CHECK(cga_pmg16_is_maximal_ancestral(NULL, &value) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(value == before);
    CHECK(cga_pmg16_is_maximal_ancestral(singleton, NULL) ==
          CGA_STATUS_INVALID_ARGUMENT);

    cga_pmg16_destroy(singleton);
    cga_pmg16_destroy(empty);
    return 0;
}

static int check_basic_maximal_graphs(void)
{
    cga_pmg16 *dag = NULL;
    cga_pmg16 *undirected = NULL;

    CHECK(cga_pmg16_create(4u, &dag) == CGA_STATUS_OK);
    CHECK(set_edge(
              dag, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              dag, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              dag, 1u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(classify_without_mutation(dag, true) == 0);

    CHECK(cga_pmg16_create(4u, &undirected) == CGA_STATUS_OK);
    CHECK(set_edge(
              undirected, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_TAIL) == 0);
    CHECK(set_edge(
              undirected, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_TAIL) == 0);
    CHECK(set_edge(
              undirected, 2u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_TAIL) == 0);
    CHECK(classify_without_mutation(undirected, true) == 0);

    cga_pmg16_destroy(undirected);
    cga_pmg16_destroy(dag);
    return 0;
}

static int check_invalid_ancestry_is_scientific_false(void)
{
    cga_pmg16 *graph = NULL;

    /* Circle marks are outside the ancestral-graph domain. */
    CHECK(cga_pmg16_create(2u, &graph) == CGA_STATUS_OK);
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_CIRCLE,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(classify_without_mutation(graph, false) == 0);
    cga_pmg16_destroy(graph);

    /* A directed cycle is not ancestral. */
    graph = NULL;
    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_TAIL,
              0u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(classify_without_mutation(graph, false) == 0);
    cga_pmg16_destroy(graph);

    /* An arrowhead into an ancestor makes an almost-directed cycle. */
    graph = NULL;
    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_ARROW,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(classify_without_mutation(graph, false) == 0);
    cga_pmg16_destroy(graph);

    /* An undirected endpoint may have no incident arrowhead. */
    graph = NULL;
    CHECK(cga_pmg16_create(3u, &graph) == CGA_STATUS_OK);
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_TAIL) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_TAIL,
              0u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(classify_without_mutation(graph, false) == 0);
    cga_pmg16_destroy(graph);
    return 0;
}

static int add_figure9a_edges(
    cga_pmg16 *graph,
    size_t alpha,
    size_t beta,
    size_t gamma,
    size_t delta)
{
    CHECK(set_edge(
              graph, alpha, CGA_ENDPOINT_ARROW,
              beta, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, beta, CGA_ENDPOINT_ARROW,
              gamma, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, alpha, CGA_ENDPOINT_ARROW,
              delta, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, alpha, CGA_ENDPOINT_TAIL,
              gamma, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, beta, CGA_ENDPOINT_TAIL,
              delta, CGA_ENDPOINT_ARROW) == 0);
    return 0;
}

static int check_richardson_spirtes_figure9(void)
{
    cga_pmg16 *graph = NULL;
    bool ancestral = false;

    /*
     * Richardson--Spirtes (2002), Figure 9(a), p. 979.  With vertices
     * alpha=0, beta=1, gamma=2, delta=3, gamma and delta are joined by
     * the primitive inducing path gamma<->beta<->alpha<->delta.
     */
    CHECK(cga_pmg16_create(4u, &graph) == CGA_STATUS_OK);
    CHECK(add_figure9a_edges(graph, 0u, 1u, 2u, 3u) == 0);
    CHECK(cga_pmg16_is_ancestral(graph, &ancestral) == CGA_STATUS_OK);
    CHECK(ancestral);
    CHECK(classify_without_mutation(graph, false) == 0);

    /* Figure 9(b) adds the missing gamma<->delta edge. */
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_ARROW,
              3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(classify_without_mutation(graph, true) == 0);
    cga_pmg16_destroy(graph);
    return 0;
}

static int check_sixteen_vertex_high_bits(void)
{
    cga_pmg16 *graph = NULL;

    CHECK(cga_pmg16_create(16u, &graph) == CGA_STATUS_OK);
    CHECK(classify_without_mutation(graph, true) == 0);

    /* Repeat Figure 9(a) across bit 15 to catch narrow-mask errors. */
    CHECK(add_figure9a_edges(graph, 0u, 1u, 14u, 15u) == 0);
    CHECK(classify_without_mutation(graph, false) == 0);

    cga_pmg16_destroy(graph);
    return 0;
}

static int check_transitive_ancestor_witness(void)
{
    cga_pmg16 *graph = NULL;
    bool ancestral = false;

    /*
     * The primitive path 0<->2<->3<->1 needs two-edge ancestor paths:
     * 2->4->1 and 3->5->0.  No four-vertex topology can pin this closure.
     */
    CHECK(cga_pmg16_create(6u, &graph) == CGA_STATUS_OK);
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_ARROW,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_ARROW,
              3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 3u, CGA_ENDPOINT_ARROW,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_TAIL,
              4u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 4u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 3u, CGA_ENDPOINT_TAIL,
              5u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 5u, CGA_ENDPOINT_TAIL,
              0u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(cga_pmg16_is_ancestral(graph, &ancestral) == CGA_STATUS_OK);
    CHECK(ancestral);
    CHECK(classify_without_mutation(graph, false) == 0);

    cga_pmg16_destroy(graph);
    return 0;
}

int main(void)
{
    CHECK(check_operation_capabilities() == 0);
    CHECK(check_vacuous_and_argument_contract() == 0);
    CHECK(check_basic_maximal_graphs() == 0);
    CHECK(check_invalid_ancestry_is_scientific_false() == 0);
    CHECK(check_richardson_spirtes_figure9() == 0);
    CHECK(check_sixteen_vertex_high_bits() == 0);
    CHECK(check_transitive_ancestor_witness() == 0);
    puts("maximality contract: passed");
    return 0;
}
