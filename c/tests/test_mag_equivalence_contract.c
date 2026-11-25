#include "cga/backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_VERTICES ((size_t)16u)

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fprintf(stderr,                                                     \
                    "MAG equivalence contract failure at line %d: %s\n",      \
                    __LINE__, #condition);                                      \
            return 1;                                                           \
        }                                                                       \
    } while (false)

_Static_assert(CGA_OPERATION_MAG_MARKOV_EQUIVALENCE == 6,
               "CGA MAG-equivalence operation ABI value changed");

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

static int compare_without_mutation(
    const cga_pmg16 *left,
    const cga_pmg16 *right,
    cga_status expected_status,
    bool expected_value)
{
    pmg_snapshot left_before;
    pmg_snapshot left_mid;
    pmg_snapshot left_after;
    pmg_snapshot right_before;
    pmg_snapshot right_mid;
    pmg_snapshot right_after;
    cga_status status_from_false;
    cga_status status_from_true;
    bool actual_from_false = false;
    bool actual_from_true = true;

    CHECK(snapshot_pmg(left, &left_before) == 0);
    CHECK(snapshot_pmg(right, &right_before) == 0);
    status_from_false = (cga_status)cga_pmg16_are_markov_equivalent(
        left, right, &actual_from_false);
    CHECK(snapshot_pmg(left, &left_mid) == 0);
    CHECK(snapshot_pmg(right, &right_mid) == 0);
    CHECK(memcmp(&left_before, &left_mid, sizeof(left_before)) == 0);
    CHECK(memcmp(&right_before, &right_mid, sizeof(right_before)) == 0);
    status_from_true = (cga_status)cga_pmg16_are_markov_equivalent(
        left, right, &actual_from_true);
    CHECK(status_from_false == expected_status);
    CHECK(status_from_true == expected_status);
    if (expected_status == CGA_STATUS_OK) {
        CHECK(actual_from_false == expected_value);
        CHECK(actual_from_true == expected_value);
    } else {
        CHECK(!actual_from_false);
        CHECK(actual_from_true);
    }
    CHECK(snapshot_pmg(left, &left_after) == 0);
    CHECK(snapshot_pmg(right, &right_after) == 0);
    CHECK(memcmp(&left_before, &left_after, sizeof(left_before)) == 0);
    CHECK(memcmp(&right_before, &right_after, sizeof(right_before)) == 0);
    return 0;
}

static int compare_with_null_output_without_mutation(
    const cga_pmg16 *left,
    const cga_pmg16 *right)
{
    pmg_snapshot left_before;
    pmg_snapshot left_after;
    pmg_snapshot right_before;
    pmg_snapshot right_after;

    CHECK(snapshot_pmg(left, &left_before) == 0);
    CHECK(snapshot_pmg(right, &right_before) == 0);
    CHECK(cga_pmg16_are_markov_equivalent(left, right, NULL) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(snapshot_pmg(left, &left_after) == 0);
    CHECK(snapshot_pmg(right, &right_after) == 0);
    CHECK(memcmp(&left_before, &left_after, sizeof(left_before)) == 0);
    CHECK(memcmp(&right_before, &right_after, sizeof(right_before)) == 0);
    return 0;
}

static int check_query_error_atomicity(
    cga_backend_family family,
    cga_backend_operation operation)
{
    bool supported = false;
    size_t max_vertices = 0u;

    CHECK(cga_backend_query_operation(
              family, operation, &supported, &max_vertices) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(!supported);
    CHECK(max_vertices == 0u);

    supported = true;
    max_vertices = 99u;
    CHECK(cga_backend_query_operation(
              family, operation, &supported, &max_vertices) ==
          CGA_STATUS_INVALID_ARGUMENT);
    CHECK(supported);
    CHECK(max_vertices == 99u);
    return 0;
}

static int check_operation_capabilities(void)
{
    bool supported = false;
    size_t max_vertices = 0u;

    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_MAG_MARKOV_EQUIVALENCE,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(supported);
    CHECK(max_vertices == 16u);

    supported = true;
    max_vertices = 99u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_DIRECTED_BIDIRECTED16,
              CGA_OPERATION_MAG_MARKOV_EQUIVALENCE,
              &supported, &max_vertices) == CGA_STATUS_OK);
    CHECK(!supported);
    CHECK(max_vertices == 0u);

    CHECK(check_query_error_atomicity(
              (cga_backend_family)999,
              CGA_OPERATION_MAG_MARKOV_EQUIVALENCE) == 0);
    CHECK(check_query_error_atomicity(
              CGA_BACKEND_PMG16,
              (cga_backend_operation)999) == 0);

    max_vertices = 0u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_MAG_MARKOV_EQUIVALENCE,
              NULL, &max_vertices) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(max_vertices == 0u);
    max_vertices = 99u;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_MAG_MARKOV_EQUIVALENCE,
              NULL, &max_vertices) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(max_vertices == 99u);

    supported = false;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_MAG_MARKOV_EQUIVALENCE,
              &supported, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(!supported);
    supported = true;
    CHECK(cga_backend_query_operation(
              CGA_BACKEND_PMG16,
              CGA_OPERATION_MAG_MARKOV_EQUIVALENCE,
              &supported, NULL) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(supported);
    return 0;
}

static int check_vacuous_and_argument_contract(void)
{
    cga_pmg16 *empty_left = NULL;
    cga_pmg16 *empty_right = NULL;
    cga_pmg16 *singleton_left = NULL;
    cga_pmg16 *singleton_right = NULL;
    pmg_snapshot singleton_before;
    pmg_snapshot singleton_mid;
    pmg_snapshot singleton_after;
    bool value;

    CHECK(cga_pmg16_create(0u, &empty_left) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(0u, &empty_right) == CGA_STATUS_OK);
    CHECK(compare_without_mutation(
              empty_left, empty_right, CGA_STATUS_OK, true) == 0);

    CHECK(cga_pmg16_create(1u, &singleton_left) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(1u, &singleton_right) == CGA_STATUS_OK);
    CHECK(compare_without_mutation(
              singleton_left, singleton_right, CGA_STATUS_OK, true) == 0);

    CHECK(compare_without_mutation(
              empty_left, singleton_left,
              CGA_STATUS_INVALID_ARGUMENT, false) == 0);
    CHECK(compare_without_mutation(
              singleton_left, empty_left,
              CGA_STATUS_INVALID_ARGUMENT, false) == 0);

    CHECK(snapshot_pmg(singleton_left, &singleton_before) == 0);
    value = false;
    CHECK(cga_pmg16_are_markov_equivalent(
              NULL, singleton_left, &value) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(!value);
    CHECK(snapshot_pmg(singleton_left, &singleton_mid) == 0);
    CHECK(memcmp(
              &singleton_before, &singleton_mid,
              sizeof(singleton_before)) == 0);
    value = true;
    CHECK(cga_pmg16_are_markov_equivalent(
              NULL, singleton_left, &value) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(value);
    CHECK(snapshot_pmg(singleton_left, &singleton_after) == 0);
    CHECK(memcmp(
              &singleton_before, &singleton_after,
              sizeof(singleton_before)) == 0);
    CHECK(snapshot_pmg(singleton_left, &singleton_before) == 0);
    value = false;
    CHECK(cga_pmg16_are_markov_equivalent(
              singleton_left, NULL, &value) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(!value);
    CHECK(snapshot_pmg(singleton_left, &singleton_mid) == 0);
    CHECK(memcmp(
              &singleton_before, &singleton_mid,
              sizeof(singleton_before)) == 0);
    value = true;
    CHECK(cga_pmg16_are_markov_equivalent(
              singleton_left, NULL, &value) == CGA_STATUS_INVALID_ARGUMENT);
    CHECK(value);
    CHECK(snapshot_pmg(singleton_left, &singleton_after) == 0);
    CHECK(memcmp(
              &singleton_before, &singleton_after,
              sizeof(singleton_before)) == 0);
    CHECK(compare_with_null_output_without_mutation(
              singleton_left, singleton_right) == 0);

    cga_pmg16_destroy(singleton_right);
    cga_pmg16_destroy(singleton_left);
    cga_pmg16_destroy(empty_right);
    cga_pmg16_destroy(empty_left);
    return 0;
}

static int add_figure4_g1_edges(cga_pmg16 *graph)
{
    /* x=0, q=1, b=2, y=3. */
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_ARROW,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_ARROW,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_ARROW,
              3u, CGA_ENDPOINT_ARROW) == 0);
    return 0;
}

static int add_figure4_g2_edges(cga_pmg16 *graph)
{
    /* x=0, q=1, b=2, y=3. */
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_ARROW,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_ARROW,
              3u, CGA_ENDPOINT_ARROW) == 0);
    return 0;
}

static int add_figure4_g3_edges(cga_pmg16 *graph)
{
    /* x=0, q=1, b=2, y=3. */
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_ARROW) == 0);
    return 0;
}

static int check_ali_richardson_spirtes_figure4(void)
{
    cga_pmg16 *g1 = NULL;
    cga_pmg16 *g2 = NULL;
    cga_pmg16 *g3 = NULL;
    bool separated = false;

    /*
     * Ali, Richardson, and Spirtes (2009), "Markov equivalence for
     * ancestral graphs", Figure 4, p. 2815, DOI 10.1214/08-AOS626.  The paper
     * reports G1 and G2 equivalent, but G1 and G3 non-equivalent; its
     * displayed witness has x m-separated from y given q in G1 and
     * m-connected given q in G3.
     */
    CHECK(cga_pmg16_create(4u, &g1) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(4u, &g2) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(4u, &g3) == CGA_STATUS_OK);
    CHECK(add_figure4_g1_edges(g1) == 0);
    CHECK(add_figure4_g2_edges(g2) == 0);
    CHECK(add_figure4_g3_edges(g3) == 0);

    CHECK(cga_pmg16_m_separated(
              g1, UINT16_C(1), UINT16_C(8), UINT16_C(2),
              &separated) == CGA_STATUS_OK);
    CHECK(separated);
    CHECK(cga_pmg16_m_separated(
              g3, UINT16_C(1), UINT16_C(8), UINT16_C(2),
              &separated) == CGA_STATUS_OK);
    CHECK(!separated);

    CHECK(compare_without_mutation(g1, g2, CGA_STATUS_OK, true) == 0);
    CHECK(compare_without_mutation(g2, g1, CGA_STATUS_OK, true) == 0);
    CHECK(compare_without_mutation(g1, g3, CGA_STATUS_OK, false) == 0);
    CHECK(compare_without_mutation(g3, g1, CGA_STATUS_OK, false) == 0);
    CHECK(compare_without_mutation(g2, g3, CGA_STATUS_OK, false) == 0);

    cga_pmg16_destroy(g3);
    cga_pmg16_destroy(g2);
    cga_pmg16_destroy(g1);
    return 0;
}

static int check_fixed_index_semantics(void)
{
    cga_pmg16 *left = NULL;
    cga_pmg16 *reoriented = NULL;
    cga_pmg16 *relabelled = NULL;

    CHECK(cga_pmg16_create(3u, &left) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(3u, &reoriented) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(3u, &relabelled) == CGA_STATUS_OK);
    CHECK(set_edge(
              left, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              reoriented, 0u, CGA_ENDPOINT_ARROW,
              1u, CGA_ENDPOINT_TAIL) == 0);
    CHECK(set_edge(
              relabelled, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == 0);

    /* One saturated pair has the same model under either orientation. */
    CHECK(compare_without_mutation(
              left, reoriented, CGA_STATUS_OK, true) == 0);
    /* Vertex indices are fixed labels; isomorphic relabelling is not done. */
    CHECK(compare_without_mutation(
              left, relabelled, CGA_STATUS_OK, false) == 0);

    cga_pmg16_destroy(relabelled);
    cga_pmg16_destroy(reoriented);
    cga_pmg16_destroy(left);
    return 0;
}

static int add_figure9a_edges(cga_pmg16 *graph)
{
    /* Richardson--Spirtes (2002), Figure 9(a), alpha=0,beta=1,gamma=2,delta=3. */
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_ARROW,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_ARROW,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_ARROW,
              3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_ARROW) == 0);
    return 0;
}

static int check_invalid_graphs_fail_closed(void)
{
    cga_pmg16 *valid2 = NULL;
    cga_pmg16 *valid3 = NULL;
    cga_pmg16 *valid4 = NULL;
    cga_pmg16 *circle = NULL;
    cga_pmg16 *cycle = NULL;
    cga_pmg16 *nonmaximal = NULL;

    CHECK(cga_pmg16_create(2u, &valid2) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(3u, &valid3) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(4u, &valid4) == CGA_STATUS_OK);

    CHECK(cga_pmg16_create(2u, &circle) == CGA_STATUS_OK);
    CHECK(set_edge(
              circle, 0u, CGA_ENDPOINT_CIRCLE,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(compare_without_mutation(
              circle, valid2, CGA_STATUS_INVALID_GRAPH, false) == 0);
    CHECK(compare_without_mutation(
              valid2, circle, CGA_STATUS_INVALID_GRAPH, false) == 0);

    CHECK(cga_pmg16_create(3u, &cycle) == CGA_STATUS_OK);
    CHECK(set_edge(
              cycle, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              cycle, 1u, CGA_ENDPOINT_TAIL,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              cycle, 2u, CGA_ENDPOINT_TAIL,
              0u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(compare_without_mutation(
              cycle, valid3, CGA_STATUS_INVALID_GRAPH, false) == 0);
    CHECK(compare_without_mutation(
              valid3, cycle, CGA_STATUS_INVALID_GRAPH, false) == 0);

    CHECK(cga_pmg16_create(4u, &nonmaximal) == CGA_STATUS_OK);
    CHECK(add_figure9a_edges(nonmaximal) == 0);
    CHECK(compare_without_mutation(
              nonmaximal, valid4, CGA_STATUS_INVALID_GRAPH, false) == 0);
    CHECK(compare_without_mutation(
              valid4, nonmaximal, CGA_STATUS_INVALID_GRAPH, false) == 0);
    CHECK(compare_without_mutation(
              nonmaximal, nonmaximal,
              CGA_STATUS_INVALID_GRAPH, false) == 0);

    cga_pmg16_destroy(nonmaximal);
    cga_pmg16_destroy(cycle);
    cga_pmg16_destroy(circle);
    cga_pmg16_destroy(valid4);
    cga_pmg16_destroy(valid3);
    cga_pmg16_destroy(valid2);
    return 0;
}

static int add_order_two_shared_edges(cga_pmg16 *graph)
{
    /* x=0, q1=1, q2=2, b=3, y=4. */
    CHECK(set_edge(
              graph, 0u, CGA_ENDPOINT_TAIL,
              1u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_ARROW,
              2u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_ARROW,
              3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              3u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 1u, CGA_ENDPOINT_TAIL,
              4u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              graph, 2u, CGA_ENDPOINT_TAIL,
              4u, CGA_ENDPOINT_ARROW) == 0);
    return 0;
}

static int check_order_two_discriminating_collider(void)
{
    cga_pmg16 *order_two = NULL;
    cga_pmg16 *noncollider = NULL;
    bool separated = false;

    /* Constructed from Ali--Richardson--Spirtes Definition 3.11. */
    CHECK(cga_pmg16_create(5u, &order_two) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(5u, &noncollider) == CGA_STATUS_OK);
    CHECK(add_order_two_shared_edges(order_two) == 0);
    CHECK(add_order_two_shared_edges(noncollider) == 0);
    CHECK(set_edge(
              order_two, 3u, CGA_ENDPOINT_ARROW,
              4u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              noncollider, 3u, CGA_ENDPOINT_TAIL,
              4u, CGA_ENDPOINT_ARROW) == 0);

    CHECK(cga_pmg16_m_separated(
              order_two, UINT16_C(1), UINT16_C(16), UINT16_C(6),
              &separated) == CGA_STATUS_OK);
    CHECK(separated);
    CHECK(cga_pmg16_m_separated(
              noncollider, UINT16_C(1), UINT16_C(16), UINT16_C(6),
              &separated) == CGA_STATUS_OK);
    CHECK(!separated);
    CHECK(compare_without_mutation(
              order_two, noncollider, CGA_STATUS_OK, false) == 0);
    CHECK(compare_without_mutation(
              noncollider, order_two, CGA_STATUS_OK, false) == 0);

    cga_pmg16_destroy(noncollider);
    cga_pmg16_destroy(order_two);
    return 0;
}

static int check_sixteen_vertex_high_bits(void)
{
    cga_pmg16 *empty_left = NULL;
    cga_pmg16 *empty_right = NULL;
    cga_pmg16 *figure4_g1 = NULL;
    cga_pmg16 *figure4_g2 = NULL;
    cga_pmg16 *figure4_g3 = NULL;
    cga_pmg16 *one_edge = NULL;

    CHECK(cga_pmg16_create(16u, &empty_left) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(16u, &empty_right) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(16u, &figure4_g1) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(16u, &figure4_g2) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(16u, &figure4_g3) == CGA_STATUS_OK);
    CHECK(cga_pmg16_create(16u, &one_edge) == CGA_STATUS_OK);
    CHECK(set_edge(
              one_edge, 14u, CGA_ENDPOINT_TAIL,
              15u, CGA_ENDPOINT_ARROW) == 0);

    /* Figure 4 shifted from 0..3 to 12..15 exercises high triple indices. */
    CHECK(set_edge(
              figure4_g1, 12u, CGA_ENDPOINT_ARROW,
              13u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              figure4_g1, 13u, CGA_ENDPOINT_ARROW,
              14u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              figure4_g1, 13u, CGA_ENDPOINT_TAIL,
              15u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              figure4_g1, 14u, CGA_ENDPOINT_ARROW,
              15u, CGA_ENDPOINT_ARROW) == 0);

    CHECK(set_edge(
              figure4_g2, 12u, CGA_ENDPOINT_TAIL,
              13u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              figure4_g2, 13u, CGA_ENDPOINT_ARROW,
              14u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              figure4_g2, 13u, CGA_ENDPOINT_TAIL,
              15u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              figure4_g2, 14u, CGA_ENDPOINT_ARROW,
              15u, CGA_ENDPOINT_ARROW) == 0);

    CHECK(set_edge(
              figure4_g3, 12u, CGA_ENDPOINT_TAIL,
              13u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              figure4_g3, 14u, CGA_ENDPOINT_TAIL,
              13u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              figure4_g3, 13u, CGA_ENDPOINT_TAIL,
              15u, CGA_ENDPOINT_ARROW) == 0);
    CHECK(set_edge(
              figure4_g3, 14u, CGA_ENDPOINT_TAIL,
              15u, CGA_ENDPOINT_ARROW) == 0);

    CHECK(compare_without_mutation(
              empty_left, empty_right, CGA_STATUS_OK, true) == 0);
    CHECK(compare_without_mutation(
              empty_left, one_edge, CGA_STATUS_OK, false) == 0);
    CHECK(compare_without_mutation(
              one_edge, empty_left, CGA_STATUS_OK, false) == 0);
    CHECK(compare_without_mutation(
              figure4_g1, figure4_g2, CGA_STATUS_OK, true) == 0);
    CHECK(compare_without_mutation(
              figure4_g2, figure4_g1, CGA_STATUS_OK, true) == 0);
    CHECK(compare_without_mutation(
              figure4_g1, figure4_g3, CGA_STATUS_OK, false) == 0);
    CHECK(compare_without_mutation(
              figure4_g3, figure4_g1, CGA_STATUS_OK, false) == 0);

    cga_pmg16_destroy(one_edge);
    cga_pmg16_destroy(figure4_g3);
    cga_pmg16_destroy(figure4_g2);
    cga_pmg16_destroy(figure4_g1);
    cga_pmg16_destroy(empty_right);
    cga_pmg16_destroy(empty_left);
    return 0;
}

int main(void)
{
    CHECK(check_operation_capabilities() == 0);
    CHECK(check_vacuous_and_argument_contract() == 0);
    CHECK(check_ali_richardson_spirtes_figure4() == 0);
    CHECK(check_fixed_index_semantics() == 0);
    CHECK(check_invalid_graphs_fail_closed() == 0);
    CHECK(check_order_two_discriminating_collider() == 0);
    CHECK(check_sixteen_vertex_high_bits() == 0);
    puts("MAG equivalence contract: passed");
    return 0;
}
