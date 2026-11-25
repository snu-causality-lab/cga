#include "cga/backend.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define PMG_BASE ((size_t)5u)
#define PMG_ROWS ((size_t)15625u)
#define PAIR_COUNT ((size_t)6u)

static const size_t PAIR_LEFT[PAIR_COUNT] = {
    0u, 0u, 0u, 1u, 1u, 2u
};
static const size_t PAIR_RIGHT[PAIR_COUNT] = {
    1u, 2u, 3u, 2u, 3u, 3u
};

static void decode_choices(
    size_t row, unsigned int choices[PAIR_COUNT])
{
    size_t pair_index;
    size_t value = row;

    for (pair_index = 0u; pair_index < PAIR_COUNT; ++pair_index) {
        choices[pair_index] = (unsigned int)(value % PMG_BASE);
        value /= PMG_BASE;
    }
}

static int set_choice(
    cga_pmg16 *graph,
    size_t left,
    size_t right,
    unsigned int choice)
{
    cga_endpoint_mark left_mark;
    cga_endpoint_mark right_mark;

    switch (choice) {
    case 0u:
        return 0;
    case 1u:
        left_mark = CGA_ENDPOINT_TAIL;
        right_mark = CGA_ENDPOINT_TAIL;
        break;
    case 2u:
        left_mark = CGA_ENDPOINT_TAIL;
        right_mark = CGA_ENDPOINT_ARROW;
        break;
    case 3u:
        left_mark = CGA_ENDPOINT_ARROW;
        right_mark = CGA_ENDPOINT_TAIL;
        break;
    case 4u:
        left_mark = CGA_ENDPOINT_ARROW;
        right_mark = CGA_ENDPOINT_ARROW;
        break;
    default:
        return 1;
    }
    return cga_pmg16_set_edge(
               graph, left, left_mark, right, right_mark) == CGA_STATUS_OK
        ? 0
        : 1;
}

static int construct_graph(
    size_t row,
    cga_pmg16 **out_graph,
    unsigned int choices[PAIR_COUNT])
{
    cga_pmg16 *graph = NULL;
    size_t pair_index;

    decode_choices(row, choices);
    if (cga_pmg16_create(4u, &graph) != CGA_STATUS_OK) {
        fprintf(stderr, "could not create PMG4 row %zu\n", row);
        return 1;
    }
    for (pair_index = 0u; pair_index < PAIR_COUNT; ++pair_index) {
        if (set_choice(
                graph,
                PAIR_LEFT[pair_index],
                PAIR_RIGHT[pair_index],
                choices[pair_index]) != 0) {
            fprintf(stderr, "could not construct PMG4 row %zu\n", row);
            cga_pmg16_destroy(graph);
            return 1;
        }
    }
    *out_graph = graph;
    return 0;
}

static int compare_valid_mags_both_directions(
    const cga_pmg16 *left,
    const cga_pmg16 *right,
    bool *out_equivalent)
{
    bool left_right_from_false = false;
    bool left_right_from_true = true;
    bool right_left_from_false = false;
    bool right_left_from_true = true;
    cga_status statuses[4];
    size_t status_index;

    statuses[0] = (cga_status)cga_pmg16_are_markov_equivalent(
        left, right, &left_right_from_false);
    statuses[1] = (cga_status)cga_pmg16_are_markov_equivalent(
        left, right, &left_right_from_true);
    statuses[2] = (cga_status)cga_pmg16_are_markov_equivalent(
        right, left, &right_left_from_false);
    statuses[3] = (cga_status)cga_pmg16_are_markov_equivalent(
        right, left, &right_left_from_true);
    for (status_index = 0u; status_index < 4u; ++status_index) {
        if (statuses[status_index] != CGA_STATUS_OK) {
            return 1;
        }
    }
    if (left_right_from_false != left_right_from_true ||
        right_left_from_false != right_left_from_true ||
        left_right_from_false != right_left_from_false) {
        return 1;
    }
    *out_equivalent = left_right_from_false;
    return 0;
}

static int dump_row(
    size_t row,
    cga_pmg16 *representatives[PMG_ROWS],
    size_t representative_rows[PMG_ROWS],
    size_t *representative_count)
{
    cga_pmg16 *graph = NULL;
    unsigned int choices[PAIR_COUNT];
    cga_status status_false;
    cga_status status_true;
    bool equivalent_from_false = false;
    bool equivalent_from_true = true;
    bool equivalent;
    size_t representative_index;
    size_t first_representative = row;
    bool found_representative = false;
    bool retained_as_representative = false;

    if (construct_graph(row, &graph, choices) != 0) {
        return 1;
    }

    status_false = (cga_status)cga_pmg16_are_markov_equivalent(
        graph, graph, &equivalent_from_false);
    status_true = (cga_status)cga_pmg16_are_markov_equivalent(
        graph, graph, &equivalent_from_true);
    if (status_false == CGA_STATUS_INVALID_GRAPH &&
        status_true == CGA_STATUS_INVALID_GRAPH) {
        if (equivalent_from_false || !equivalent_from_true) {
            fprintf(stderr,
                    "invalid PMG4 row %zu changed its output\n", row);
            cga_pmg16_destroy(graph);
            return 1;
        }
        printf("PMG4-EQ\t%zu\t%u%u%u%u%u%u\tI\t-\n",
               row,
               choices[0], choices[1], choices[2],
               choices[3], choices[4], choices[5]);
        cga_pmg16_destroy(graph);
        return 0;
    }
    if (status_false != CGA_STATUS_OK || status_true != CGA_STATUS_OK ||
        !equivalent_from_false || !equivalent_from_true) {
        fprintf(stderr,
                "PMG4 row %zu did not classify as self-equivalent MAG\n",
                row);
        cga_pmg16_destroy(graph);
        return 1;
    }

    for (representative_index = 0u;
         representative_index < *representative_count;
         ++representative_index) {
        if (compare_valid_mags_both_directions(
                graph, representatives[representative_index],
                &equivalent) != 0) {
            fprintf(stderr,
                    "could not compare PMG4 row %zu with representative %zu\n",
                    row, representative_rows[representative_index]);
            cga_pmg16_destroy(graph);
            return 1;
        }
        if (equivalent) {
            if (found_representative) {
                fprintf(stderr,
                        "PMG4 row %zu matched multiple representatives\n",
                        row);
                cga_pmg16_destroy(graph);
                return 1;
            }
            found_representative = true;
            first_representative =
                representative_rows[representative_index];
        }
    }

    if (!found_representative) {
        representatives[*representative_count] = graph;
        representative_rows[*representative_count] = row;
        *representative_count += 1u;
        retained_as_representative = true;
    }

    printf("PMG4-EQ\t%zu\t%u%u%u%u%u%u\tM\t%zu\n",
           row,
           choices[0], choices[1], choices[2],
           choices[3], choices[4], choices[5],
           first_representative);
    if (!retained_as_representative) {
        cga_pmg16_destroy(graph);
    }
    return 0;
}

static void destroy_representatives(
    cga_pmg16 *representatives[PMG_ROWS],
    size_t representative_count)
{
    size_t representative_index;

    for (representative_index = 0u;
         representative_index < representative_count;
         ++representative_index) {
        cga_pmg16_destroy(representatives[representative_index]);
    }
}

int main(void)
{
    static cga_pmg16 *representatives[PMG_ROWS];
    static size_t representative_rows[PMG_ROWS];
    size_t representative_count = 0u;
    size_t row;

    for (row = 0u; row < PMG_ROWS; ++row) {
        if (dump_row(
                row, representatives, representative_rows,
                &representative_count) != 0) {
            destroy_representatives(
                representatives, representative_count);
            return EXIT_FAILURE;
        }
    }
    destroy_representatives(representatives, representative_count);
    return EXIT_SUCCESS;
}
