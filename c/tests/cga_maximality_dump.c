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

static int dump_row(size_t row)
{
    cga_pmg16 *graph = NULL;
    unsigned int choices[PAIR_COUNT];
    size_t pair_index;
    bool ancestral = (row % 2u) == 0u;
    bool maximal_ancestral = !ancestral;

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
    if (cga_pmg16_is_ancestral(graph, &ancestral) != CGA_STATUS_OK) {
        fprintf(stderr, "could not classify ancestry for PMG4 row %zu\n", row);
        cga_pmg16_destroy(graph);
        return 1;
    }
    if (cga_pmg16_is_maximal_ancestral(
            graph, &maximal_ancestral) != CGA_STATUS_OK) {
        fprintf(stderr,
                "could not classify maximal ancestry for PMG4 row %zu\n",
                row);
        cga_pmg16_destroy(graph);
        return 1;
    }

    printf("PMG4\t%zu\t%u%u%u%u%u%u\t%c\t%c\n",
           row,
           choices[0], choices[1], choices[2],
           choices[3], choices[4], choices[5],
           ancestral ? 'A' : 'N',
           maximal_ancestral ? 'M' : 'N');
    cga_pmg16_destroy(graph);
    return 0;
}

int main(void)
{
    size_t row;

    for (row = 0u; row < PMG_ROWS; ++row) {
        if (dump_row(row) != 0) {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
