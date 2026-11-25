#include "cga/backend.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DB_BASE ((size_t)6u)
#define DB_ROWS ((size_t)216u)
#define PMG_BASE ((size_t)5u)
#define PMG_ROWS ((size_t)125u)
#define PAIR_COUNT ((size_t)3u)
#define QUERY_COUNT ((size_t)18u)
#define QUERY_MASK_LIMIT ((size_t)8u)

static const size_t PAIR_LEFT[PAIR_COUNT] = {0u, 0u, 1u};
static const size_t PAIR_RIGHT[PAIR_COUNT] = {1u, 2u, 2u};

static void decode_choices(
    size_t row, size_t base, unsigned int choices[PAIR_COUNT])
{
    size_t pair_index;
    size_t value = row;

    for (pair_index = 0u; pair_index < PAIR_COUNT; ++pair_index) {
        choices[pair_index] = (unsigned int)(value % base);
        value /= base;
    }
}

static int set_db_choice(
    cga_directed_bidirected16 *graph,
    size_t left,
    size_t right,
    unsigned int choice)
{
    cga_status status;

    switch (choice) {
    case 0u:
        return 0;
    case 1u:
        status = cga_directed_bidirected16_set_directed(
            graph, left, right, true);
        break;
    case 2u:
        status = cga_directed_bidirected16_set_directed(
            graph, right, left, true);
        break;
    case 3u:
        status = cga_directed_bidirected16_set_bidirected(
            graph, left, right, true);
        break;
    case 4u:
        status = cga_directed_bidirected16_set_directed(
            graph, left, right, true);
        if (status == CGA_STATUS_OK) {
            status = cga_directed_bidirected16_set_bidirected(
                graph, left, right, true);
        }
        break;
    case 5u:
        status = cga_directed_bidirected16_set_directed(
            graph, right, left, true);
        if (status == CGA_STATUS_OK) {
            status = cga_directed_bidirected16_set_bidirected(
                graph, left, right, true);
        }
        break;
    default:
        return 1;
    }
    return status == CGA_STATUS_OK ? 0 : 1;
}

static int set_pmg_choice(
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

static int dump_db_queries(const cga_directed_bidirected16 *graph, size_t row)
{
    size_t left_value;
    size_t right_value;
    size_t given_value;
    size_t query_index = 0u;

    for (left_value = 1u; left_value < QUERY_MASK_LIMIT; ++left_value) {
        for (right_value = 1u; right_value < QUERY_MASK_LIMIT; ++right_value) {
            for (given_value = 0u;
                 given_value < QUERY_MASK_LIMIT;
                 ++given_value) {
                const uint16_t left = (uint16_t)left_value;
                const uint16_t right = (uint16_t)right_value;
                const uint16_t given = (uint16_t)given_value;
                bool separated;
                bool before;
                cga_status status;

                if ((left & right) != UINT16_C(0) ||
                    (left & given) != UINT16_C(0) ||
                    (right & given) != UINT16_C(0)) {
                    continue;
                }
                separated = (query_index % 2u) == 0u;
                before = separated;
                status = cga_directed_bidirected16_m_separated(
                    graph, left, right, given, &separated);

                if (status == CGA_STATUS_OK) {
                    putchar(separated ? 'S' : 'C');
                } else if (status == CGA_STATUS_INVALID_GRAPH) {
                    if (separated != before) {
                        fprintf(stderr,
                                "DB row %zu changed output on INVALID_GRAPH\n",
                                row);
                        return 1;
                    }
                    putchar('I');
                } else {
                    fprintf(stderr,
                            "DB row %zu returned unexpected status %d\n",
                            row, (int)status);
                    return 1;
                }
                ++query_index;
            }
        }
    }
    if (query_index != QUERY_COUNT) {
        fprintf(stderr, "DB query census is %zu, expected %zu\n",
                query_index, QUERY_COUNT);
        return 1;
    }
    return 0;
}

static int dump_pmg_queries(const cga_pmg16 *graph, size_t row)
{
    size_t left_value;
    size_t right_value;
    size_t given_value;
    size_t query_index = 0u;

    for (left_value = 1u; left_value < QUERY_MASK_LIMIT; ++left_value) {
        for (right_value = 1u; right_value < QUERY_MASK_LIMIT; ++right_value) {
            for (given_value = 0u;
                 given_value < QUERY_MASK_LIMIT;
                 ++given_value) {
                const uint16_t left = (uint16_t)left_value;
                const uint16_t right = (uint16_t)right_value;
                const uint16_t given = (uint16_t)given_value;
                bool separated;
                bool before;
                cga_status status;

                if ((left & right) != UINT16_C(0) ||
                    (left & given) != UINT16_C(0) ||
                    (right & given) != UINT16_C(0)) {
                    continue;
                }
                separated = (query_index % 2u) != 0u;
                before = separated;
                status = cga_pmg16_m_separated(
                    graph, left, right, given, &separated);

                if (status == CGA_STATUS_OK) {
                    putchar(separated ? 'S' : 'C');
                } else if (status == CGA_STATUS_INVALID_GRAPH) {
                    if (separated != before) {
                        fprintf(stderr,
                                "PMG row %zu changed output on INVALID_GRAPH\n",
                                row);
                        return 1;
                    }
                    putchar('I');
                } else {
                    fprintf(stderr,
                            "PMG row %zu returned unexpected status %d\n",
                            row, (int)status);
                    return 1;
                }
                ++query_index;
            }
        }
    }
    if (query_index != QUERY_COUNT) {
        fprintf(stderr, "PMG query census is %zu, expected %zu\n",
                query_index, QUERY_COUNT);
        return 1;
    }
    return 0;
}

static int dump_directed_bidirected_rows(void)
{
    size_t row;

    for (row = 0u; row < DB_ROWS; ++row) {
        cga_directed_bidirected16 *graph = NULL;
        unsigned int choices[PAIR_COUNT];
        size_t pair_index;

        decode_choices(row, DB_BASE, choices);
        if (cga_directed_bidirected16_create(3u, &graph) != CGA_STATUS_OK) {
            fprintf(stderr, "could not create DB row %zu\n", row);
            return 1;
        }
        for (pair_index = 0u; pair_index < PAIR_COUNT; ++pair_index) {
            if (set_db_choice(
                    graph,
                    PAIR_LEFT[pair_index],
                    PAIR_RIGHT[pair_index],
                    choices[pair_index]) != 0) {
                fprintf(stderr, "could not construct DB row %zu\n", row);
                cga_directed_bidirected16_destroy(graph);
                return 1;
            }
        }

        printf("DB\t%zu\t%u%u%u\t", row,
               choices[0], choices[1], choices[2]);
        if (dump_db_queries(graph, row) != 0) {
            cga_directed_bidirected16_destroy(graph);
            return 1;
        }
        putchar('\n');
        cga_directed_bidirected16_destroy(graph);
    }
    return 0;
}

static int dump_endpoint_marked_rows(void)
{
    size_t row;

    for (row = 0u; row < PMG_ROWS; ++row) {
        cga_pmg16 *graph = NULL;
        unsigned int choices[PAIR_COUNT];
        size_t pair_index;
        bool ancestral = false;

        decode_choices(row, PMG_BASE, choices);
        if (cga_pmg16_create(3u, &graph) != CGA_STATUS_OK) {
            fprintf(stderr, "could not create PMG row %zu\n", row);
            return 1;
        }
        for (pair_index = 0u; pair_index < PAIR_COUNT; ++pair_index) {
            if (set_pmg_choice(
                    graph,
                    PAIR_LEFT[pair_index],
                    PAIR_RIGHT[pair_index],
                    choices[pair_index]) != 0) {
                fprintf(stderr, "could not construct PMG row %zu\n", row);
                cga_pmg16_destroy(graph);
                return 1;
            }
        }
        if (cga_pmg16_is_ancestral(graph, &ancestral) != CGA_STATUS_OK) {
            fprintf(stderr, "could not classify PMG row %zu\n", row);
            cga_pmg16_destroy(graph);
            return 1;
        }

        printf("PMG\t%zu\t%u%u%u\t%c\t", row,
               choices[0], choices[1], choices[2],
               ancestral ? 'A' : 'N');
        if (dump_pmg_queries(graph, row) != 0) {
            cga_pmg16_destroy(graph);
            return 1;
        }
        putchar('\n');
        cga_pmg16_destroy(graph);
    }
    return 0;
}

int main(void)
{
    if (dump_directed_bidirected_rows() != 0 ||
        dump_endpoint_marked_rows() != 0) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
