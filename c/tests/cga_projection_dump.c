#include "cga/backend.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define DB_BASE ((size_t)6u)
#define RAW_DB_ROWS ((size_t)216u)
#define ACYCLIC_DB_ROWS ((size_t)200u)
#define MASK_COUNT ((size_t)8u)
#define PAIR_COUNT ((size_t)3u)
#define VERTEX_COUNT ((size_t)3u)

static const size_t PAIR_LEFT[PAIR_COUNT] = {0u, 0u, 1u};
static const size_t PAIR_RIGHT[PAIR_COUNT] = {1u, 2u, 2u};

static void decode_choices(size_t row, unsigned int choices[PAIR_COUNT])
{
    size_t pair_index;
    size_t value = row;

    for (pair_index = 0u; pair_index < PAIR_COUNT; ++pair_index) {
        choices[pair_index] = (unsigned int)(value % DB_BASE);
        value /= DB_BASE;
    }
}

static bool choices_are_acyclic(const unsigned int choices[PAIR_COUNT])
{
    bool reach[VERTEX_COUNT][VERTEX_COUNT] = {{false}};
    size_t pair_index;
    size_t intermediate;
    size_t source;
    size_t target;

    for (pair_index = 0u; pair_index < PAIR_COUNT; ++pair_index) {
        const unsigned int choice = choices[pair_index];
        const size_t left = PAIR_LEFT[pair_index];
        const size_t right = PAIR_RIGHT[pair_index];

        if (choice == 1u || choice == 4u) {
            reach[left][right] = true;
        } else if (choice == 2u || choice == 5u) {
            reach[right][left] = true;
        }
    }
    for (intermediate = 0u;
         intermediate < VERTEX_COUNT;
         ++intermediate) {
        for (source = 0u; source < VERTEX_COUNT; ++source) {
            for (target = 0u; target < VERTEX_COUNT; ++target) {
                reach[source][target] =
                    reach[source][target] ||
                    (reach[source][intermediate] &&
                     reach[intermediate][target]);
            }
        }
    }
    for (source = 0u; source < VERTEX_COUNT; ++source) {
        if (reach[source][source]) {
            return false;
        }
    }
    return true;
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

static size_t retained_vertex_count(uint16_t retained)
{
    size_t count = 0u;
    size_t vertex;

    for (vertex = 0u; vertex < VERTEX_COUNT; ++vertex) {
        if ((retained & (uint16_t)(UINT32_C(1) << vertex)) !=
            UINT16_C(0)) {
            ++count;
        }
    }
    return count;
}

/*
 * Output atoms use a fixed stride of three: bit (3 * first + second).
 * Directed atoms use ordered pairs; bidirected atoms use first < second.
 */
static int encode_projection(
    const cga_directed_bidirected16 *graph,
    size_t expected_vertex_count,
    uint16_t *out_directed,
    uint16_t *out_bidirected)
{
    size_t vertex_count = SIZE_MAX;
    uint16_t directed = UINT16_C(0);
    uint16_t bidirected = UINT16_C(0);
    size_t first;
    size_t second;

    if (cga_directed_bidirected16_vertex_count(
            graph, &vertex_count) != CGA_STATUS_OK ||
        vertex_count != expected_vertex_count) {
        return 1;
    }
    for (first = 0u; first < vertex_count; ++first) {
        for (second = 0u; second < vertex_count; ++second) {
            bool present = false;
            const size_t bit_index = VERTEX_COUNT * first + second;

            if (first == second) {
                continue;
            }
            if (cga_directed_bidirected16_has_directed(
                    graph, first, second, &present) != CGA_STATUS_OK) {
                return 1;
            }
            if (present) {
                directed |= (uint16_t)(UINT32_C(1) << bit_index);
            }
        }
    }
    for (first = 0u; first < vertex_count; ++first) {
        for (second = first + 1u; second < vertex_count; ++second) {
            bool present = false;
            const size_t bit_index = VERTEX_COUNT * first + second;

            if (cga_directed_bidirected16_has_bidirected(
                    graph, first, second, &present) != CGA_STATUS_OK) {
                return 1;
            }
            if (present) {
                bidirected |= (uint16_t)(UINT32_C(1) << bit_index);
            }
        }
    }
    *out_directed = directed;
    *out_bidirected = bidirected;
    return 0;
}

int main(void)
{
    size_t raw_row;
    size_t valid_graph = 0u;
    size_t output_rows = 0u;

    for (raw_row = 0u; raw_row < RAW_DB_ROWS; ++raw_row) {
        cga_directed_bidirected16 *source = NULL;
        unsigned int choices[PAIR_COUNT];
        size_t pair_index;
        size_t mask_value;

        decode_choices(raw_row, choices);
        if (!choices_are_acyclic(choices)) {
            continue;
        }
        if (cga_directed_bidirected16_create(
                VERTEX_COUNT, &source) != CGA_STATUS_OK) {
            fprintf(stderr, "could not create DB row %zu\n", raw_row);
            return EXIT_FAILURE;
        }
        for (pair_index = 0u; pair_index < PAIR_COUNT; ++pair_index) {
            if (set_db_choice(
                    source,
                    PAIR_LEFT[pair_index],
                    PAIR_RIGHT[pair_index],
                    choices[pair_index]) != 0) {
                fprintf(stderr, "could not construct DB row %zu\n", raw_row);
                cga_directed_bidirected16_destroy(source);
                return EXIT_FAILURE;
            }
        }

        for (mask_value = 0u; mask_value < MASK_COUNT; ++mask_value) {
            cga_directed_bidirected16 *projected = NULL;
            const uint16_t retained = (uint16_t)mask_value;
            const size_t projected_vertices =
                retained_vertex_count(retained);
            uint16_t directed;
            uint16_t bidirected;
            cga_status status;

            status = cga_directed_bidirected16_latent_projection(
                source, retained, &projected);
            if (status != CGA_STATUS_OK) {
                fprintf(stderr,
                        "projection row %zu mask %zu returned status %d\n",
                        raw_row, mask_value, (int)status);
                cga_directed_bidirected16_destroy(source);
                return EXIT_FAILURE;
            }
            if (encode_projection(
                    projected, projected_vertices,
                    &directed, &bidirected) != 0) {
                fprintf(stderr,
                        "could not encode projection row %zu mask %zu\n",
                        raw_row, mask_value);
                cga_directed_bidirected16_destroy(projected);
                cga_directed_bidirected16_destroy(source);
                return EXIT_FAILURE;
            }
            printf("LP\t%zu\t%zu\t%u%u%u\t%zu\t%zu\t%03x\t%03x\n",
                   valid_graph, raw_row,
                   choices[0], choices[1], choices[2],
                   mask_value, projected_vertices,
                   (unsigned int)directed, (unsigned int)bidirected);
            ++output_rows;
            cga_directed_bidirected16_destroy(projected);
        }
        ++valid_graph;
        cga_directed_bidirected16_destroy(source);
    }

    if (valid_graph != ACYCLIC_DB_ROWS ||
        output_rows != ACYCLIC_DB_ROWS * MASK_COUNT) {
        fprintf(stderr,
                "projection census is %zu graphs/%zu rows, "
                "expected %zu/%zu\n",
                valid_graph, output_rows,
                ACYCLIC_DB_ROWS, ACYCLIC_DB_ROWS * MASK_COUNT);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
