#include "cga/leg.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Input: n followed by two endpoint digits per lexicographic vertex pair.
 * Digits are NONE=0, TAIL=1, ARROW=2, CIRCLE=3; n < 2 uses "-".
 * Output: SEED/LEG <case> <n> <marks>, followed by END <case> <count>.
 * This driver serializes public outputs; Python independently supplies truth.
 */

static int write_graph(
    const char *kind, size_t case_index, const cga_pmg16 *graph)
{
    size_t n;
    size_t left;
    size_t right;

    if (cga_pmg16_vertex_count(graph, &n) != CGA_STATUS_OK) {
        return 1;
    }
    printf("%s\t%zu\t%zu\t", kind, case_index, n);
    if (n < 2u) {
        putchar('-');
    }
    for (left = 0u; left < n; ++left) {
        for (right = left + 1u; right < n; ++right) {
            cga_endpoint_mark left_mark;
            cga_endpoint_mark right_mark;

            if (cga_pmg16_get_edge(
                    graph, left, right, &left_mark, &right_mark) !=
                CGA_STATUS_OK) {
                return 1;
            }
            printf("%u%u", (unsigned int)left_mark,
                   (unsigned int)right_mark);
        }
    }
    putchar('\n');
    return ferror(stdout) ? 1 : 0;
}

static int construct_pag(size_t n, const char *marks, cga_pmg16 **out_pag)
{
    cga_pmg16 *pag = NULL;
    size_t left;
    size_t right;
    size_t offset = 0u;

    if (n > 16u || (n < 2u ? strcmp(marks, "-") != 0
                              : strlen(marks) != n * (n - 1u))) {
        return 1;
    }
    if (cga_pmg16_create(n, &pag) != CGA_STATUS_OK) {
        return 1;
    }
    for (left = 0u; left < n; ++left) {
        for (right = left + 1u; right < n; ++right) {
            char left_digit = marks[offset++];
            char right_digit = marks[offset++];

            if (left_digit < '0' || left_digit > '3' ||
                right_digit < '0' || right_digit > '3' ||
                ((left_digit == '0') != (right_digit == '0'))) {
                cga_pmg16_destroy(pag);
                return 1;
            }
            if (left_digit != '0' &&
                cga_pmg16_set_edge(
                    pag, left, (cga_endpoint_mark)(left_digit - '0'),
                    right, (cga_endpoint_mark)(right_digit - '0')) !=
                    CGA_STATUS_OK) {
                cga_pmg16_destroy(pag);
                return 1;
            }
        }
    }
    *out_pag = pag;
    return 0;
}

static int dump_case(size_t case_index, size_t n, const char *marks)
{
    cga_pmg16 *pag = NULL;
    cga_pmg16 *graph = NULL;
    cga_leg16_iterator *iterator = NULL;
    bool has_graph = false;
    cga_status status;
    size_t count = 0u;
    int failed = 1;

    if (construct_pag(n, marks, &pag) != 0) {
        fprintf(stderr, "case %zu: malformed PAG record\n", case_index);
        return 1;
    }
    status = cga_pmg16_representative_leg(pag, &graph);
    if (status != CGA_STATUS_OK) {
        fprintf(stderr, "case %zu: representative: %s\n", case_index,
                cga_status_name(status));
        goto cleanup;
    }
    if (write_graph("SEED", case_index, graph) != 0) {
        goto cleanup;
    }
    cga_pmg16_destroy(graph);
    graph = NULL;
    status = cga_leg16_iterator_create(pag, &iterator);
    if (status != CGA_STATUS_OK) {
        fprintf(stderr, "case %zu: iterator create: %s\n", case_index,
                cga_status_name(status));
        goto cleanup;
    }
    for (;;) {
        status = cga_leg16_iterator_next(iterator, &graph, &has_graph);
        if (status != CGA_STATUS_OK) {
            fprintf(stderr, "case %zu: iterator next: %s\n", case_index,
                    cga_status_name(status));
            goto cleanup;
        }
        if (!has_graph) {
            break;
        }
        if (write_graph("LEG", case_index, graph) != 0) {
            goto cleanup;
        }
        cga_pmg16_destroy(graph);
        graph = NULL;
        ++count;
    }
    printf("END\t%zu\t%zu\n", case_index, count);
    failed = ferror(stdout) ? 1 : 0;

cleanup:
    cga_pmg16_destroy(graph);
    cga_leg16_iterator_destroy(iterator);
    cga_pmg16_destroy(pag);
    return failed;
}

int main(void)
{
    char line[1024];
    size_t case_index = 0u;

    while (fgets(line, sizeof(line), stdin) != NULL) {
        char marks[241];
        char extra;
        size_t n;

        if (sscanf(line, "%zu %240s %c", &n, marks, &extra) != 2 ||
            dump_case(case_index, n, marks) != 0) {
            fprintf(stderr, "LEG dump failed at input case %zu\n", case_index);
            return EXIT_FAILURE;
        }
        ++case_index;
    }
    if (ferror(stdin) || fflush(stdout) != 0) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
