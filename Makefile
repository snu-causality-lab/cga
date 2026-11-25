CC ?= cc
PYTHON ?= python3
.DEFAULT_GOAL := test

WARNINGS := -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion \
	-Wstrict-prototypes -Wmissing-prototypes -Wpointer-arith -Werror
BASE := -std=c11 $(WARNINGS) -pedantic -Ic/include
SANITIZERS := -fsanitize=address,undefined -fno-sanitize=enum \
	-fno-omit-frame-pointer
SOURCES := c/src/status.c c/src/backend.c c/src/leg.c c/src/adjustment.c
HEADERS := $(wildcard c/include/cga/*.h)
OBJECTS := $(patsubst c/src/%.c,build/%.o,$(SOURCES))
CONTRACTS := test_backend_contract test_separation_contract \
	test_projection_contract test_maximality_contract \
	test_mag_equivalence_contract test_leg_contract test_adjustment_contract \
	test_allocation_failure
DUMPS := msep projection maximality mag_equivalence leg adjustment
ALLOCATOR := -DCGA_CALLOC=cga_test_calloc -include c/tests/failing_allocator.h

ifeq ($(shell uname -s),Darwin)
ASAN_OPTIONS_VALUE := detect_leaks=0:abort_on_error=1
else
ASAN_OPTIONS_VALUE := detect_leaks=1:abort_on_error=1
endif

.PHONY: test test-python library test-c test-differential test-c-asan clean
test: test-python test-c test-differential test-c-asan

test-python:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest -p no:cacheprovider

library: build/libcga.a

build/libcga.a: $(OBJECTS)
	$(AR) rcs $@ $^

build/%.o: c/src/%.c $(HEADERS) Makefile | build
	$(CC) $(BASE) -O2 -c $< -o $@

test-c: $(addprefix build/,$(CONTRACTS)) build/leg_enumeration
	@set -e; for name in $(CONTRACTS); do ./build/$$name; done
	./build/leg_enumeration

build/%: c/tests/%.c build/libcga.a $(HEADERS) Makefile
	$(CC) $(BASE) -O2 $< build/libcga.a -o $@

build/test_allocation_failure: c/tests/test_allocation_failure.c $(SOURCES) $(HEADERS) c/tests/failing_allocator.h Makefile | build
	$(CC) $(BASE) -O2 $(ALLOCATOR) $(SOURCES) $< -o $@

build/leg_enumeration: c/examples/leg_enumeration.c build/libcga.a $(HEADERS) Makefile
	$(CC) $(BASE) -O2 $< build/libcga.a -o $@

test-differential: $(DUMPS:%=build/cga_%_dump)
	@set -e; for name in $(DUMPS); do \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) tests/differential/check_c_$${name}_dump.py ./build/cga_$${name}_dump; \
	done

test-c-asan: $(addprefix build-asan/,$(CONTRACTS))
	@set -e; for name in $(CONTRACTS); do \
		ASAN_OPTIONS=$(ASAN_OPTIONS_VALUE) UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ./build-asan/$$name; \
	done

build-asan/%: c/tests/%.c $(SOURCES) $(HEADERS) Makefile | build-asan
	$(CC) $(BASE) -O1 -g $(SANITIZERS) $(SOURCES) $< -o $@

build-asan/test_allocation_failure: c/tests/test_allocation_failure.c $(SOURCES) $(HEADERS) c/tests/failing_allocator.h Makefile | build-asan
	$(CC) $(BASE) -O1 -g $(SANITIZERS) $(ALLOCATOR) $(SOURCES) $< -o $@

build build-asan:
	mkdir -p $@

clean:
	rm -rf build build-asan
