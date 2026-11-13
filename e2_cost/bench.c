/* Experiment E2: time per weight of a matrix-vector product under each weight format.
 *
 * Usage: bench <machine_name>
 * Prints one JSON line per (format, regime), then the memory-copy bandwidth of the machine.
 *
 * The matrix has COLUMNS activations per row, so the activations always fit in the cache and
 * only the weights are streamed. "in_cache" reuses one row; "out_of_cache" streams a matrix
 * far larger than the caches.
 */
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernels.h"

#define COLUMNS 4096
#define BLOCKS_PER_ROW (COLUMNS / WEIGHTS_PER_BLOCK)
#define OUT_OF_CACHE_ROWS 16384
#define TIMED_REPETITIONS 5
#define COPY_BYTES ((size_t)256 << 20)
#define NANOSECONDS_PER_SECOND 1e9
#define BYTES_PER_GIGABYTE 1e9

#ifndef BUILD_FLAGS
#define BUILD_FLAGS "unknown"
#endif

typedef struct {
    const char *name;
    size_t rows;
    size_t passes;
} Regime;

static const Regime REGIMES[] = {
    {"in_cache", 1, OUT_OF_CACHE_ROWS},
    {"out_of_cache", OUT_OF_CACHE_ROWS, 1},
};

static double seconds_now(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / NANOSECONDS_PER_SECOND;
}

static void *allocate(size_t bytes) {
    void *memory = malloc(bytes ? bytes : 1);
    if (!memory) {
        fprintf(stderr, "out of memory allocating %zu bytes\n", bytes);
        exit(1);
    }
    return memory;
}

static float matrix_vector_sum(const WeightFormat *format, const uint8_t *elements, const uint8_t *scales,
                               const float *activations, const Regime *regime) {
    float checksum = 0.0f;
    for (size_t pass = 0; pass < regime->passes; pass++)
        for (size_t row = 0; row < regime->rows; row++)
            checksum += format->dot(elements + row * BLOCKS_PER_ROW * format->element_bytes_per_block,
                                    scales + row * BLOCKS_PER_ROW * format->scale_bytes_per_block,
                                    activations, BLOCKS_PER_ROW);
    return checksum;
}

static void report(const char *machine, const WeightFormat *format, const Regime *regime, double seconds,
                   float checksum) {
    double weights = (double)regime->rows * regime->passes * COLUMNS;
    printf("{\"machine\": \"%s\", \"build\": \"%s\", \"format\": \"%s\", \"regime\": \"%s\", "
           "\"weights\": %.0f, \"bytes_per_weight\": %.5f, \"ns_per_weight\": %.4f, "
           "\"weight_gb_per_s\": %.3f, \"checksum\": %.6g}\n",
           machine, BUILD_FLAGS, format->name, regime->name, weights, bytes_per_weight(format),
           seconds / weights * NANOSECONDS_PER_SECOND,
           weights * bytes_per_weight(format) / seconds / BYTES_PER_GIGABYTE, checksum);
    fflush(stdout);
}

static void time_format(const char *machine, const WeightFormat *format, const Regime *regime,
                        const float *activations) {
    size_t blocks = regime->rows * BLOCKS_PER_ROW;
    uint8_t *elements = allocate(blocks * format->element_bytes_per_block);
    uint8_t *scales = allocate(blocks * format->scale_bytes_per_block);
    double fastest = 0.0;
    float checksum = 0.0f;
    format->fill(elements, scales, blocks, 1);
    for (int repetition = 0; repetition < TIMED_REPETITIONS; repetition++) {
        double started = seconds_now();
        checksum = matrix_vector_sum(format, elements, scales, activations, regime);
        double elapsed = seconds_now() - started;
        if (repetition == 0 || elapsed < fastest) fastest = elapsed;
    }
    report(machine, format, regime, fastest, checksum);
    free(elements);
    free(scales);
}

/* Called through a volatile pointer so the compiler cannot remove the copy being timed. */
static void *(*volatile copy_bytes)(void *, const void *, size_t) = memcpy;

static void time_memory_copy(const char *machine) {
    uint8_t *source = allocate(COPY_BYTES);
    uint8_t *destination = allocate(COPY_BYTES);
    double fastest = 0.0;
    memset(source, 1, COPY_BYTES);
    memset(destination, 2, COPY_BYTES);
    for (int repetition = 0; repetition < TIMED_REPETITIONS; repetition++) {
        double started = seconds_now();
        copy_bytes(destination, source, COPY_BYTES);
        double elapsed = seconds_now() - started;
        if (repetition == 0 || elapsed < fastest) fastest = elapsed;
    }
    printf("{\"machine\": \"%s\", \"build\": \"%s\", \"format\": \"memory_copy\", \"bytes\": %zu, "
           "\"copy_gb_per_s\": %.3f, \"checksum\": %d}\n",
           machine, BUILD_FLAGS, COPY_BYTES, (double)COPY_BYTES / fastest / BYTES_PER_GIGABYTE,
           destination[COPY_BYTES / 2]);
    free(source);
    free(destination);
}

int main(int argc, char **argv) {
    const char *machine = argc > 1 ? argv[1] : "unnamed";
    float activations[COLUMNS];
    uint32_t state = 3;
    for (size_t i = 0; i < COLUMNS; i++) activations[i] = (float)(next_random(&state) % 2001) / 1000.0f - 1.0f;
    for (size_t regime = 0; regime < sizeof(REGIMES) / sizeof(REGIMES[0]); regime++)
        for (size_t format = 0; format < WEIGHT_FORMAT_COUNT; format++)
            time_format(machine, &WEIGHT_FORMATS[format], &REGIMES[regime], activations);
    time_memory_copy(machine);
    return 0;
}
