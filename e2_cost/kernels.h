/* Dot-product kernels: one block-scaled weight format against 32-bit float activations.
 *
 * Every kernel reads weights in blocks of WEIGHTS_PER_BLOCK. A format stores each block as
 * `element_bytes_per_block` bytes of elements plus `scale_bytes_per_block` bytes of scale.
 */
#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define WEIGHTS_PER_BLOCK 32

typedef float (*DotProduct)(const uint8_t *elements, const uint8_t *scales,
                            const float *activations, size_t blocks);
typedef void (*FillWeights)(uint8_t *elements, uint8_t *scales, size_t blocks, uint32_t seed);

typedef struct {
    const char *name;
    size_t element_bytes_per_block;
    size_t scale_bytes_per_block;
    DotProduct dot;
    FillWeights fill;
} WeightFormat;

extern const WeightFormat WEIGHT_FORMATS[];
extern const size_t WEIGHT_FORMAT_COUNT;

double bytes_per_weight(const WeightFormat *format);

/* The value of one 4-bit E2M1 code: sign, two exponent bits, one mantissa bit. */
float e2m1_value(uint8_t code);
/* The power-of-two block scale stored in one E8M0 byte. */
float e8m0_scale(uint8_t biased_exponent);
/* The weight at `index` of a block, decoded to a 32-bit float. */
float decoded_weight(const WeightFormat *format, const uint8_t *elements, const uint8_t *scales, size_t index);

uint32_t next_random(uint32_t *state);

#endif
