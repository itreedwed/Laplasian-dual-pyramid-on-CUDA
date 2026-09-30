#pragma once

struct LdpFilterParams {
    float alpha;
    float beta;
};

// Validates an odd positive aperture and derives the recursive coefficients.
LdpFilterParams makeLdpFilterParams(int apertureK);
