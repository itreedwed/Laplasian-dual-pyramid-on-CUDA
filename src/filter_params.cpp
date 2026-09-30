#include "filter_params.h"

#include <cstdlib>
#include <iostream>

LdpFilterParams makeLdpFilterParams(int apertureK) {
    if (apertureK <= 0 || apertureK % 2 == 0) {
        std::cerr << "LDP filter error: aperture K must be a positive odd number; received K = "
                  << apertureK << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const float alpha = 2.0f / (static_cast<float>(apertureK) + 1.0f);
    return {alpha, 1.0f - alpha};
}
