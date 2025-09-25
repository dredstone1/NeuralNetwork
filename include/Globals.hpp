#ifndef GLOBAL
#define GLOBAL

#include <cstdint>
#include <tensor.hpp>

namespace nn::global {

// ============================================================================
// VISUALIZATION CONSTANTS
// ============================================================================

/// Default width for neuron visualization in pixels
constexpr std::uint32_t NEURON_WIDTH = 40;

/// Minimum allowed width for neuron visualization in pixels
constexpr float MIN_NEURON_WIDTH = 6.0f;

/// Maximum allowed width for neuron visualization in pixels (same as NEURON_WIDTH)
constexpr float MAX_NEURON_WIDTH = NEURON_WIDTH;

/// Minimum gap between visual elements in pixels
constexpr float MIN_GAP = 1.0f;

/// Minimum font size for text rendering in pixels
constexpr int MIN_FONT_SIZE = 5;

} // namespace nn::global

#endif // GLOBAL
