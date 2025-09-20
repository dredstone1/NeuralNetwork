#ifndef GLOBAL
#define GLOBAL

#include <cstdint>
#include <tensor.hpp>

namespace nn::global {

/**
 * @struct Prediction
 * @brief Represents a prediction result from the neural network
 * 
 * This structure holds the result of a neural network prediction, including
 * both the predicted class index and the confidence value associated with
 * that prediction.
 * 
 * @var size_t index The index of the predicted class
 * @var ValueType value The confidence value or probability for this prediction
 */
struct Prediction {
	size_t index;                    ///< The predicted class index (0-based)
	global::ValueType value;         ///< The confidence value for this prediction
	
	/**
	 * @brief Default constructor - creates a prediction with zero values
	 */
	Prediction() : index(0), value(0) {}
	
	/**
	 * @brief Parameterized constructor - creates a prediction with specified values
	 * @param index_ The predicted class index
	 * @param value_ The confidence value for the prediction
	 */
	Prediction(const size_t index_, const global::ValueType value_)
	    : index(index_),
	      value(value_) {}
};

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
