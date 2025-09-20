/**
 * @file panel.cpp
 * @brief Implementation of the Panel base class for UI components
 * 
 * This file implements the core functionality of the Panel base class,
 * providing update management and rendering coordination for all UI
 * components in the neural network visualizer.
 */

#include "panel.hpp"

namespace nn::visualizer {

/**
 * @brief Renders the panel if an update is needed
 * 
 * This method implements the rendering logic by first calling observe() to
 * check for any state changes, then conditionally calling doRender() if
 * the panel needs updating. After rendering, the update flag is reset.
 * 
 * @return true if the panel was rendered, false if no update was needed
 */
int Panel::render() {
	// Check for state changes that might require an update
	observe();
	
	// Only render if an update is needed
	if (updateStatus()) {
		doRender();
		need_update = false;
		return true;
	}

	return false;
}

/**
 * @brief Marks the panel as needing an update
 * 
 * Sets the internal update flag to trigger a re-render on the next
 * call to render(). The wait parameter is currently unused but
 * provides extensibility for future timing control features.
 * 
 * @param wait Unused parameter for potential future timing control
 */
void Panel::setUpdate(const bool) {
	need_update = true;
}

} // namespace nn::visualizer
