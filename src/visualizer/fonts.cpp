/**
 * @file fonts.cpp
 * @brief Implementation of font management for the neural network visualizer
 * 
 * This file implements font loading and management functionality for the
 * visualization system. It provides a centralized way to access fonts
 * for rendering text in various UI components.
 */

#include "fonts.hpp"

namespace nn::visualizer {

/**
 * @brief Gets the default font for text rendering in the visualizer
 * 
 * This function implements a singleton pattern to load and return the
 * default font used throughout the visualization interface. The font
 * is loaded once from the resources directory and cached for subsequent
 * use across all UI components.
 * 
 * @return Reference to the loaded font object
 * @note If the font file cannot be found, an error message is printed
 *       and the function returns an empty font object
 */
sf::Font &Fonts::getFont() {
	static sf::Font font;     // Static font instance for singleton pattern
	static bool loaded = false; // Flag to ensure font is loaded only once

	if (!loaded) {
		// Construct path to font file in resources directory
		std::string path = std::string(RESOURCE_DIR) + "/Inter.ttc";

		// Attempt to load the font file
		if (!font.openFromFile(path)) {
			printf("Font not found: %s\n", path.c_str());
		}

		loaded = true;
	}
	return font;
}

} // namespace nn::visualizer
