/**
 * @file fonts.hpp
 * @brief Font management interface for the neural network visualizer
 * 
 * This file defines the Fonts utility class that provides centralized
 * font loading and access for all text rendering in the visualization
 * system. It implements a singleton pattern to ensure efficient resource
 * management and consistent font usage across UI components.
 */

#ifndef FONTS
#define FONTS

#include <SFML/Graphics.hpp>

namespace nn::visualizer {

/**
 * @class Fonts
 * @brief Static utility class for font management
 * 
 * The Fonts class provides a centralized interface for loading and accessing
 * fonts used in the neural network visualizer. It implements a singleton
 * pattern to ensure that fonts are loaded only once and shared across all
 * UI components that need text rendering capabilities.
 * 
 * Key features:
 * - Singleton font loading to avoid resource duplication
 * - Centralized font management for consistent UI appearance
 * - Error handling for missing font files
 * - Static interface for easy access from any UI component
 * 
 * @note This class cannot be instantiated (constructor is deleted)
 * @note All methods are static and thread-safe
 */
class Fonts {
  public:
	/** @brief Deleted constructor - this is a static utility class */
	Fonts() = delete;
	
	/**
	 * @brief Gets the default font for the visualizer
	 * 
	 * Returns a reference to the default font used throughout the
	 * visualization interface. The font is loaded on first access
	 * and cached for subsequent calls.
	 * 
	 * @return Reference to the default SFML font object
	 * @note The font is loaded from the RESOURCE_DIR/Inter.ttc file
	 */
	static sf::Font &getFont();
	
	/** @brief Default destructor */
	~Fonts() = default;
};

} // namespace nn::visualizer

#endif // FONTS
