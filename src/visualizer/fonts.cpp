#include "fonts.hpp"
#include <string>

namespace Visualizer {
sf::Font &Fonts::getFont() {
	static sf::Font font;
	static bool loaded = false;
	if (!loaded) {
		std::string path = std::string(RESOURCE_DIR) + "/Inter.ttc";
		font.loadFromFile(path);
		loaded = true;
	}
	return font;
}
} // namespace Visualizer
