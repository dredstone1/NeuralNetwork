#include "fonts.hpp"
#include <string>

namespace Visualizer {
sf::Font &Fonts::getFont() {
	static sf::Font font;
	static bool loaded = false;
	if (!loaded) {
		std::string path = std::string(RESOURCE_DIR) + "/Inter.ttc";
		if (!font.openFromFile(path)) {
			printf("Font not found: %s\n", path.c_str());
		}

		loaded = true;
	}
	return font;
}
} // namespace Visualizer
