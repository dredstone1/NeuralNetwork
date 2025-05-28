#include "fonts.hpp"
#include <string>

namespace Visualizer {

sf::Font &Fonts::getFont() {
	static sf::Font font;
	static std::string path = std::string(RESOURCE_DIR) + "/Inter.ttc";
	font.loadFromFile(path);
	return font;
}
} // namespace Visualizer
