#include "fonts.hpp"
#include <string>

using namespace std;

namespace Visualizer {

sf::Font &Fonts::getFont() {
	static sf::Font font;
	static string path = string(RESOURCE_DIR) + "/Inter.ttc";
	font.loadFromFile(path);
	return font;
}
} // namespace Visualizer
