#ifndef FONTS
#define FONTS

#include <SFML/Graphics.hpp>

namespace Visualizer {
class Fonts {
  public:
	Fonts() = delete;
	static sf::Font &getFont();
	~Fonts() = default;
};
} // namespace Visualizer

#endif // FONTS
