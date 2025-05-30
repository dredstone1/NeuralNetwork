#ifndef FONTS
#define FONTS

#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>

namespace Visualizer {
class Fonts {
  public:
	Fonts() = delete;
	static sf::Font &getFont();
	~Fonts() = default;
};
} // namespace Visualizer

#endif // FONTS
