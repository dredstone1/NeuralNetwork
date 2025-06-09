#ifndef FONTS
#define FONTS

#include <SFML/Graphics.hpp>

namespace nn::visualizer {
class Fonts {
  public:
	Fonts() = delete;
	static sf::Font &getFont();
	~Fonts() = default;
};
} // namespace nn::visualizer

#endif // FONTS
