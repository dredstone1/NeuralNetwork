#ifndef FONTS
#define FONTS

#include <SFML/Graphics.hpp>

namespace nn {
namespace Visualizer {
class Fonts {
  public:
	Fonts() = delete;
	static sf::Font &getFont();
	~Fonts() = default;
};
} // namespace Visualizer
} // namespace nn

#endif // FONTS
