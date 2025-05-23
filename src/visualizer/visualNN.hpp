#ifndef VISUALNN
#define VISUALNN

#include "../model/neuralNetwork.hpp"
#include "../trainer/gradient.hpp"
#include "visualL.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <vector>

using namespace std;

namespace Visualizer {
class visualNN {
  private:
	gradient LocalGradient;
	const int NnLength;
	vector<visualL *> layers;
	sf::RenderTexture NNRender;
	void createNnVisual();
	void clear();
	void display();
	void renderLayers(const long layersD, const long layersW);
	void renderLayer(const int layer, const float posx);
	static bool getBit(const long num, const int index);

  public:
	visualNN(const neural_network &network);
	~visualNN() = default;
	sf::Sprite getSprite();
	void render(const long layersD, const long layersW);
	void updateDots(const int layer, vector<double> out, vector<double> net);
	void update(const int layer, const LayerParameters &gradients);
};
} // namespace Visualizer

#endif // VISUALNN
