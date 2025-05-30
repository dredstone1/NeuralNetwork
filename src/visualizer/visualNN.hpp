#ifndef VISUALNN
#define VISUALNN

#include "../model/LayerParameters.hpp"
#include "../model/neuralNetwork.hpp"
#include "../trainer/gradient.hpp"
#include "state.hpp"
#include "visualL.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <vector>

namespace Visualizer {
class visualNN {
  private:
	const NetworkConfig &config;
	std::vector<visualL *> layers;
	sf::RenderTexture NNRender;
	int current_rendred_layer;
	void createNnVisual();
	void clear();
	void display();
	void renderLayers();
	void renderLayer(const int layer, const float posx);
	static bool getBit(const long num, const int index);
	state &vstate;

  public:
	visualNN(const neural_network &network, state &state_);
	~visualNN();
	sf::Sprite getSprite();
	void render();
	void updateDots(const int layer, const std::vector<double> out, const std::vector<double> net);
	void update(const int layer, const LayerParameters &gradients);
	void update(const gradient new_grad);
};
} // namespace Visualizer

#endif // VISUALNN
