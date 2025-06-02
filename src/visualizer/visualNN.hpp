#ifndef VISUALNN
#define VISUALNN

#include "../model/LayerParameters.hpp"
#include "../model/neuralNetwork.hpp"
#include "../trainer/gradient.hpp"
#include "panel.hpp"
#include "state.hpp"
#include "visualL.hpp"
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <memory>
#include <vector>

namespace Visualizer {
class visualNN : public panel {
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
    void do_render() override;

  public:
	visualNN(const neural_network &network, std::shared_ptr<state> state_);
	~visualNN();
	sf::Sprite getSprite();
	void updateDots(const int layer, const std::vector<double> out, const std::vector<double> net);
	void update(const int layer, const LayerParameters &gradients);
	void update(const gradient new_grad);
};
} // namespace Visualizer

#endif // VISUALNN
