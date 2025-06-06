#ifndef VISUALNN
#define VISUALNN

#include "../model/neuralNetwork.hpp"
#include "../trainer/gradient.hpp"
#include "visualL.hpp"
#include <SFML/Graphics.hpp>
#include <memory>

namespace nn {
namespace Visualizer {
constexpr sf::Color NN_PANEL_BG = PANELS_BG;

class visualNN : public panel {
  private:
	std::vector<std::unique_ptr<visualL>> layers;
	sf::RenderTexture NNRender;
	int current_rendred_layer;
	void clear();
	void display();
	void renderLayers();
	void renderLayer(const int layer, const float posx);
	static bool getBit(const long num, const int index);
	void do_render() override;
	void render_active_layer(const sf::Vector2f box, const sf::Vector2f pos);

  public:
	visualNN(const neural_network &network, const std::shared_ptr<state> state_);
	~visualNN() = default;
	sf::Sprite getSprite();
	void updateDots(const int layer, const std::vector<Global::ValueType> &out, const std::vector<Global::ValueType> &net);
	void update(const int layer, const LayerParameters &gradients);
	void update(const gradient &new_grad);
	void update_prediction(const int index);
};
} // namespace Visualizer
} // namespace nn

#endif // VISUALNN
