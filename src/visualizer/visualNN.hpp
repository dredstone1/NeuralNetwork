#ifndef VISUALNN
#define VISUALNN

#include "../model/neuralNetwork.hpp"
#include "../trainer/gradient.hpp"
#include "visualL.hpp"

namespace nn::visualizer {
constexpr sf::Color NN_PANEL_BG = PANELS_BG;

class NNPanel : public Panel {
  private:
	std::vector<std::unique_ptr<visualLayer>> layers;
	sf::RenderTexture NNRender;
	int current_rendred_layer;

	void clear();
	void display();
	void renderLayers();
	void renderLayer(const int layer, const float posx);
	void doRender() override;
	void render_active_layer(const sf::Vector2f box, const sf::Vector2f pos);

  public:
	NNPanel(const model::NeuralNetwork &network, const std::shared_ptr<StateManager> state_);
	~NNPanel() = default;

	sf::Sprite getSprite();

	void updateDots(const int layer, const model::Neurons &newNeurons);
	void update(const int layer, const model::LayerParameters &gradients);
	void update(const training::gradient &new_grad);
	void update_prediction(const int index);
};
} // namespace nn::visualizer

#endif // VISUALNN
