#include "visualNN.hpp"
#include "visualL.hpp"
#include <memory>

namespace Visualizer {
visualNN::visualNN(const neural_network &network, std::shared_ptr<state> state_)
    : panel(state_),
      NNRender({NN_WIDTH, NN_HEIGHT}),
      current_rendred_layer(0) {
	layers.reserve(network.getLayerCount() + 1);

	layers.emplace_back(new VEmptyLayer(vstate->config.network_config.input_size, network.getLayerCount() + 1, vstate));
	for (int layer = 0; layer < network.getLayerCount(); layer++) {
		layers.emplace_back(new VParamLayer(*network.layers.at(layer), vstate));
	}
}

void visualNN::display() {
	NNRender.display();
}

void visualNN::clear() {
	NNRender.clear(NN_PANEL_BG);
}

void visualNN::renderLayers() {
	float posx = 0;
	for (int layer = 0; layer < vstate->config.network_config.hidden_layer_count() + 2; layer++) {
		renderLayer(layer, posx);
		posx += layers[layer]->WIDTH;
	}
}

void visualNN::do_render() {
	clear();
	renderLayers();
    display();
}

void visualNN::render_active_layer(const sf::Vector2f box, const sf::Vector2f pos) {
	sf::RectangleShape shape(box);
	shape.setPosition(pos);
	shape.setFillColor(ACTIVE_BG_LAYER);

	NNRender.draw(shape);
}

void visualNN::renderLayer(const int layer, const float posx) {
	layers[layer]->render();

	sf::Sprite newSprite = layers[layer]->getSprite();
	if (layer == current_rendred_layer)
		render_active_layer(newSprite.getLocalBounds().size, {posx, 0});
	newSprite.setPosition({posx, 0});

	NNRender.draw(newSprite);
}

sf::Sprite visualNN::getSprite() {
	return sf::Sprite(NNRender.getTexture());
}

void visualNN::updateDots(const int layer, const std::vector<double> &out, const std::vector<double> &net) {
	current_rendred_layer = layer;
	layers[layer]->setDots(out, net);

	set_update();
}

void visualNN::update(const int layer, const LayerParameters &gradients) {
	current_rendred_layer = layer;
	layers[layer]->set_weights(gradients);

	set_update();
}

void visualNN::update(const gradient &new_grad) {
	for (size_t i = 1; i < layers.size(); i++) {
		((VParamLayer *)layers[i])->updateGrad(new_grad.gradients[i - 1]);
	}

	set_update();
}

visualNN::~visualNN() {
	for (size_t i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}
} // namespace Visualizer
