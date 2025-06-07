#include "visualNN.hpp"
#include "Globals.hpp"
#include "visualL.hpp"
#include <memory>
#include <vector>

namespace nn {
namespace Visualizer {
visualNN::visualNN(const neural_network &network, std::shared_ptr<state> state_)
    : panel(state_),
      NNRender({NN_WIDTH, NN_HEIGHT}),
      current_rendred_layer(0) {
	layers.reserve(network.getLayerCount() + 2);
	size_t layer = 0;

	layers.emplace_back(std::make_unique<VEmptyLayer>(vstate->config.network_config.input_size, vstate));

	for (; layer < network.getLayerCount() - 1; layer++) {
		layers.emplace_back(std::make_unique<VParamLayer>(vstate->config.network_config.layers_config[layer].size, layers[layer]->getSize(), vstate));
		layers[layers.size() - 1]->reset();
		layers[layers.size() - 1]->add(network.layers[layer]->getParms());
	}

	layers.emplace_back(std::make_unique<VParamLayer>(vstate->config.network_config.output_size, layers[layer]->getSize(), vstate));
	layers[layers.size() - 1]->reset();
	layers[layers.size() - 1]->add(network.layers[layer]->getParms());
	layers.emplace_back(std::make_unique<VEmptyLayer>(vstate->config.network_config.output_size, vstate));
}

void visualNN::display() {
	NNRender.display();
}

void visualNN::clear() {
	NNRender.clear(NN_PANEL_BG);
}

void visualNN::renderLayers() {
	float posx = 0;
	for (size_t layer = 0; layer < vstate->config.network_config.hidden_layer_count() + 3; layer++) {
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
	if (!vstate->settings.preciseMode)
		return;

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

void visualNN::updateDots(const int layer, const std::vector<Global::ValueType> &out, const std::vector<Global::ValueType> &net) {
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
	for (size_t i = 1; i < layers.size() - 1; i++) {
		VParamLayer *test = dynamic_cast<VParamLayer *>(layers[i].get());
		test->updateGrad(new_grad.gradients[i - 1]);
	}

	set_update();
}

void visualNN::update_prediction(const int index) {
	std::vector<Global::ValueType> pre(layers[layers.size() - 1]->getSize(), 0);
	pre[index] = 1;

	layers[layers.size() - 1]->setDots(pre, pre);

	set_update();
}
} // namespace Visualizer
} // namespace nn
