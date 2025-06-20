#include "visualNN.hpp"

namespace nn::visualizer {
NNPanel::NNPanel(const model::NeuralNetwork &network, std::shared_ptr<StateManager> state_)
    : Panel(state_),
      NNRender({NN_WIDTH, NN_HEIGHT}) {
	layers.reserve(network.getLayerCount() + 2);
	size_t layer = 0;

	layers.emplace_back(std::make_unique<visualEmptyLayer>(vstate->config.network_config.input_size, vstate));

	for (; layer < network.getLayerCount() - 1; layer++) {
		layers.emplace_back(std::make_unique<visualParamLayer>(vstate->config.network_config.layers_config[layer].size, layers[layer]->getSize(), vstate));
		layers[layers.size() - 1]->setParams(network.layers[layer]->getParms());
	}

	layers.emplace_back(std::make_unique<visualParamLayer>(vstate->config.network_config.output_size, layers[layer]->getSize(), vstate));
	layers[layers.size() - 1]->setParams(network.layers[layer]->getParms());

	layers.emplace_back(std::make_unique<visualEmptyLayer>(vstate->config.network_config.output_size, vstate));
}

void NNPanel::display() {
	NNRender.display();
}

void NNPanel::clear() {
	NNRender.clear(NN_PANEL_BG);
}

void NNPanel::renderLayers() {
	float posx = 0;

	for (size_t layer = 0; layer < vstate->config.network_config.hidden_layer_count() + 3; layer++) {
		renderLayer(layer, posx);
		posx += layers[layer]->getWidth();
	}
}

void NNPanel::doRender() {
	clear();
	renderLayers();
	display();
}

void NNPanel::render_active_layer(const sf::Vector2f box, const sf::Vector2f pos) {
	if (!vstate->settings.preciseMode)
		return;

	sf::RectangleShape shape(box);
	shape.setPosition(pos);
	shape.setFillColor(ACTIVE_BG_LAYER);

	NNRender.draw(shape);
}

void NNPanel::renderLayer(const int layer, const float posx) {
	layers[layer]->render();

	sf::Sprite newSprite = layers[layer]->getSprite();
	if (layer == current_rendred_layer)
		render_active_layer(newSprite.getLocalBounds().size, {posx, 0});
	newSprite.setPosition({posx, 0});

	NNRender.draw(newSprite);
}

sf::Sprite NNPanel::getSprite() {
	return sf::Sprite(NNRender.getTexture());
}

void NNPanel::updateDots(const int layer, const model::Neurons &newNeurons) {
	current_rendred_layer = layer;
	layers[layer]->setDots(newNeurons);

	setUpdate();
}

void NNPanel::update(const int layer, const model::LayerParameters &gradients) {
	current_rendred_layer = layer;
	layers[layer]->set_weights(gradients);

	setUpdate();
}

void NNPanel::update(const training::gradient &new_grad) {
	for (size_t i = 1; i < layers.size() - 1; i++) {
		visualParamLayer *test = dynamic_cast<visualParamLayer *>(layers[i].get());
		test->updateGrad(new_grad.gradients[i - 1]);
	}

	setUpdate();
}

void NNPanel::update_prediction(const int index) {
	global::ParamMetrix pre(layers[layers.size() - 1]->getSize(), 0);
	pre[index] = 1;

	layers[layers.size() - 1]->setDots({pre, pre});

	setUpdate();
}
} // namespace nn::visualizer
