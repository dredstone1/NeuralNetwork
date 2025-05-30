#include "visualNN.hpp"
#include "model/LayerParameters.hpp"
#include "model/neuralNetwork.hpp"
#include "trainer/gradient.hpp"
#include "visualL.hpp"
#include "visualizer/state.hpp"
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <bits/types/locale_t.h>
#include <cstddef>
#include <cstdio>

namespace Visualizer {
visualNN::visualNN(const neural_network &network, state &state_) : config(network.config), current_rendred_layer(0), vstate(state_) {
	layers.reserve(network.getLayerCount() + 1);

	layers.emplace_back(new VEmptyLayer(config.input_size, 0, network.getLayerCount() + 1));
	for (int layer = 0; layer < network.getLayerCount(); layer++) {
		layers.emplace_back(new VParamLayer(*network.layers.at(layer), network.getLayerCount() + 1));
	}

	createNnVisual();
}

void visualNN::createNnVisual() {
	NNRender.create(NN_WIDTH, NN_HEIGHT);
	clear();
}

void visualNN::display() {
	NNRender.display();
}

void visualNN::clear() {
	NNRender.clear(sf::Color::Green);
}

void visualNN::renderLayers() {
	float posx = 0;
	for (int layer = 0; layer < config.hidden_layer_count() + 2; layer++) {
		renderLayer(layer, posx);
		posx += layers[layer]->WIDTH;
	}
}

void visualNN::render() {
	renderLayers();
}

void visualNN::renderLayer(const int layer, const float posx) {
	layers[layer]->renderLayer(layer == current_rendred_layer);

	sf::Sprite newSprite = layers[layer]->getSprite();
	newSprite.setPosition(posx, 0);

	NNRender.draw(newSprite);
}

sf::Sprite visualNN::getSprite() {
	display();
	return sf::Sprite(NNRender.getTexture());
}

void visualNN::updateDots(const int layer, std::vector<double> out, std::vector<double> net) {
	current_rendred_layer = layer;
	layers[layer]->setDots(out, net);
}

void visualNN::update(const int layer, const LayerParameters &gradients) {
	current_rendred_layer = layer;
	layers[layer]->add(gradients);
}

void visualNN::update(const gradient new_grad) {
	for (size_t i = 1; i < layers.size(); i++) {
		((VParamLayer *)layers[i])->updateGrad(new_grad.gradients[i - 1]);
	}
}

visualNN::~visualNN() {
	for (size_t i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}
} // namespace Visualizer
