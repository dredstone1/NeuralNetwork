#include "visualNN.hpp"
#include "visualL.hpp"
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/System/Vector2.hpp>
#include <bits/types/locale_t.h>
#include <cstdio>

namespace Visualizer {
visualNN::visualNN(const neural_network &network) : config(network.config), current_rendred_layer(0) {
	layers.reserve(network.getLayerCount() + 1);

	layers.push_back(new visualL(config.input_size, 0, network.getLayerCount() + 1));
	for (int layer = 0; layer < network.getLayerCount(); layer++) {
		layers.push_back(new visualL(*network.layers[layer], network.getLayerCount() + 1));
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

void visualNN::updateDots(const int layer, vector<double> out, vector<double> net) {
    current_rendred_layer = layer;
	layers[layer]->setDots(out, net);
}

void visualNN::update(const int layer, const LayerParameters &gradients) {
    current_rendred_layer = layer;
	layers[layer]->add(gradients);
}
} // namespace Visualizer
