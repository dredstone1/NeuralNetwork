#include "VisualizerRenderer.hpp"
#include "state.hpp"
#include "visualL.hpp"
#include "visualNN.hpp"
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window/Event.hpp>
#include <cstdio>

namespace Visualizer {
VisualizerRenderer::VisualizerRenderer(const neural_network &network, state *vstate) : window(sf::VideoMode(1600, 800), "Visualizer", sf::Style::Titlebar | sf::Style::Titlebar), visualNetwork(network), Vstate(vstate), interface(vstate) {}

void VisualizerRenderer::processEvents() {
	sf::Event event;
	while (window.pollEvent(event)) {
		if (event.type == sf::Event::Closed) {
			close();
		}
		if (event.type == sf::Event::Resized) {
			needUpdate = true;
		}
		if (event.type == sf::Event::MouseButtonPressed) {
			sf::Vector2i mousePos = sf::Mouse::getPosition(window);
			interface.handleClick(mousePos, {NN_WIDTH + 50 + 15, 50});
			needUpdate = true;
		} else if (event.type == sf::Event::MouseButtonReleased) {
			interface.handleNoClick();
			needUpdate = true;
		}
	}
}

void VisualizerRenderer::renderObjects() {
	visualNetwork.render(updatedLayersDot, updatedLayersWeight);

	sf::Sprite visualNetworkSprite = visualNetwork.getSprite();
	visualNetworkSprite.setPosition(50, 50);
	window.draw(visualNetworkSprite);

	interface.renderInterface();
	sf::Sprite interfaceSprite = interface.getSprite();
	interfaceSprite.setPosition(visualNetworkSprite.getGlobalBounds().getSize().x + visualNetworkSprite.getGlobalBounds().getPosition().x + 15, 50);
	window.draw(interfaceSprite);
}

void VisualizerRenderer::update() {
	window.clear(sf::Color::White);

	renderObjects();

	window.display();
}

void VisualizerRenderer::renderLoop() {
	running = true;
	while (window.isOpen() && running) {
		processEvents();

		if (needUpdate) {
			update();

			needUpdate = false;
			updatedLayersWeight = 0;
			updatedLayersDot = 0;
		}
	}

	window.close();
}

void VisualizerRenderer::close() {
	running = false;
}

void VisualizerRenderer::start() {
	running = true;
	renderLoop();
}

void VisualizerRenderer::setUpdate(int Dot, int Weight) {
	if (Weight != -1)
		updatedLayersWeight |= (1 << Weight);

	if (Dot != -1)
		updatedLayersDot |= (1 << Dot);
}

void VisualizerRenderer::updateDots(const int layer, vector<double> out, vector<double> net) {
	visualNetwork.updateDots(layer, out, net);
	needUpdate = true;
	setUpdate(layer, -1);
}

void VisualizerRenderer::update(const int layer, const LayerParameters &gradients) {
	visualNetwork.update(layer, gradients);
	needUpdate = true;
}

VisualizerRenderer::~VisualizerRenderer() {
	close();
}
} // namespace Visualizer
