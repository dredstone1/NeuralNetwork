#include "VisualizerRenderer.hpp"
#include "model/neuralNetwork.hpp"
#include "state.hpp"
#include "trainer/gradient.hpp"
#include "visualL.hpp"
#include "visualNN.hpp"
#include "visualizer/Vstatus.hpp"
#include <SFML/Graphics/Sprite.hpp>
#include <SFML/Window/Event.hpp>
#include <cstdio>
#include <memory>

namespace Visualizer {
VisualizerRenderer::VisualizerRenderer(const neural_network &network, std::shared_ptr<state> vstate)
    : window(sf::VideoMode(1600, 800), "Visualizer", sf::Style::Titlebar | sf::Style::Titlebar),
      visualNetwork(network, vstate),
      Vstate(vstate),
      interface(vstate),
      statusV(vstate) {}

void VisualizerRenderer::processEvents() {
	sf::Event event;
	while (window.pollEvent(event)) {
		if (event.type == sf::Event::Closed) {
			close();
		}
		if (event.type == sf::Event::Resized) {
			needUpdate.store(true);
		}
		if (event.type == sf::Event::MouseButtonPressed) {
			sf::Vector2i mousePos = sf::Mouse::getPosition(window);
			interface.handleClick(mousePos, {NN_WIDTH + UI_GAP + UI_GAP, UI_GAP});
			needUpdate.store(true);
		} else if (event.type == sf::Event::MouseButtonReleased) {
			interface.handleNoClick();
			needUpdate.store(true);
		}
	}
}

void VisualizerRenderer::renderObjects() {
	visualNetwork.render();

	sf::Sprite visualNetworkSprite = visualNetwork.getSprite();
	visualNetworkSprite.setPosition(UI_GAP, UI_GAP);
	window.draw(visualNetworkSprite);

	interface.render();
	sf::Sprite interfaceSprite = interface.getSprite();
	interfaceSprite.setPosition(visualNetworkSprite.getGlobalBounds().getSize().x + visualNetworkSprite.getGlobalBounds().getPosition().x + UI_GAP, UI_GAP);
	window.draw(interfaceSprite);

	statusV.render();
	sf::Sprite statusSprite = statusV.getSprite();
	statusSprite.setPosition(visualNetworkSprite.getGlobalBounds().getSize().x + visualNetworkSprite.getGlobalBounds().getPosition().x + UI_GAP, UI_GAP + interfaceSprite.getGlobalBounds().getSize().y + interfaceSprite.getGlobalBounds().getPosition().y);
	window.draw(statusSprite);
}

void VisualizerRenderer::render_frame() {
	// window.clear(sf::Color::White);

	renderObjects();

	window.display();
}

void VisualizerRenderer::renderLoop() {
	running.store(true);
	sf::Clock fpsClock;
	int frameCount = 0;

	while (window.isOpen() && running) {
		processEvents();
		frameCount++;

		if (fpsClock.getElapsedTime().asSeconds() >= 1.0f) {
			fps = frameCount / fpsClock.getElapsedTime().asSeconds();

			fpsClock.restart();
			frameCount = 0;
			statusV.update_fps(fps);
			needUpdate = true;
		}

		render_frame();

		window.display();
		needUpdate.store(false);
	}

	window.close();
}

void VisualizerRenderer::close() {
	running.store(false);
}

bool VisualizerRenderer::updateStatus() {
	return interface.updateStatus() || statusV.updateStatus();
}

void VisualizerRenderer::start() {
	running.store(true);
	renderLoop();
}

void VisualizerRenderer::updateDots(const int layer, const std::vector<double> out, const std::vector<double> net) {
	visualNetwork.updateDots(layer, out, net);
	needUpdate.store(true);
}

void VisualizerRenderer::update(const int layer, const LayerParameters &gradients) {
	visualNetwork.update(layer, gradients);
	needUpdate.store(true);
}

void VisualizerRenderer::update(const gradient new_grad) {
	visualNetwork.update(new_grad);
	needUpdate.store(true);
}

VisualizerRenderer::~VisualizerRenderer() {
	close();
}

void VisualizerRenderer::setNewPhaseMode(const NNmode nn_mode) {
	Vstate->nnMode.store(nn_mode);
	needUpdate.store(true);
}
} // namespace Visualizer
