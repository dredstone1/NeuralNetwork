#include "VisualizerRenderer.hpp"
#include "model/neuralNetwork.hpp"
#include "state.hpp"
#include "trainer/gradient.hpp"
#include "visualL.hpp"
#include "visualNN.hpp"
#include "visualizer/VInterface.hpp"
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
		if (event.type == sf::Event::MouseButtonPressed) {
			sf::Vector2i mousePos = sf::Mouse::getPosition(window);
			interface.handleClick(mousePos, {NN_WIDTH + UI_GAP + UI_GAP, UI_GAP});
		} else if (event.type == sf::Event::MouseButtonReleased) {
			interface.handleNoClick();
		}
	}
}

void VisualizerRenderer::renderObjects() {
	if (visualNetwork.render()) {
		sf::Sprite visualNetworkSprite = visualNetwork.getSprite();
		visualNetworkSprite.setPosition(UI_GAP, UI_GAP);
		window.draw(visualNetworkSprite);
	}

	if (interface.render()) {
		sf::Sprite interfaceSprite = interface.getSprite();
		interfaceSprite.setPosition(NN_WIDTH + UI_GAP + UI_GAP, UI_GAP);
		window.draw(interfaceSprite);
	}

	if (statusV.render()) {
		sf::Sprite statusSprite = statusV.getSprite();
		statusSprite.setPosition(NN_WIDTH + UI_GAP + UI_GAP, UI_GAP + UI_GAP + VINTERFACE_HEIGHT);
		window.draw(statusSprite);
	}
}

void VisualizerRenderer::full_update() {
	clear();
	statusV.set_update();
	interface.set_update();
	visualNetwork.set_update();
}

void VisualizerRenderer::do_frame(int &frameCount, sf::Clock &fpsClock) {
	processEvents();

	if (fpsClock.getElapsedTime().asSeconds() >= 1.0f) {
		fps = frameCount / fpsClock.getElapsedTime().asSeconds();

		fpsClock.restart();
		frameCount = 0;
		statusV.update_fps(fps);

		full_update();
	}

	if (updateStatus()) {
		frameCount++;

		renderObjects();
		window.display();
	}
}

void VisualizerRenderer::clear() {
	window.clear(sf::Color(100, 100, 100));
}

void VisualizerRenderer::renderLoop() {
	running.store(true);
	sf::Clock fpsClock;
	int frameCount = 0;

	window.setFramerateLimit(FPS_LIMIT);
	clear();

	while (window.isOpen() && running) {
		do_frame(frameCount, fpsClock);
	}

	window.close();
}

void VisualizerRenderer::close() {
	running.store(false);
}

bool VisualizerRenderer::updateStatus() {
	return interface.updateStatus() || statusV.updateStatus() || visualNetwork.updateStatus();
}

void VisualizerRenderer::start() {
	running.store(true);
	renderLoop();
}

void VisualizerRenderer::updateDots(const int layer, const std::vector<double> out, const std::vector<double> net) {
	visualNetwork.updateDots(layer, out, net);
}

void VisualizerRenderer::update(const int layer, const LayerParameters &gradients) {
	visualNetwork.update(layer, gradients);
}

void VisualizerRenderer::update(const gradient new_grad) {
	visualNetwork.update(new_grad);
}

VisualizerRenderer::~VisualizerRenderer() {
	close();
}

void VisualizerRenderer::setNewPhaseMode(const NNmode nn_mode) {
	statusV.set_update();
	Vstate->nnMode.store(nn_mode);
}
} // namespace Visualizer
