#include "VisualizerRenderer.hpp"
#include "state.hpp"
#include "visualL.hpp"
#include "visualNN.hpp"
#include <cstdio>
#include <memory>

namespace Visualizer {
VisualizerRenderer::VisualizerRenderer(const neural_network &network, std::shared_ptr<state> vstate)
    : window(sf::VideoMode(1600, 800), "Visualizer", sf::Style::Titlebar | sf::Style::Titlebar),
      visualNetwork(network, vstate),
      Vstate(vstate),
      interface(vstate),
      statusV(vstate),
      Vgraph(vstate) {}

void VisualizerRenderer::processEvents() {
	sf::Event event;
	while (window.pollEvent(event)) {
		switch (event.type) {
		case sf::Event::Closed:
			close();
			break;
		case sf::Event::MouseButtonReleased:
			interface.handleNoClick();
			break;
		case sf::Event::MouseButtonPressed:
			interface.handleClick(sf::Mouse::getPosition(window), {NN_WIDTH + UI_GAP + UI_GAP, UI_GAP});
			break;
		default:
			break;
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

	if (Vgraph.render()) {
		sf::Sprite graphSprite = Vgraph.getSprite();
		graphSprite.setPosition(NN_WIDTH + UI_GAP + UI_GAP, UI_GAP + UI_GAP + VINTERFACE_HEIGHT + VSTATUS_HEIGHT + UI_GAP);
		window.draw(graphSprite);
	}
}

void VisualizerRenderer::full_update() {
	clear();
	statusV.set_update();
	interface.set_update();
	visualNetwork.set_update();
    Vgraph.set_update();
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
	window.clear(BG_COLOR);
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

void VisualizerRenderer::updateDots(const int layer, const std::vector<double> &out, const std::vector<double> &net) {
	visualNetwork.updateDots(layer, out, net);
}

void VisualizerRenderer::update(const int layer, const LayerParameters &gradients) {
	visualNetwork.update(layer, gradients);
}

void VisualizerRenderer::updateBatchCounter(const double error) {
    Vgraph.add_data(error);
}

void VisualizerRenderer::update(const gradient &new_grad) {
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
