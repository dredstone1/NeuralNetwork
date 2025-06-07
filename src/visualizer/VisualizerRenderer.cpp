#include "VisualizerRenderer.hpp"
#include "Globals.hpp"
#include "state.hpp"
#include "visualL.hpp"
#include "visualNN.hpp"
#include "visualizer/Vstatus.hpp"
#include <memory>

namespace nn {
namespace Visualizer {
VisualizerRenderer::VisualizerRenderer(const neural_network &network, std::shared_ptr<state> vstate)
    : window(sf::VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}), WINDOW_TITLE.data()),
      visualNetwork(network, vstate),
      Vstate(vstate),
      interface(vstate),
      statusV(vstate),
      Vgraph(vstate) {
}

void VisualizerRenderer::processEvents() {
	while (const std::optional event = window.pollEvent()) {
		if (event->is<sf::Event::Closed>()) {
			window.close();
		} else if (event->is<sf::Event::MouseButtonPressed>()) {
			interface.handleNoClick();
		} else if (event->is<sf::Event::MouseButtonReleased>()) {
			interface.handleClick(sf::Mouse::getPosition(window), {NN_WIDTH + UI_GAP + UI_GAP, UI_GAP});
		} else if (event->is<sf::Event::Resized>()) {
			need_resize = true;
		}
	}
}

void VisualizerRenderer::reset_size() {
	if (need_resize) {
		window.setSize({WINDOW_WIDTH, WINDOW_HEIGHT});
	}

	need_resize = false;
}

void VisualizerRenderer::renderPanels() {
	visualNetwork.render();
	sf::Sprite visualNetworkSprite = visualNetwork.getSprite();
	visualNetworkSprite.setPosition({UI_GAP, UI_GAP});
	window.draw(visualNetworkSprite);

	interface.render();
	sf::Sprite interfaceSprite = interface.getSprite();
	interfaceSprite.setPosition({NN_WIDTH + UI_GAP + UI_GAP, UI_GAP});
	window.draw(interfaceSprite);

	statusV.render();

	sf::Sprite statusSprite = statusV.getSprite();
	statusSprite.setPosition({NN_WIDTH + UI_GAP + UI_GAP, UI_GAP + UI_GAP + VINTERFACE_HEIGHT});
	window.draw(statusSprite);

	Vgraph.render();
	sf::Sprite graphSprite = Vgraph.getSprite();
	graphSprite.setPosition({NN_WIDTH + UI_GAP + UI_GAP, UI_GAP + UI_GAP + VINTERFACE_HEIGHT + VSTATUS_HEIGHT + UI_GAP});
	window.draw(graphSprite);
}

void VisualizerRenderer::full_update() {
	reset_size();
	statusV.set_update();
	interface.set_update();
	visualNetwork.set_update();
	Vgraph.set_update();
}

void VisualizerRenderer::do_frame(int &frameCount, int &batchCount, sf::Clock &fpsClock) {
	processEvents();

	if (fpsClock.getElapsedTime().asSeconds() >= 1.0f) {
		fps = frameCount / fpsClock.getElapsedTime().asSeconds();
		bps = (Vstate->current_batch - batchCount) / fpsClock.getElapsedTime().asSeconds();

		fpsClock.restart();
		frameCount = 0;
		batchCount = Vstate->current_batch;
		statusV.update_fps(fps);
		statusV.update_bps(bps);
		full_update();
	}

	if (updateStatus()) {
		clear();
		frameCount++;

		renderPanels();
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
	int batchCount = 0;

	window.setFramerateLimit(FPS_LIMIT);

	clear();
	while (window.isOpen() && running) {
		do_frame(frameCount, batchCount, fpsClock);
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

void VisualizerRenderer::updateDots(const int layer, const std::vector<Global::ValueType> &out, const std::vector<Global::ValueType> &net) {
	visualNetwork.updateDots(layer, out, net);
}

void VisualizerRenderer::update(const int layer, const LayerParameters &gradients) {
	visualNetwork.update(layer, gradients);
}

void VisualizerRenderer::updateBatchCounter(const Global::ValueType error, const int index) {
	Vgraph.add_data(error, index);
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

void VisualizerRenderer::update_prediction(const int index) {
	visualNetwork.update_prediction(index);
}
void VisualizerRenderer::update_lr(const Global::ValueType lr) {
	statusV.update_lr(lr);
}
} // namespace Visualizer
} // namespace nn
