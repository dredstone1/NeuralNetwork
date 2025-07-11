#include "VisualizerRenderer.hpp"
#include "Globals.hpp"

namespace nn::visualizer {
constexpr std::uint32_t NN_WIDTH = 1055u;
VisualRender::VisualRender(std::shared_ptr<StateManager> vstate)
    : window(sf::VideoMode({WINDOW_WIDTH, WINDOW_HEIGHT}), WINDOW_TITLE.data()),
      visualModel(vstate),
      Vstate(vstate),
      interface(vstate),
      statusV(vstate),
      Vgraph(vstate) {}

void VisualRender::processEvents() {
	while (const std::optional event = window.pollEvent()) {
		if (event->is<sf::Event::Closed>()) {
			window.close();
		} else if (event->is<sf::Event::MouseButtonPressed>()) {
			interface.handleNoClick();
		} else if (event->is<sf::Event::MouseButtonReleased>()) {
			interface.handleClick(sf::Mouse::getPosition(window), {NN_WIDTH + UI_GAP * 2, UI_GAP});
		} else if (event->is<sf::Event::Resized>()) {
			need_resize = true;
		}
	}
}

void VisualRender::resetSize() {
	if (need_resize) {
		window.setSize({WINDOW_WIDTH, WINDOW_HEIGHT});
	}

	need_resize = false;
}

void VisualRender::renderPanels() {
	visualModel.render();
	sf::Sprite visualNetworkSprite = visualModel.getSprite();
	visualNetworkSprite.setPosition({UI_GAP, UI_GAP});
	window.draw(visualNetworkSprite);

	interface.render();
	sf::Sprite interfaceSprite = interface.getSprite();
	interfaceSprite.setPosition({NN_WIDTH + UI_GAP * 2, UI_GAP});
	window.draw(interfaceSprite);

	statusV.render();
	sf::Sprite statusSprite = statusV.getSprite();
	statusSprite.setPosition({NN_WIDTH + UI_GAP * 2, UI_GAP * 2 + VINTERFACE_HEIGHT});
	window.draw(statusSprite);

	Vgraph.render();
	sf::Sprite graphSprite = Vgraph.getSprite();
	graphSprite.setPosition({NN_WIDTH + UI_GAP * 2, UI_GAP * 3 + VINTERFACE_HEIGHT + VSTATUS_HEIGHT});
	window.draw(graphSprite);
}

void VisualRender::fullUpdate() {
	resetSize();

	statusV.setUpdate();
	interface.setUpdate();
	visualModel.setUpdate();
	Vgraph.setUpdate();
}

void VisualRender::doFrame(int &frameCount, int &batchCount, sf::Clock &fpsClock) {
	processEvents();

	if (fpsClock.getElapsedTime().asSeconds() >= 1.0f) {
		const float timeOffset = fpsClock.getElapsedTime().asSeconds();
		fps = frameCount / timeOffset;
		bps = (Vstate->currentBatch - batchCount) / timeOffset;

		fpsClock.restart();
		frameCount = 0;
		batchCount = Vstate->currentBatch;
		statusV.updateFps(fps);
		statusV.updateBps(bps);
		fullUpdate();
	}

	if (updateStatus()) {
		clear();
		frameCount++;

		renderPanels();
		window.display();
	}
}

void VisualRender::clear() {
	window.clear(BG_COLOR);
}

void VisualRender::renderLoop() {
	running.store(true);
	sf::Clock fpsClock;
	int frameCount = 0, batchCount = 0;

	window.setFramerateLimit(FPS_LIMIT);

	clear();
	while (window.isOpen() && running) {
		doFrame(frameCount, batchCount, fpsClock);
	}

	window.close();
}

void VisualRender::close() {
	running.store(false);
}

bool VisualRender::updateStatus() {
	return interface.updateStatus() || statusV.updateStatus() || visualModel.updateStatus();
}

void VisualRender::start() {
	running.store(true);
	renderLoop();
}

void VisualRender::updateBatchCounter(const global::ValueType error, const int index) {
	Vgraph.addData(error, index);
}

VisualRender::~VisualRender() {
	close();
}

void VisualRender::setNewPhaseMode(const NnMode nn_mode) {
	statusV.setUpdate();
	Vstate->nnMode.store(nn_mode);
}

void VisualRender::updatePrediction(const global::Prediction &pre) {
	visualModel.setPrediction(pre);
}

void VisualRender::updateInput(const global::ParamMetrix &input) {
	visualModel.setInput(input);
}

void VisualRender::updateLearningRate(const global::ValueType lr) {
	statusV.updateLerningRate(lr);
}

void VisualRender::addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual) {
	visualModel.addVisualSubNetwork(newVisual);
}
} // namespace nn::visualizer
