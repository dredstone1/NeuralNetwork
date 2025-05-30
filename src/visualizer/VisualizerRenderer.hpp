#ifndef VISUALIZERRENDERER
#define VISUALIZERRENDERER

#include "../model/neuralNetwork.hpp"
#include "VInterface.hpp"
#include "Vstatus.hpp"
#include "state.hpp"
#include "visualNN.hpp"
#include <SFML/Graphics.hpp>
#include <atomic>
#include <climits>

namespace Visualizer {
#define UI_GAP 15

class VisualizerRenderer {
  private:
	sf::RenderWindow window;
	visualNN visualNetwork;
	state &Vstate;
	vInteface interface;
	vStatus statusV;
	std::atomic<int> needUpdate{true};
	std::atomic<bool> running{false};
	void update();
	void renderLoop();
	void processEvents();
	void renderObjects();

  public:
	VisualizerRenderer(const neural_network &network, state &vstate);
	~VisualizerRenderer();
	void close();
	void updateDots(const int layer, const std::vector<double> out, const std::vector<double> net);
	bool updateStatus() { return needUpdate; }
	void update(const gradient new_grad);
	void start();
	void update(const int layer, const LayerParameters &gradients);
	void setNewPhaseMode(const NNmode nn_mode);
};
} // namespace Visualizer

#endif // VISUALIZERRENDERER
