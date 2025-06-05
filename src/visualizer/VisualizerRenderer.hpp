#ifndef VISUALIZERRENDERER
#define VISUALIZERRENDERER

#include "../model/neuralNetwork.hpp"
#include "VInterface.hpp"
#include "Vstatus.hpp"
#include "graph.hpp"
#include "state.hpp"
#include "visualNN.hpp"
#include <SFML/Graphics.hpp>

namespace Visualizer {
constexpr sf::Color BG_COLOR(100, 100, 100);
constexpr std::uint32_t UI_GAP = 15;

constexpr std::uint32_t WINDOW_WIDTH = 1600;
constexpr std::uint32_t WINDOW_HEIGHT = 800;
constexpr std::string_view WINDOW_TITLE = "Visualizer";

class VisualizerRenderer {
  private:
	sf::RenderWindow window;
	visualNN visualNetwork;
	std::shared_ptr<state> Vstate;
	vInteface interface;
	vStatus statusV;
	GraphUI Vgraph;
	std::atomic<bool> running{false};
	float fps;
	void renderLoop();
	void processEvents();
	void renderPanels();
	void clear();
	void full_update();
	void do_frame(int &frameCount, sf::Clock &fpsClock);

  public:
	VisualizerRenderer(const neural_network &network, const std::shared_ptr<state> vstate);
	~VisualizerRenderer();
	void close();
	void updateDots(const int layer, const std::vector<double> &out, const std::vector<double> &net);
	bool updateStatus();
	void update(const gradient &new_grad);
	void start();
	void update(const int layer, const LayerParameters &gradients);
	void updateBatchCounter(const double error, const int index);
	void setNewPhaseMode(const NNmode nn_mode);
};
} // namespace Visualizer

#endif // VISUALIZERRENDERER
