#ifndef VISUALIZERRENDERER
#define VISUALIZERRENDERER

#include "../model/neuralNetwork.hpp"
#include "Globals.hpp"
#include "VInterface.hpp"
#include "Vstatus.hpp"
#include "graph.hpp"
#include "state.hpp"
#include "visualNN.hpp"

namespace nn::visualizer {
constexpr sf::Color BG_COLOR(100, 100, 100);
constexpr std::uint32_t UI_GAP = 15;

constexpr std::uint32_t WINDOW_WIDTH = 1600;
constexpr std::uint32_t WINDOW_HEIGHT = 800;
constexpr std::string_view WINDOW_TITLE = "Visualizer";

class VisualizerRenderer {
  private:
	sf::RenderWindow window;
	visualNN visualNetwork;
	std::shared_ptr<StateManager> Vstate;
	vInteface interface;
	vStatus statusV;
	GraphUI Vgraph;
	std::atomic<bool> running{false};
	float fps;
	float bps;
	bool need_resize{false};

	void renderLoop();
	void processEvents();
	void renderPanels();
	void clear();
	void full_update();
	void do_frame(int &frameCount, int &batchCount, sf::Clock &fpsClock);
	void reset_size();

  public:
	VisualizerRenderer(const model::NeuralNetwork &network, const std::shared_ptr<StateManager> vstate);
	~VisualizerRenderer();
	void close();
	void updateDots(const int layer, const model::Neurons &newNeurons);
	bool updateStatus();
	void update(const training::gradient &new_grad);
	void start();
	void update(const int layer, const model::LayerParameters &gradients);
	void updateBatchCounter(const global::ValueType error, const int index);
	void setNewPhaseMode(const NnMode nn_mode);
	void update_prediction(const int index);
	void update_lr(const global::ValueType newLerningRate);
};
} // namespace nn::visualizer

#endif // VISUALIZERRENDERER
