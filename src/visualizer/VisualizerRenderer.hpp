#ifndef VISUALIZERRENDERER
#define VISUALIZERRENDERER

#include "VInterface.hpp"
#include "Vstatus.hpp"
#include "graph.hpp"
#include "visualModel.hpp"

namespace nn::visualizer {
constexpr sf::Color BG_COLOR(100, 100, 100);
constexpr std::uint32_t UI_GAP = 15;

constexpr std::uint32_t WINDOW_WIDTH = 1600;
constexpr std::uint32_t WINDOW_HEIGHT = 800;
constexpr std::string_view WINDOW_TITLE = "Visualizer";

class VisualRender {
  private:
	sf::RenderWindow window;
	ModelPanel visualModel;
	std::shared_ptr<StateManager> Vstate;
	IntefacePanel interface;
	StatusPanel statusV;
	GraphUIPanel Vgraph;
	std::atomic<bool> running{false};
	float fps;
	float bps;
	bool need_resize{false};

	void renderLoop();
	void processEvents();
	void renderPanels();
	void clear();
	void fullUpdate();
	void doFrame(int &frameCount, int &batchCount, sf::Clock &fpsClock);
	void resetSize();

  public:
	VisualRender(const std::shared_ptr<StateManager> vstate);
	~VisualRender();

	void close();
	void start();

	bool updateStatus();
	void updateBatchCounter(
	    const global::ValueType newDataEvaluate,
	    const global::ValueType newDataLost,
	    int index);

	void updateLearningRate(const global::ValueType newLerningRate);
	void setNewPhaseMode(const NnMode nn_mode);

	void updatePrediction(const global::Prediction &pre);
	void updateInput(const global::ParamMetrix &input);

	void addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual);
};
} // namespace nn::visualizer

#endif // VISUALIZERRENDERER
