#ifndef VISUALIZERRENDERER
#define VISUALIZERRENDERER

#include "VInterface.hpp"
#include "Vstatus.hpp"
#include "graph.hpp"
#include "tensor.hpp"
#include "visualModel.hpp"
#include <memory>

namespace nn::visualizer {

constexpr sf::Color BG_COLOR(100, 100, 100);
constexpr std::uint32_t UI_GAP = 15;

constexpr std::uint32_t WINDOW_WIDTH = 1600;
constexpr std::uint32_t WINDOW_HEIGHT = 800;
constexpr std::string_view WINDOW_TITLE = "Visualizer";

class VisualRender {
  private:
	std::shared_ptr<StateManager> Vstate;
	sf::Vector2u winSize;
	sf::RenderWindow window;
	std::unique_ptr<ModelPanel> visualModel;
	InterfacePanel interface;
	StatusPanel statusV;
	GraphUIPanel Vgraph;
	std::atomic<bool> running{false};
	float fps;
	float bps;
	bool need_resize{false};

	sf::Vector2u getWinSize(bool enableNetwork);

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
	void updateEvaluate(
	    const global::ValueType newDataEvaluate,
	    int index);

	void updateLoss(
	    const global::ValueType newDataLoss,
	    int index);

	void updateLearningRate(const global::ValueType newLerningRate);
	void setNewPhaseMode(const NnMode nn_mode);

	void updatePrediction(const global::Tensor &out);
	void updateInput(const global::Tensor &input);

	void addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual);

    void resetGraph();
};

} // namespace nn::visualizer

#endif // VISUALIZERRENDERER
