#ifndef VISUALIZERCONTROLLER
#define VISUALIZERCONTROLLER

#include "VisualizerRenderer.hpp"
#include <network/IvisualNetwork.hpp>
#include <thread>

namespace nn::model {
class Model;
}

namespace nn::visualizer {
class VisualManager {
  private:
	void updateDisplay();
	std::atomic<bool> running{false};
	std::unique_ptr<VisualRender> renderer;
	const model::Config &config;
	std::shared_ptr<StateManager> Vstate;
	std::thread displayThread;

	void stop();
	void startVisuals();

	inline bool checkPointers() { return renderer && Vstate; }
	void initState();

	friend class model::Model;

  public:
	VisualManager(const model::Config &_config);
	~VisualManager();

	void start();
	void setNewPhaseMode(const NnMode nn_mode);
	void updateBatchCounter(const int batch);
	void updateLost(
	    const global::ValueType newDataLost,
	    int index);
	void updateEvaluate(
	    const global::ValueType newDataEvaluate,
	    int index);
	void updateAlgorithmMode(const AlgorithmMode algorithm_mode);
	void updateLearningRate(const global::ValueType newLerningRate);

	bool exitTraining();

	void updatePrediction(const global::Prediction &pre);
	void updateInput(const global::ParamMetrix &input);

	void addVisualSubNetwork(const std::shared_ptr<IVisualNetwork> newVisual);
};
} // namespace nn::visualizer

#endif // VISUALIZERCONTROLLER
