#ifndef VISUALIZERCONTROLLER
#define VISUALIZERCONTROLLER

#include "VisualizerRenderer.hpp"
#include <thread>

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

  public:
	VisualManager(const model::Config &_config);
	~VisualManager();

	void start();
	void setNewPhaseMode(const NnMode nn_mode);
	void updateBatchCounter(const int batch);
	void updateError(const global::ValueType error, const int index);
	void updateAlgorithmMode(const AlgorithmMode algorithm_mode);
	void updatePrediction(const int index);
	void updateLearningRate(const global::ValueType newLerningRate);

	bool exitTraining();
};
} // namespace nn::visualizer

#endif // VISUALIZERCONTROLLER
