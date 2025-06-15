#ifndef VISUALIZERCONTROLLER
#define VISUALIZERCONTROLLER

#include "VisualizerRenderer.hpp"
#include <thread>

namespace nn::visualizer {
class VisualManager {
  private:
	void update_display();
	std::atomic<bool> running{false};
	std::unique_ptr<VisualRender> renderer;
	const model::ConfigData &config;
	std::shared_ptr<StateManager> Vstate;
	std::thread display_thread;

	void stop();
	void start_visuals(const model::NeuralNetwork &network);

	inline bool checkPointers() { return renderer && Vstate; }
	void initState();

  public:
	VisualManager(const model::ConfigData &config);
	~VisualManager();

	void start(const model::NeuralNetwork &network);
	void updateDots(const int layer, const model::Neurons &newNeurons);
	void update(const int layer, const model::LayerParameters &gradient);
	void setNewPhaseMode(const NnMode nn_mode);
	void update(const training::gradient &new_grad);
	void updateBatchCounter(const int batch);
	void updateError(const global::ValueType error, const int index);
	void updateAlgoritemMode(const AlgorithmMode algoritem_mode);
	void updatePrediction(const int index);
	void updateLearningRate(const global::ValueType newLerningRate);

	bool exit_training();
};
} // namespace nn::visualizer

#endif // VISUALIZERCONTROLLER
