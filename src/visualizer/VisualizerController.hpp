#ifndef VISUALIZERCONTROLLER
#define VISUALIZERCONTROLLER

#include "VisualizerRenderer.hpp"
#include <thread>

namespace nn::visualizer {
constexpr int WAIT_DELAY_MM = 1;

class visualizerController {
  private:
	void update_display();
	std::atomic<bool> running{false};
	std::unique_ptr<VisualizerRenderer> renderer;
	const model::ConfigData &config;
	std::shared_ptr<StateManager> Vstate;
	std::thread display_thread;

	void stop();
	void start_visuals(const model::NeuralNetwork &network);
	bool checkP();
	void wait_until_updated();
	void wait_until_started();
	void pause();
	void autoPause();
	void initState();
	void handleStates();

  public:
	visualizerController(const model::ConfigData &config);
	~visualizerController();
	void updateDots(const int layer, const model::Neurons &newNeurons);
	void update(const int layer, const model::LayerParameters &gradient);
	void setNewPhaseMode(const NnMode nn_mode);
	void start(const model::NeuralNetwork &network);
	void update(const training::gradient &new_grad);
	void updateBatchCounter(const int batch);
	void updateError(const global::ValueType error, const int index);
	void updateAlgoritemMode(const AlgorithmMode algoritem_mode);
	void update_prediction(const int index);
	void update_lr(const global::ValueType newLerningRate);
	bool exit_training();
};
} // namespace nn::visualizer

#endif // VISUALIZERCONTROLLER
