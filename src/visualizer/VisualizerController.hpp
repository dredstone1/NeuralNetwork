#ifndef VISUALIZERCONTROLLER
#define VISUALIZERCONTROLLER

#include "VisualizerRenderer.hpp"
#include <SFML/Graphics.hpp>
#include <thread>

namespace nn {
namespace Visualizer {
constexpr int WAIT_DELAY_MM = 1;

class visualizerController {
  private:
	void update_display();
	std::atomic<bool> running{false};
	std::unique_ptr<VisualizerRenderer> renderer;
	const ConfigData &config;
	std::shared_ptr<state> Vstate;
	void stop();
	std::thread display_thread;
	void start_visuals(const neural_network &network);
	bool checkP();
	void wait_until_updated();
	void wait_until_started();
	void pause();
	void autoPause();
	void initState();
	void handleStates();

  public:
	visualizerController(const ConfigData &config);
	~visualizerController();
	void updateDots(const int layer, const std::vector<Global::ValueType> out, const std::vector<Global::ValueType> net);
	void update(const int layer, const LayerParameters &gradient);
	void setNewPhaseMode(const NNmode nn_mode);
	void start(const neural_network &network);
	void update(const gradient &new_grad);
	void updateBatchCounter(const int batch);
	void updateError(const Global::ValueType error, const int index);
	void updateAlgoritemMode(const algorithmMode algoritem_mode);
	void update_prediction(const int index);
};
} // namespace Visualizer
} // namespace nn

#endif // VISUALIZERCONTROLLER
