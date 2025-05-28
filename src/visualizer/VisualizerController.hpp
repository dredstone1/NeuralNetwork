#ifndef VISUALIZERCONTROLLER
#define VISUALIZERCONTROLLER

#include "../model/neuralNetwork.hpp"
#include "VisualizerRenderer.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <mutex>
#include <thread>

namespace Visualizer {
class visualizerController {
  private:
	void update_display();
    std::mutex mtx;
    std::atomic<bool> running{false};
	VisualizerRenderer *renderer;
	VisualizerConfig &config;
	state *Vstate;
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
	visualizerController(VisualizerConfig &config);
	~visualizerController();
	void updateDots(const int layer, std::vector<double> out, std::vector<double> net);
	void update(const int layer, const LayerParameters &gradient);
	void setNewPhaseMode(const NNmode nn_mode);
	void start(const neural_network &network);
};
} // namespace Visualizer

#endif // VISUALIZERCONTROLLER
