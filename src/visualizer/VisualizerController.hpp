#ifndef VISUALIZERCONTROLLER
#define VISUALIZERCONTROLLER

#include "VisualizerRenderer.hpp"
#include "state.hpp"
#include <SFML/Graphics.hpp>
#include <mutex>
#include <thread>

namespace Visualizer {
class visualizerController {
  private:
	void update_display();
	mutex mtx;
	atomic<bool> running{false};
	VisualizerRenderer *renderer;
	state *Vstate;
	void stop();
	void start(const neural_network &network);
	thread display_thread;
	void start_visuals(const neural_network &network);
	void wait_until_updated();
	void pause();
	void autoPause();

  public:
	visualizerController(const neural_network &network);
	~visualizerController();
	void updateDots(const int layer, vector<double> out, vector<double> net);
	void update(const int layer, const LayerParameters &gradient);
};
} // namespace Visualizer

#endif // VISUALIZERCONTROLLER
