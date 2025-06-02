#ifndef VISUALIZERRENDERER
#define VISUALIZERRENDERER

#include "../model/neuralNetwork.hpp"
#include "VInterface.hpp"
#include "Vstatus.hpp"
#include "state.hpp"
#include "visualNN.hpp"
#include <SFML/Graphics.hpp>
#include <atomic>
#include <climits>
#include <memory>

namespace Visualizer {
constexpr const int UI_GAP = 15;
inline const sf::Color BG_COLOR = sf::Color(100, 100, 100);

class VisualizerRenderer {
  private:
	sf::RenderWindow window;
	visualNN visualNetwork;
	std::shared_ptr<state> Vstate;
	vInteface interface;
	vStatus statusV;
	std::atomic<bool> running{false};
	float fps;
	void renderLoop();
	void processEvents();
	void renderObjects();
	void full_update();
	inline void clear();
	void do_frame(int &frameCount, sf::Clock &fpsClock);

  public:
	VisualizerRenderer(const neural_network &network, const std::shared_ptr<state> vstate);
	~VisualizerRenderer();
	void close();
	void updateDots(const int layer, const std::vector<double> out, const std::vector<double> net);
	bool updateStatus();
	void update(const gradient new_grad);
	void start();
	void update(const int layer, const LayerParameters &gradients);
	void setNewPhaseMode(const NNmode nn_mode);
};
} // namespace Visualizer

#endif // VISUALIZERRENDERER
