#ifndef PANEL
#define PANEL

#include "state.hpp"
#include <memory>

namespace Visualizer {
class panel {
  private:
	virtual void do_render() = 0;

  protected:
	bool need_update;
	std::shared_ptr<state> vstate;

  public:
	panel(std::shared_ptr<state> vstate_) : vstate(vstate_) {}
	void render() {
		if (need_update) {
			do_render();
			need_update = false;
		}
	}
	virtual ~panel() = default;
	bool updateStatus() { return need_update; }
};
} // namespace Visualizer

#endif // PANEL
