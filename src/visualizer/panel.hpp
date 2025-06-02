#ifndef PANEL
#define PANEL

#include "state.hpp"
#include <memory>

namespace Visualizer {
class panel {
  private:
	virtual void do_render() = 0;
	bool need_update;

  protected:
	std::shared_ptr<state> vstate;

  public:
	panel(const std::shared_ptr<state> vstate_)
	    : vstate(vstate_) {}
	int render();
	virtual ~panel() = default;
	bool updateStatus() const { return need_update; }
	void set_update() { need_update = true; }
};
} // namespace Visualizer

#endif // PANEL
