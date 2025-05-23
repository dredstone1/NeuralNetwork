#include "state.hpp"
#include <cstdio>

namespace Visualizer {
void state::toggle(states state_) {
	switch (state_) {
	case Pause:
		pause.store(!pause.load());
		break;
	case PreciseMode:
		preciseMode.store(!preciseMode.load());
		break;
	case AutoPause:
		autoPause.store(!autoPause.load());
		break;
	default:
		break;
	}
}

string state::getString(states state_) {
	return statesName[state_];
}
} // namespace Visualizer
