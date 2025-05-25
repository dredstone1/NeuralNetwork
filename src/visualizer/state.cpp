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

bool state::getState(states state_) {
	switch (state_) {
	case Pause:
		return pause.load();
		break;
	case PreciseMode:
		return preciseMode.load();
		break;
	case AutoPause:
		return autoPause.load();
		break;
	default:
		return 0;
		break;
	}
}
} // namespace Visualizer
