#include "state.hpp"
#include <cstdio>
#include <cstring>

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

states state::getStatefromString(string &state_) {
	for (int i = 0; i < STATES_COUNT; i++) {
		if (!statesName[i].compare(state_)) {
			return (states)i;
		}
	}

	return None;
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

void state::setState(states state_, bool stateM) {
	switch (state_) {
	case Pause:
		pause.store(stateM);
		break;
	case PreciseMode:
		preciseMode.store(stateM);
		break;
	case AutoPause:
		autoPause.store(stateM);
		break;
	default:
		break;
	}
}
} // namespace Visualizer
