#include "state.hpp"
#include <cstdio>
#include <cstring>

namespace Visualizer {
void state::toggle(states state_) {
	switch (state_) {
	case states::Pause:
		pause.store(!pause.load());
		break;
	case states::PreciseMode:
		preciseMode.store(!preciseMode.load());
		break;
	case states::AutoPause:
		autoPause.store(!autoPause.load());
		break;
	default:
		break;
	}
}

std::string state::getStateString(states state_) {
	return statesName[(int)state_];
}

states state::getStatefromString(const std::string &state_) {
	for (int i = 0; i < STATES_COUNT; i++) {
		if (!statesName[i].compare(state_)) {
			return (states)i;
		}
	}

	return states::None;
}

bool state::getState(states state_) {
	switch (state_) {
	case states::Pause:
		return pause.load();
		break;
	case states::PreciseMode:
		return preciseMode.load();
		break;
	case states::AutoPause:
		return autoPause.load();
		break;
	default:
		return 0;
		break;
	}
}

void state::setState(states state_, bool stateM) {
	switch (state_) {
	case states::Pause:
		pause.store(stateM);
		break;
	case states::PreciseMode:
		preciseMode.store(stateM);
		break;
	case states::AutoPause:
		autoPause.store(stateM);
		break;
	default:
		break;
	}
}
} // namespace Visualizer
