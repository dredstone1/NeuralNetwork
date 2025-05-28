#ifndef STATE
#define STATE

#include <atomic>
#include <string>

namespace Visualizer {
#define STATES_COUNT 3

enum class states {
	Pause,
	PreciseMode,
	AutoPause,
	None,
};

const std::string statesName[] = {
    "pause",
    "precise mode",
    "auto pause",
};

enum class NNmode {
	Forword,
	Backward,
	None,
};

const std::string NNmodeName[] = {
    "Forword",
    "Backward",
};

typedef struct state {
    std::atomic<bool> pause{false};
    std::atomic<bool> preciseMode{true};
    std::atomic<bool> autoPause{true};
    std::atomic<bool> update_mode{false};
    std::atomic<NNmode> nnMode{NNmode::Forword};
	void toggle(states state_);
	void toggle(std::string state_) { toggle(getStatefromString(state_)); }
    std::string getStateString(states state_);
	states getStatefromString(std::string &_state);
	bool getState(states state_);
	void setState(states state_, bool stateM);
	void setState(std::string &state_, bool stateM) { setState(getStatefromString(state_), stateM); }
} state;
} // namespace Visualizer

#endif // STATE
