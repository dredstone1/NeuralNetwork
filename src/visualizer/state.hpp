#ifndef STATE
#define STATE

#include <atomic>
#include <string>

using namespace std;

namespace Visualizer {
#define STATES_COUNT 3

enum class states {
	Pause,
	PreciseMode,
	AutoPause,
	None,
};

const string statesName[] = {
    "pause",
    "precise mode",
    "auto pause",
};

enum class NNmode {
	Forword,
	Backward,
	None,
};

const string NNmodeName[] = {
    "Forword",
    "Backward",
};

typedef struct state {
	atomic<bool> pause{false};
	atomic<bool> preciseMode{true};
	atomic<bool> autoPause{true};
	atomic<bool> update_mode{false};
	atomic<NNmode> nnMode{NNmode::Forword};
	void toggle(states state_);
	void toggle(string state_) { toggle(getStatefromString(state_)); }
	string getStateString(states state_);
	states getStatefromString(string &_state);
	bool getState(states state_);
	void setState(states state_, bool stateM);
	void setState(string &state_, bool stateM) { setState(getStatefromString(state_), stateM); }
} state;
} // namespace Visualizer

#endif // STATE
