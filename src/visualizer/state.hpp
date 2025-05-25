#ifndef STATE
#define STATE

#include <atomic>
#include <string>

using namespace std;

namespace Visualizer {
#define STATES_COUNT 3

enum states {
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

typedef struct state {
	atomic<bool> pause{false};
	atomic<bool> preciseMode{true};
	atomic<bool> autoPause{true};
	void toggle(states state_);
	string getString(states state_);
    bool getState(states state_);
} state;
} // namespace Visualizer

#endif // STATE
