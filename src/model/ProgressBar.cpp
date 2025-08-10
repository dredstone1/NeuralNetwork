#include "ProgressBar.hpp"
#include <iostream>

namespace nn {
void ProgressBar::printBar() {
	if (total == 0)
		return;

	int percentage = 100.0 * current / total;

	if (percentage == last_percentage)
		return;

	last_percentage = percentage;

	const int pos = BAR_WIDTH * current / total;

	char bar[BAR_WIDTH + 64];
	int index = 0;

	bar[index++] = '[';
	for (int i = 0; i < BAR_WIDTH; ++i) {
		if (i < pos)
			bar[index++] = '=';
		else if (i == pos)
			bar[index++] = '>';
		else
			bar[index++] = ' ';
	}
	bar[index++] = ']';
	bar[index++] = ' ';

	int written = std::snprintf(bar + index, sizeof(bar) - index, "%3d %%", percentage);
	if (written > 0)
		index += written;

	bar[index++] = (percentage == 100) ? '\n' : '\r';
	bar[index] = '\0';

	std::cout << header << bar << std::flush;
}

ProgressBar ProgressBar::operator++(int) {
	ProgressBar temp = *this;
	++current;
	if (current > total)
		current = total;
	return temp;
}

ProgressBar ProgressBar::operator=(int value) {
	ProgressBar temp = *this;
	current = value;
	if (current > total)
		current = total;
	return temp;
}

void ProgressBar::reset() {
	current = 0;
	last_percentage = -1;
}

}; // namespace nn
