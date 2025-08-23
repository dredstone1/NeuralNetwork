#include "ProgressBar.hpp"
#include <iomanip>
#include <iostream>

namespace nn {

void ProgressBar::printBar() {
	if (!headerPrinted) {
		std::cout << header << "\n";
		headerPrinted = true;
	}

	int percentage = (total > 0) ? (current * 100 / total) : 0;
	if (percentage == last_percentage) {
		return;
	}
	last_percentage = percentage;

	int filled = (BAR_WIDTH * percentage) / 100;

	std::cout << "\r[";
	for (int i = 0; i < BAR_WIDTH; ++i) {
		if (i < filled) {
			std::cout << "=";
		} else {
			std::cout << " ";
		}
	}
	std::cout << "] " << std::setw(3) << percentage << "%" << std::flush;
}

void ProgressBar::endPrint() {
	current = total;
	last_percentage = -1;
	printBar();

	std::cout << std::endl;
}

ProgressBar ProgressBar::operator++(int) {
	ProgressBar temp = *this;
	++current;
	if (current > total) {
		current = total;
	}
	return temp;
}

ProgressBar ProgressBar::operator=(int value) {
	ProgressBar temp = *this;
	current = value;
	if (current > total) {
		current = total;
	}
	return temp;
}

void ProgressBar::reset() {
	current = 0;
	last_percentage = -1;
}

}; // namespace nn
