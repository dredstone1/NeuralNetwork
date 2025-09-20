

#include "ProgressBar.hpp"
#include <iomanip>
#include <iostream>

namespace nn {

void ProgressBar::printBar() {
	// Print header on first call
	if (!headerPrinted) {
		std::cout << header << "\n";
		headerPrinted = true;
	}

	// Calculate current percentage (0-100)
	int percentage = (total > 0) ? (current * 100 / total) : 0;

	// Only update if percentage has changed to reduce console spam
	if (percentage == last_percentage) {
		return;
	}
	last_percentage = percentage;

	// Calculate number of filled characters in the progress bar
	int filled = (BAR_WIDTH * percentage) / 100;

	// Render the progress bar with carriage return for in-place updates
	std::cout << "\r[";
	for (int i = 0; i < BAR_WIDTH; ++i) {
		if (i < filled) {
			std::cout << "="; // Filled portion
		} else {
			std::cout << " "; // Empty portion
		}
	}
	std::cout << "] " << std::setw(3) << percentage << "%" << std::flush;
}

void ProgressBar::endPrint() {
	current = total;
	last_percentage = -1; // Force update
	printBar();

	std::cout << std::endl; // Move to next line
}


ProgressBar &ProgressBar::operator++() {
	++current;
	if (current > total) {
		current = total; // Clamp to maximum value
	}
	return *this;
}


ProgressBar ProgressBar::operator++(int) {
	ProgressBar temp = *this;
	++(*this);
	return temp;
}


ProgressBar ProgressBar::operator=(int value) {
	ProgressBar temp = *this;
	current = value;
	if (current > total) {
		current = total; // Clamp to maximum value
	}
	return temp;
}

}; // namespace nn
