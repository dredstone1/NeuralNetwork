/**
 * @file ProgressBar.cpp
 * @brief Implementation of console progress bar for neural network training
 * 
 * This file implements a visual progress bar that displays training progress
 * in the console. It provides real-time feedback to users about training
 * completion status with percentage indicators and visual progress indication.
 */

#include "ProgressBar.hpp"
#include <iomanip>
#include <iostream>

namespace nn {

/**
 * @brief Prints the progress bar to the console
 * 
 * Renders a visual progress bar showing current completion percentage.
 * The bar is only updated when the percentage changes to avoid excessive
 * console output. On first call, prints the header message.
 * 
 * Format: [========  ] 75%
 */
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
			std::cout << "=";  // Filled portion
		} else {
			std::cout << " ";  // Empty portion
		}
	}
	std::cout << "] " << std::setw(3) << percentage << "%" << std::flush;
}

/**
 * @brief Completes the progress bar and moves to next line
 * 
 * Forces the progress bar to show 100% completion and prints a newline
 * to move the cursor to the next line for subsequent output.
 */
void ProgressBar::endPrint() {
	current = total;
	last_percentage = -1;  // Force update
	printBar();

	std::cout << std::endl;  // Move to next line
}

/**
 * @brief Pre-increment operator for advancing progress
 * 
 * Increments the current progress counter and ensures it doesn't
 * exceed the total value.
 * 
 * @return Reference to this ProgressBar for method chaining
 */
ProgressBar &ProgressBar::operator++() {
	++current;
	if (current > total) {
		current = total;  // Clamp to maximum value
	}
	return *this;
}

/**
 * @brief Post-increment operator for advancing progress
 * 
 * Creates a copy of the current state, increments the progress,
 * and returns the copy.
 * 
 * @return Copy of the ProgressBar before incrementing
 */
ProgressBar ProgressBar::operator++(int) {
	ProgressBar temp = *this;
	++(*this);
	return temp;
}

/**
 * @brief Assignment operator for setting progress value
 * 
 * Sets the current progress to a specific value, ensuring it doesn't
 * exceed the total. Returns a copy of the previous state.
 * 
 * @param value New progress value to set
 * @return Copy of the ProgressBar before assignment
 */
ProgressBar ProgressBar::operator=(int value) {
	ProgressBar temp = *this;
	current = value;
	if (current > total) {
		current = total;  // Clamp to maximum value
	}
	return temp;
}

}; // namespace nn
