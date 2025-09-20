/**
 * @file ProgressBar.hpp
 * @brief Implementation of console progress bar for neural network training
 *
 * This file implements a visual progress bar that displays training progress
 * in the console. It provides real-time feedback to users about training
 * completion status with percentage indicators and visual progress indication.
 */

#ifndef PROGRESSBAR
#define PROGRESSBAR

#include <Globals.hpp>
#include <string>

namespace nn {
constexpr int BAR_WIDTH = 100;

class ProgressBar {
	const int total;
	const std::string header;

	int current{0};
	int last_percentage{-1};
	bool headerPrinted{false};

  public:
	ProgressBar(const int total_, const std::string header_)
	    : total(total_),
	      header(header_ + ": ") {}
	~ProgressBar() = default;

	/**
	 * @brief Prints the progress bar to the console
	 *
	 * Renders a visual progress bar showing current completion percentage.
	 * The bar is only updated when the percentage changes to avoid excessive
	 * console output. On first call, prints the header message.
	 *
	 * Header:
	 * [========  ] 75%
	 */
	void printBar();

	/**
	 * @brief Completes the progress bar and moves to next line
	 *
	 * Forces the progress bar to show 100% completion and prints a newline
	 * to move the cursor to the next line for subsequent output.
	 */
	void endPrint();

	/**
	 * @brief Pre-increment operator for advancing progress
	 *
	 * Increments the current progress counter and ensures it doesn't
	 * exceed the total value.
	 *
	 * @return Reference to this ProgressBar for method chaining
	 */
	ProgressBar &operator++();

	/**
	 * @brief Post-increment operator for advancing progress
	 *
	 * Creates a copy of the current state, increments the progress,
	 * and returns the copy.
	 *
	 * @return Copy of the ProgressBar before incrementing
	 */
	ProgressBar operator++(int);

	/**
	 * @brief Assignment operator for setting progress value
	 *
	 * Sets the current progress to a specific value, ensuring it doesn't
	 * exceed the total. Returns a copy of the previous state.
	 *
	 * @param value New progress value to set
	 * @return Copy of the ProgressBar before assignment
	 */
	ProgressBar operator=(int value);
};
} // namespace nn

#endif // PROGRESSBAR
