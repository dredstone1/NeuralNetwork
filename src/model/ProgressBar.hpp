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

	void printBar();
	void endPrint();

	ProgressBar operator++(int);
	ProgressBar operator=(int value);

	void reset();
};
} // namespace nn

#endif // PROGRESSBAR
