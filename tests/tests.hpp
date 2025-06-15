#ifndef TESTS
#define TESTS

#include <string>
#include <string_view>

namespace tests {
constexpr std::string_view DATA_PATH = "../tests/data/";

// This is the correct, idiomatic, and high-performance way.
inline std::string appendToBase(std::string_view suffix) {
	std::string result;
	result.reserve(DATA_PATH.size() + suffix.size());

	result.append(DATA_PATH);
	result.append(suffix);

	return result;
}

} // namespace tests

#endif // VINTERFACE
