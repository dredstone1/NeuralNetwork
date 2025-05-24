#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using namespace std;

struct LayerConfig {
	int size;
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(LayerConfig, size);

struct NetworkConfig {
	int input_size;
	int output_size;
	vector<LayerConfig> layers_config;
	int hidden_layer_count() const { return layers_config.size(); }
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(NetworkConfig, input_size, output_size, layers_config);

class Config {
  public:
	Config(const string &config_filepath);
	NetworkConfig network_config;

  private:
};

#endif // CONFIG
