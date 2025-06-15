#ifndef TRAINER
#define TRAINER

#include "../src/trainer/backPropagation.hpp"

namespace nn::training {
constexpr int BAR_WIDTH = 100;
constexpr int SECONDS_IN_MINUTE = 60;

class Trainer {
  private:
	model::TrainingConfig &config;
	DataBase dataBase;
	AiModel &model;
	learningRateParams learningRate;
	BackPropagation backPropagation;
	int last_progress;
	void print_progress_bar(const int current, const int total);

  public:
	Trainer(AiModel &_model)
	    : config(_model.getConfig().config_data.training_config),
	      dataBase(config),
	      model(_model),
	      learningRate(model.config.config_data.training_config.lr_init_value),
	      backPropagation(_model, learningRate),
	      last_progress(-1) {}
	~Trainer() = default;

	void train();
};
} // namespace nn::training

#endif // TRAINER
