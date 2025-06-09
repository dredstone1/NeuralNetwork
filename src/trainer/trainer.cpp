#include <iostream>
#include <trainer.hpp>

namespace nn::training {
void Trainer::print_progress_bar(const int current, const int total) {
	float progress = (float)current / total;
	int progress_percentage = int(progress * BAR_WIDTH);

	if (progress_percentage != last_progress) {
		int pos = BAR_WIDTH * progress;
		last_progress = progress_percentage;

		std::ostringstream oss;
		oss << "[";
		for (int i = 0; i < BAR_WIDTH; ++i) {
			if (i < pos)
				oss << "=";
			else if (i == pos)
				oss << ">";
			else
				oss << " ";
		}
		oss << "] " << progress_percentage << " %\r";

		std::cout << oss.str();
		std::cout.flush();
	}
}

void Trainer::train() {
	std::cout << "Training AI" << std::endl;

	const auto start = std::chrono::high_resolution_clock::now();
	global::ValueType error = 0.0;

	model.model->visual.updateAlgoritemMode(visualizer::AlgorithmMode::Training);
	model.model->visual.update_lr(lr.getLearningRate());
	for (int loop_index = 0; loop_index < config.batch_count + 1; loop_index++) {
		model.model->visual.updateBatchCounter(loop_index);

		Batch &batch = dataBase.get_Batch();
		error = backPropagation.run_back_propagation(batch);

		model.model->visual.updateError(error, loop_index);

		print_progress_bar(loop_index + 1, config.batch_count);

		model.model->visual.update_lr(lr.getLearningRate());
		if (model.model->visual.exit_training() == true)
			break;
	}

	const auto end = std::chrono::high_resolution_clock::now();
	const int time_taken = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	const int minutes = time_taken / SECONDS_IN_MINUTE;
	const int seconds = time_taken % SECONDS_IN_MINUTE;
	const int time_taken_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << std::endl
	          << "Training Done!" << std::endl
	          << "Training time: "
	          << minutes << " minutes "
	          << seconds << " seconds" << " ("
	          << time_taken_milliseconds << " ms)" << std::endl
	          << "final_score: " << error << std::endl;

	model.model->visual.updateAlgoritemMode(visualizer::AlgorithmMode::Normal);
}
} // namespace nn::training
