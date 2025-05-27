#include "../../include/trainer.hpp"
#include "dataBase.hpp"
#include "model/config.hpp"
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>

void Trainer::print_progress_bar(int current, int total) {
	float progress = (float)current / total;
	int progress_percentage = int(progress * 100.0);

	if (progress_percentage != last_progress) {
		int pos = BAR_WIDTH * progress;
		last_progress = progress_percentage;

		ostringstream oss;
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

		cout << oss.str();
		cout.flush();
	}
}

void Trainer::train() {
	cout << "Training AI" << endl;

	const double graph_resolution = min(GRAPH_RESOLUTION, config.batch_count);
	const int graph_draw_interval = config.batch_count / graph_resolution;
	cout << "Graph resolution: " << graph_resolution << endl;
	vector<double> errors(graph_resolution, 0.0);
	double error = 0.0;

	const auto start = chrono::high_resolution_clock::now();

	for (int loop_index = 0; loop_index < config.batch_count; loop_index++) {
		Batch &batch = dataBase.get_Batch();
		error += backPropagation.run_back_propagation(batch, config.learning_rate);

		if ((loop_index + 1) % graph_draw_interval == 0) {
			errors[loop_index / graph_draw_interval] = error / graph_draw_interval;
			error = 0.0;
		}

		print_progress_bar(loop_index + 1, config.batch_count);
	}

	const auto end = chrono::high_resolution_clock::now();
	const int time_taken = chrono::duration_cast<chrono::seconds>(end - start).count();
	const int minutes = time_taken / SECONDS_IN_MINUTE;
	const int seconds = time_taken % SECONDS_IN_MINUTE;
	const int time_taken_milliseconds = chrono::duration_cast<chrono::milliseconds>(end - start).count();
	cout << endl
	     << "Training Done!" << endl
	     << "Training time: " << minutes << " minutes " << seconds << " seconds" << " (" << time_taken_milliseconds << " ms)" << endl;

	double min = *min_element(errors.begin(), errors.end());
	printf("Minimum error: %f\n", min);
}

Trainer::Trainer(AiModel *_model) : config(_model->getConfig().config_data.training_config), dataBase(config), backPropagation(*_model) {
	model = _model;
	last_progress = -1;
}
