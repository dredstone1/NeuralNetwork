#include <AiModel.hpp>
#include <iostream>
#include <trainer.hpp>

#define IS_BIT_ON(x, n) ((x) & (1 << (n)))

using namespace std;

enum mode {
	load = 0,   // 0001
	save = 1,   // 0010
	train = 2,  // 0100
	visual = 4, // 1000
};

int main(int argc, char *argv[]) {
	if (argc > 1) {
		int mods = 0;
		char *arg = argv[1];
		while (*(arg) != '\0') {
			switch (*arg) {
			case 'l':
				mods |= load;
				break;
			case 's':
				mods |= save;
				break;
			case 't':
				mods |= train;
				break;
			case 'v':
				mods |= visual;
				break;
			}

			arg++;
		}

		AiModel model(mods &visual);

		if (mods & load) {
			model.load("model1");
		}

		if (mods & train) {
			int batch_size = 64, batch_count = 10000;

			double learning_rate = 0.000001;

			Trainer trainer(
			    "database",
			    &model,
			    batch_size,
			    batch_count,
			    learning_rate);

			trainer.train();
		}

		if (mods & save) {
			model.save("model1");
		}

		return 0;
	}

	cout << "Need at least 1 input!" << endl;
	return 1;
}
