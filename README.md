# 🧠 Neural Network Library

A comprehensive C++ neural network library designed for building, training, and evaluating deep learning models. This project provides a high-performance, modular implementation with GPU acceleration, real-time visualization, and extensive documentation.

## ✨ Key Features

- **🚀 Modern C++**: Built with C++17 for optimal performance and clean code
- **🔧 Modular Architecture**: Easily extensible design for custom network architectures and activations 
- **⚡ Neural Network Components**: Complete support for sub-networks, activation functions, loss computations, and more
- **📦 CMake Ready**: Seamless build system integration
- **🎯 Research Focused**: Designed with experimentation and learning in mind
- **🧠 Real-time Visualization**:
    - **Full Live Network**: Real-time display of all layers, neurons, and activations for a complete live view of the model  
    - **Status Only**: Lightweight view showing just the current network status, ideal for fast debugging or monitoring  
- **💾 Model Persistence**: Save and load trained model parameters for reuse and deployment
- **⚙️ GPU Support**: Leverages CUDA for accelerated training and inference on compatible NVIDIA GPUs, boosting performance for large-scale models  
- **💻 Flexible Device Support**: Simple to switch between CPU and GPU for training and inference


## 🏗️ Architecture Overview

### Core Components

- **Tensor Class**: Multi-dimensional array with CPU/GPU support
- **Activation Functions**: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax
- **Network Types**: Fully Connected (FNN) and Convolutional (CNN) networks
- **Optimizers**: Various gradient descent optimizers - still in work
- **Data Management**: Flexible dataset loading and preprocessing
- **Visualization**: Real-time network visualization and monitoring

### Design Principles

- **Performance First**: Optimized for both CPU and GPU execution
- **Modularity**: Clean separation of concerns for easy extension
- **Documentation**: Comprehensive inline documentation and examples

## 🚀 Quick Start

### Prerequisites

- **C++17-compatible compiler** (GCC 7+, Clang 5+, MSVC 2017+)
- **CMake 3.28+** for build system
- **CUDA Toolkit 11.0+** (optional, for GPU support)
- **OpenGL** (for visualization features)

### Installation

```bash
# Clone the repository
git clone https://github.com/dredstone1/NeuralNetwork.git

# Navigate to project directory
cd NeuralNetwork

# Create and enter build directory
mkdir build && cd build

# Configure and build
cmake ..
make
```

## 📁 Project Structure

```
NeuralNetwork/
├── 📂 include/                    # Public API headers
│   ├── 📄 tensor.hpp             # Core tensor class
│   ├── 📄 model.hpp              # Main model interface
│   ├── 📄 dataBase.hpp           # Dataset management
│   ├── 📄 Globals.hpp            # Global constants and types
│   └── 📂 network/               # Network interface definitions
├── 📂 src/                       # Implementation files
│   ├── 📂 model/                 # Core model implementation
│   │   ├── 📄 tensor.cpp         # Tensor implementation
│   │   ├── 📄 activations.cpp    # Activation functions
│   │   ├── 📄 optimizers.cpp     # Optimization algorithms
│   │   └── 📄 dataBase.cpp       # Dataset handling
│   ├── 📂 networks/              # Network implementations
│   │   ├── 📂 fnn/               # Fully Connected Networks
│   │   └── 📂 cnn/               # Convolutional Networks
│   └── 📂 visualizer/            # Visualization components
├── 📂 tests/                     # Unit tests and examples
│   ├── 📂 data/                  # Test datasets
│   └── 📄 *.cpp                  # Test files
├── 📂 resources/                 # Static assets
│   └── 📂 fonts/                 # Font files for visualization
├── 📄 CMakeLists.txt             # Build configuration
├── 📄 Doxyfile                   # Documentation configuration
└── 📖 README.md                  # This file
```

## 🤝 Contributing

Contributions are more than welcome! Whether you're fixing bugs, adding features, or improving documentation:

- 🐛 Open issues for bugs or feature requests
- 🔀 Submit pull requests with your improvements
- 💡 Share ideas and suggestions
- 📖 Help improve documentation

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.

## 👨‍💻 Author

**dredstone1** - [GitHub Profile](https://github.com/dredstone1)

## 🙏 Acknowledgments

- CUDA team for GPU acceleration support
- OpenGL community for visualization capabilities
- C++ community for modern language features

---

⭐ If you find this project interesting, consider giving it a star!
