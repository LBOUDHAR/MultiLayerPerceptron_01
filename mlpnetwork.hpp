// mlpnetwork.h for Multi Layer Perceptron network: 
// Program created on 21.06.2022 by Lies BOUDHAR
// Modified on 31.07.2022	at 21:44

#ifndef MLPNETWORK_H
#define MLPNETWORK_H

#include <vector>

using namespace std;

#define SEGMOID 01
#define TANH 02
#define RELU 03
#define SOFTMAX 04
#define GELU 05
#define ELU 06
#define ARCTAN 07

unsigned int func = SEGMOID;

typedef vector<vector<float>> Mat2d;

struct dim {
	unsigned int row;
	unsigned int col;
};

// Multi Layer Perceptron network implementation class
// Topology contain the number of neurons in each layer. The size of the topology is a number of layers
// The first layer is not a true layer then the nombre of layer is (topology.size()-1)
class MLP_Network {
public:
	vector <float> inputs;      
	vector <float> outputs;
	
	MLP_Network(const vector<unsigned int>& _topology, const vector<float>& _inputs, const vector<float>& _outputs );
	~MLP_Network();
	void forwardPropagat(unsigned int _layerActFunc, unsigned int _lastLayerActFunc);		// function for forward propagation of data
	void backwardPropagat(float alpha);	// function for backward propagation of errors made by neurons
	void train();
	void showWeightsVals();
	void showLayersZVals();
	void showLayersAVals();
	void showBiasVals();

private:
	vector<unsigned int> topology;			// it contains the number of neurons for each layer
	vector<dim> dims;						// it contains the size of each layer weight in layersWeights.
	dim couple = { 0, 0 };
	Mat2d weights;							//it contains weights for one layer
	Mat2d out_layersA;						//it contains outputs values for each layer after activation
	Mat2d out_layersZ;						//it contains outputs values for each layer before activation
	vector<Mat2d> layersWeights;			//it contains weights for each layer
	Mat2d bias;
    float segmoid(float value);
	float relu(float value);
};

#endif