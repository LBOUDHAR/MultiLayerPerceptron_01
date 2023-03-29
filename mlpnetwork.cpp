#include "mlpnetwork.hpp"

#include <windows.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;




MLP_Network::MLP_Network(const vector<unsigned int>& _topology, const vector<float>& _inputs, const vector<float>& _outputs) {
	topology = _topology;
	inputs = _inputs;
	outputs = _outputs;
	// initialize weights matrix
	random_device rd;
	default_random_engine generator(rd());
	uniform_real_distribution<float> distribution(-0.5, 0.5);
	for (unsigned int i = 0; i < topology.size()-1; i++) {
		for (unsigned int j = 0; j < topology[i+1]; j++) {
			vector<float> v;
			for (unsigned int k = 0; k < topology[i]; k++) {
				v.push_back(distribution(generator));                   
			}
			weights.push_back(v);
			couple.row = v.size();
		}
		layersWeights.push_back(weights);
		couple.col = weights.size();
		dims.push_back(couple);		//save the size of each layer weight.
		weights.clear();
	}
	// initialize out_layersA and out_layersZ matrix and bias matrix
	for (unsigned int i = 0; i < topology.size() - 1; i++) {
		vector<float> v, w;
		for (unsigned int j = 0; j < topology[i + 1]; j++) {
			v.push_back(0);
			w.push_back(1);
		}
		out_layersZ.push_back(v);
		out_layersA.push_back(v);
		bias.push_back(w);
	}
}

MLP_Network::~MLP_Network() {

}

float MLP_Network::segmoid(float value){
    return 1 / (1 + exp(-value));
}

float MLP_Network::relu(float value){
	float x;
    if (value > 0.0) {
        x = value;
    } else {
        x = 0.0;
    }
    return x;
}


void MLP_Network::forwardPropagat(unsigned int _layerActFunc, unsigned int _lastLayerActFunc) {
	int LastLayer = layersWeights.size()-1 ;
	float somme = 0;
	for (unsigned int i = 0; i < layersWeights.size(); i++) {
		for (unsigned int j = 0; j < dims[i].col; j++) {
			for (unsigned int k = 0; k < dims[i].row; k++) {
				if (i == 0) out_layersZ[i][j] += inputs[k] * layersWeights[i][j][k] ;
				if (i != 0) out_layersZ[i][j] += out_layersZ[i-1][k] * layersWeights[i][j][k];   // (i-1) is num of layer (k) is num of perceptron for out_layer
			}
			out_layersZ[i][j] += bias[i][j];
			if ( i == LastLayer && _lastLayerActFunc == 04) somme = somme + exp(out_layersZ[i][j]); // for Activation function ( softmax )
			if( i < LastLayer ){
				if ( _layerActFunc == 01 ) out_layersA[i][j] = segmoid(out_layersZ[i][j]);		// Activation function ( sigmoide )
            	if ( _layerActFunc == 02 ) out_layersA[i][j] = tanh(out_layersZ[i][j]);		// Activation function ( tanh )
				if ( _layerActFunc == 03 ) out_layersA[i][j] = relu(out_layersZ[i][j]);		// Activation function ( relu )
			}else{
				if ( i == LastLayer && _lastLayerActFunc == 01) out_layersZ[i][j] = segmoid(out_layersA[i][j]);	// Activation function ( sigmoide )

			}
		}
		if ( i == LastLayer && _lastLayerActFunc == 04){
			for (unsigned int j = 0; j < dims[i].col; j++) {
				out_layersA[i][j] = exp(out_layersZ[i][j])/somme;	// Activation function ( softmax)
			}
		}		
	}
}


void MLP_Network::backwardPropagat(float alpha) {


}


void MLP_Network::train() {


}

void MLP_Network::showWeightsVals() {
	for (unsigned int i = 0; i < layersWeights.size(); i++) {
		for (unsigned int j = 0; j < dims[i].col; j++) {
			for (unsigned int k = 0; k < dims[i].row; k++) {
				cout << layersWeights[i][j][k] << "  ";
			}
			cout << endl;
		}
		cout << endl;
	}
}

void MLP_Network::showLayersZVals(){
	for (unsigned int i = 0; i < out_layersZ.size(); i++) {
		for (unsigned int j = 0; j < topology[i + 1]; j++) {
			cout << out_layersZ[i][j] << "  ";
		}
		cout << endl;
	}
    cout << endl;
}

void MLP_Network::showLayersAVals(){
	for (unsigned int i = 0; i < out_layersA.size(); i++) {
		for (unsigned int j = 0; j < topology[i + 1]; j++) {
			cout << out_layersA[i][j] << "  ";
		}
		cout << endl;
	}
    cout << endl;
}

void MLP_Network::showBiasVals(){
	for (unsigned int i = 0; i < bias.size(); i++) {
		for (unsigned int j = 0; j < topology[i + 1]; j++) {
			cout << bias[i][j] << "  ";
		}
		cout << endl;
	}
    cout << endl;
}
