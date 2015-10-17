#include <iostream>
#include <array>
#include "neural_network/neurons/Neuron.h"
#include "neural_network/transfer_functions/FastSigmoidTransferFunction.h"
#include "neural_network/ForwardPropagationNN.h"
#include "neural_network/random/RandomUniformDoubleGenerator.h"


template <unsigned N>
using NeuronTemplate = neural_network::neurons::Neuron<N, neural_network::transfer_functions::FastSigmoidTransferFunction<>>;



int main()
{
	neural_network::RandomUniformDoubleGenerator rnd;
	neural_network::ForwardPropagationNN<NeuronTemplate, 2, 1, 1> network;
	network.randomizeWeights(rnd);

	std::array<double, 2> input1 = {1,1};
	std::array<double, 1> expected_output1 = {1};
	std::array<double, 2> input2 = { 1,0 };
	std::array<double, 1> expected_output2 = { 0 };
	auto output_before1 = network.calculateOutput(input1);
	auto output_before2 = network.calculateOutput(input2);
	
	for (auto i = 0; i < 1000; i++)
	{
		network.train(input1, expected_output1);
		//network.train(input2, expected_output2);
	}
	auto output_after1 = network.calculateOutput(input1);
	auto output_after2 = network.calculateOutput(input2);

	std::cout << "Before: " << output_before1[0] << " " << output_before2[0] << " after: " << output_after1[0] << " " << output_after2[0];

	getchar();

	return 0;
}
