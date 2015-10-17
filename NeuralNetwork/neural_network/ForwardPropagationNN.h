#pragma once
#include "NeuronLayer.h"

namespace neural_network
{
//ToDo: Variadic template layers
template <template<unsigned> class NeuronTemplate, unsigned InputSize, unsigned HiddenLayerSize, unsigned OutputLayerSize>
class ForwardPropagationNN
{
	typedef typename NeuronTemplate<0>::value_type value_type;
	typedef typename NeuronTemplate<0>::weight_type weight_type;
	NeuronLayer<NeuronTemplate, HiddenLayerSize, InputSize> m_hidden_layer;
	NeuronLayer<NeuronTemplate, OutputLayerSize, HiddenLayerSize> m_output_layer;

public:
	std::array<value_type, OutputLayerSize> calculateOutput(const std::array<value_type, InputSize>& input) const;
	void train(const std::array<value_type, InputSize>& input, const std::array<value_type, OutputLayerSize>& expected_output);
	void randomizeWeights(IRandomGenerator<typename weight_type>& random_generator);
};

template <template <unsigned> class NeuronTemplate, unsigned InputSize, unsigned HiddenLayerSize, unsigned OutputLayerSize>
std::array<typename ForwardPropagationNN<NeuronTemplate, InputSize, HiddenLayerSize, OutputLayerSize>::value_type, OutputLayerSize> ForwardPropagationNN<NeuronTemplate, InputSize, HiddenLayerSize, OutputLayerSize>::calculateOutput(const std::array<value_type, InputSize>& input) const
{
	auto hidden_layer_output = m_hidden_layer.calculateOutput(input);
	auto output = m_output_layer.calculateOutput(hidden_layer_output);
	return output;
}

template <template <unsigned> class NeuronTemplate, unsigned InputSize, unsigned HiddenLayerSize, unsigned OutputLayerSize>
void ForwardPropagationNN<NeuronTemplate, InputSize, HiddenLayerSize, OutputLayerSize>::train(const std::array<value_type, InputSize>& input, const std::array<value_type, OutputLayerSize>& expected_output)
{
	auto hidden_layer_output = m_hidden_layer.calculateOutput(input);
	auto output = m_output_layer.calculateOutput(hidden_layer_output);
	std::array<value_type, OutputLayerSize> output_error;
	for (unsigned i = 0; i < OutputLayerSize; i++)
	{
		output_error[i] = expected_output[i] - output[i];
	}
	auto hidden_layer_error = m_output_layer.propagateError(output_error);
	m_output_layer.compensateForError(output_error, hidden_layer_output);
	m_hidden_layer.compensateForError(hidden_layer_error, input);
}

template <template <unsigned> class NeuronTemplate, unsigned InputSize, unsigned HiddenLayerSize, unsigned OutputLayerSize>
void ForwardPropagationNN<NeuronTemplate, InputSize, HiddenLayerSize, OutputLayerSize>::randomizeWeights(IRandomGenerator<weight_type>& random_generator)
{
	m_hidden_layer.randomizeWeights(random_generator);
	m_output_layer.randomizeWeights(random_generator);
}
}
