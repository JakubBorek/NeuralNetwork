#pragma once
#include <array>
#include "neurons/INeuron.h"
#include "random/IRandomGenerator.h"

namespace neural_network
{
template <template<unsigned> typename NeuronTemplate, unsigned NeuronCount, unsigned InputCount>
class NeuronLayer
{
	typedef NeuronTemplate<InputCount> NeuronType;
	typedef typename NeuronType::value_type value_type;
	static_assert(std::is_base_of<neurons::INeuron<InputCount, value_type>, NeuronType>::value, "");
	std::array<NeuronType, NeuronCount> m_neurons;
public:
	std::array<value_type, NeuronCount> calculateOutput(const std::array<value_type, InputCount>& input) const;
	std::array<value_type, InputCount> propagateError(std::array<value_type, NeuronCount> errors) const;
	void compensateForError(std::array<value_type, NeuronCount> error, const std::array<value_type, InputCount>& input);
	void randomizeWeights(IRandomGenerator<typename NeuronType::weight_type>& random_generator);
private:
	template<class ActionType> void forEachNeuronIndex(const ActionType& action) const;
};

template <template <unsigned> class NeuronTemplate, unsigned NeuronCount, unsigned InputCount>
std::array<typename NeuronLayer<NeuronTemplate, NeuronCount, InputCount>::value_type, NeuronCount> NeuronLayer<NeuronTemplate, NeuronCount, InputCount>::calculateOutput(const std::array<value_type, InputCount>& input) const
{
	std::array<value_type, NeuronCount> output;
	forEachNeuronIndex([&](unsigned i)
	{
		output[i] = m_neurons[i].calculateOutput(input);		
	});
	return output;
}

template <template <unsigned> class NeuronTemplate, unsigned NeuronCount, unsigned InputCount>
std::array<typename NeuronLayer<NeuronTemplate, NeuronCount, InputCount>::value_type, InputCount> NeuronLayer<NeuronTemplate, NeuronCount, InputCount>::propagateError(std::array<value_type, NeuronCount> errors) const
{
	auto propagated_error = std::array<value_type, InputCount>();
	for (unsigned i = 0; i < NeuronCount; i++)
	{
		auto& neuron = m_neurons[i];
		auto& error = errors[i];
		auto weights = neuron.getInputWeights();
		std::transform(propagated_error.begin(), propagated_error.end(), weights.begin(), propagated_error.begin(), [&](const auto& sum, const auto& w)
		               {
			               return sum + w * error;
		               });
	}
	return propagated_error;
}

template <template <unsigned> class NeuronTemplate, unsigned NeuronCount, unsigned InputCount>
void NeuronLayer<NeuronTemplate, NeuronCount, InputCount>::compensateForError(std::array<value_type, NeuronCount> error, const std::array<value_type, InputCount>& input)
{
	forEachNeuronIndex([&](unsigned i)
	{
		m_neurons[i].compensateForError(error[i], input);
	});
}

template <template <unsigned> class NeuronTemplate, unsigned NeuronCount, unsigned InputCount>
void NeuronLayer<NeuronTemplate, NeuronCount, InputCount>::randomizeWeights(IRandomGenerator<typename NeuronType::weight_type>& random_generator)
{
	forEachNeuronIndex([&](unsigned i)
	{
		m_neurons[i].randomizeWeights(random_generator);
	});
}

template <template <unsigned> class NeuronTemplate, unsigned NeuronCount, unsigned InputCount>
template <class ActionType>
void NeuronLayer<NeuronTemplate, NeuronCount, InputCount>::forEachNeuronIndex(const ActionType& action) const
{
	for (unsigned i = 0; i < NeuronCount; i++)
	{
		action(i);
	}
}
}
