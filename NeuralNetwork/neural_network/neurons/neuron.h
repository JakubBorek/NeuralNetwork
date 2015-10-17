#pragma once
#include "INeuron.h"
#include "../random/IRandomGenerator.h"

namespace neural_network
{
namespace neurons
{
template <unsigned NumberOfInputs, class ActivationFunction, typename ValueType = double, typename WeightType = double>
class Neuron : public INeuron<NumberOfInputs, ValueType>
{
public:
	static constexpr WeightType learning_coefficient = 0.1;
	static constexpr unsigned number_of_inputs = NumberOfInputs;
	typedef ValueType value_type;
	typedef WeightType weight_type;
private:
	std::array<WeightType, NumberOfInputs> m_input_weights;
	WeightType m_bias_weight;
public:
	Neuron() = default;
	virtual ValueType calculateOutput(const std::array<ValueType, NumberOfInputs>& input) const override;
	const std::array<double, NumberOfInputs>& getInputWeights() const;
	void compensateForError(ValueType error, std::array<ValueType, NumberOfInputs> input);
	void randomizeWeights(IRandomGenerator<WeightType>& random_generator);
private:
	ValueType calculateWeightedSum(const std::array<ValueType, NumberOfInputs>& input) const;
	template<class ActionType> void forEachInputIndex(const ActionType& action) const;
};
}}


template <unsigned NumberOfInputs, class ActivationFunction,typename ValueType, typename WeightType>
inline ValueType neural_network::neurons::Neuron<NumberOfInputs, ActivationFunction, ValueType, WeightType>::calculateOutput(const std::array<ValueType, NumberOfInputs>& input) const
{
	auto sum = calculateWeightedSum(input);
	auto output = ActivationFunction::calculate(sum);
	return output;
}

template <unsigned NumberOfInputs, class ActivationFunction,typename ValueType, typename WeightType>
const std::array<double, NumberOfInputs>& neural_network::neurons::Neuron<NumberOfInputs, ActivationFunction, ValueType, WeightType>::getInputWeights() const
{
	return m_input_weights;
}

template <unsigned NumberOfInputs, class ActivationFunction,typename ValueType, typename WeightType>
void neural_network::neurons::Neuron<NumberOfInputs, ActivationFunction, ValueType, WeightType>::compensateForError(ValueType error, std::array<ValueType, NumberOfInputs> input)
{
	auto sum = calculateWeightedSum(input);
	auto change_coefficient = learning_coefficient * error * ActivationFunction::calculateDerivative(sum);
	m_bias_weight += change_coefficient;
	forEachInputIndex([&](unsigned i) {m_input_weights[i] += change_coefficient * input[i]; });
}

template <unsigned NumberOfInputs, class ActivationFunction, typename ValueType, typename WeightType>
void neural_network::neurons::Neuron<NumberOfInputs, ActivationFunction, ValueType, WeightType>::randomizeWeights(IRandomGenerator<WeightType>& random_generator)
{
	m_bias_weight = random_generator.getNext();
	forEachInputIndex([&](unsigned i) {m_input_weights[i] = random_generator.getNext(); });
}

template <unsigned NumberOfInputs, class ActivationFunction,typename ValueType, typename WeightType>
ValueType neural_network::neurons::Neuron<NumberOfInputs, ActivationFunction, ValueType, WeightType>::calculateWeightedSum(const std::array<ValueType, NumberOfInputs>& input) const
{
	auto sum = ValueType(1)*m_bias_weight;
	forEachInputIndex([&](unsigned i) {sum += m_input_weights[i] * input[i];});
	return sum;
}

template <unsigned NumberOfInputs, class ActivationFunction, typename ValueType, typename WeightType>
template <class ActionType>
void neural_network::neurons::Neuron<NumberOfInputs, ActivationFunction, ValueType, WeightType>::forEachInputIndex(const ActionType& action) const
{
	for (unsigned i = 0; i < NumberOfInputs; i++)
	{
		action(i);
	}
}