#pragma once
#include <array>

namespace neural_network
{
namespace neurons
{
template <unsigned NumberOfInputs, typename ValueType = double>
class INeuron
{
public:
	virtual ValueType calculateOutput(const std::array<ValueType, NumberOfInputs>& input) const = 0;
	virtual ~INeuron() = default;
};
}}
