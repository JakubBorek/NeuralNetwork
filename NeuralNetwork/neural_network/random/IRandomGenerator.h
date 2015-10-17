#pragma once

namespace neural_network
{
template<typename FloatType>
struct IRandomGenerator
{
	virtual FloatType getNext() = 0;
	virtual ~IRandomGenerator() = default;
};
}
