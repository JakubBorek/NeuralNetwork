#pragma once
#include <cmath>

namespace neural_network
{
namespace transfer_functions
{
template <typename T = double>
class FastSigmoidTransferFunction
{
public:
	static T calculate(T input);
	static T calculateDerivative(T input);
	
};

template <typename T>
T FastSigmoidTransferFunction<T>::calculate(T input)
{
	return input / (1 + abs(input));
}

template <typename T>
T FastSigmoidTransferFunction<T>::calculateDerivative(T input)
{
	auto absX = abs(input);
	auto onePlusAbsX = 1 + absX;
	return 1 / (onePlusAbsX*onePlusAbsX);
}
}}
