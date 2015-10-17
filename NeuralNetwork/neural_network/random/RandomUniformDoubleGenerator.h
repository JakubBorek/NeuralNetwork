#pragma once
#include <random>
#include "IRandomGenerator.h"

namespace neural_network
{
class RandomUniformDoubleGenerator : public IRandomGenerator<double>
{
	std::uniform_real_distribution<double> m_distribution{-1,1};
	std::mt19937 m_engine;
public:
	double getNext() override;
};	
}
