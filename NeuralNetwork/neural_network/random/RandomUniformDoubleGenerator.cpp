#include "RandomUniformDoubleGenerator.h"

double neural_network::RandomUniformDoubleGenerator::getNext()
{
	return m_distribution(m_engine);
}