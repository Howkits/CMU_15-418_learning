#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;

void absSerial(float *values, float *output, int N)
{
	for (int i = 0; i < N; i++)
	{
		float x = values[i];
		if (x < 0)
		{
			output[i] = -x;
		}
		else
		{
			output[i] = x;
		}
	}
}

// implementation of absolute value using 15418 instrinsics
void absVector(float *values, float *output, int N)
{
	__cmu418_vec_float x;
	__cmu418_vec_float result;
	__cmu418_vec_float zero = _cmu418_vset_float(0.f);
	__cmu418_mask maskAll, maskWork, maskIsNegative, maskIsNotNegative, maskIsNotNegativeAndWork;

	//  Note: Take a careful look at this loop indexing.  This example
	//  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
	//  Why is that the case?
	for (int i = 0; i < N; i += VECTOR_WIDTH)
	{

		// All ones
		maskAll = _cmu418_init_ones(); // all to one  default=8

		// All zeros
		maskIsNegative = _cmu418_init_ones(0); //all to zero

		if ((N - i) >= VECTOR_WIDTH)
			maskWork = maskAll;
		else
		{
			maskWork = _cmu418_init_ones(N - i);
			//printf("maskwork:%d\n", _cmu418_cntbits(maskWork));
		}

		// Load vector of values from contiguous memory addresses
		_cmu418_vload_float(x, values + i, maskWork); // x = values[i]; maskall update all

		// Set mask according to predicate
		_cmu418_vlt_float(maskIsNegative, x, zero, maskWork); // if (x < 0) {

		//printf("maskisNegative in step%d:%d\n", i / VECTOR_WIDTH, _cmu418_cntbits(maskIsNegative));

		// Execute instruction using mask ("if" clause)
		_cmu418_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

		// Inverse maskIsNegative to generate "else" mask
		maskIsNotNegative = _cmu418_mask_not(maskIsNegative); // } else {

		//maskIsNotNegativeAndWork = _cmu418_mask_and(maskWork, maskIsNotNegative);
		//printf("maskisNotNegativeAndWork in step%d:%d\n", i / VECTOR_WIDTH, _cmu418_cntbits(maskIsNotNegativeAndWork));

		// Execute instruction ("else" clause)
		_cmu418_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

		// Write results back to memory
		_cmu418_vstore_float(output + i, result, maskWork);
	}
}

// Accepts an array of values and an array of exponents
// For each element, compute values[i]^exponents[i] and clamp value to
// 4.18.  Store result in outputs.
// Uses iterative squaring, so that total iterations is proportional
// to the log_2 of the exponent
void clampedExpSerial(float *values, int *exponents, float *output, int N)
{
	for (int i = 0; i < N; i++)
	{
		float x = values[i];
		float result = 1.f;
		int y = exponents[i];
		float xpower = x;
		while (y > 0)
		{
			if (y & 0x1)
				result *= xpower;
			xpower = xpower * xpower;
			y >>= 1;
		}
		if (result > 4.18f)
		{
			result = 4.18f;
		}
		output[i] = result;
	}
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
	__cmu418_vec_float x;
	__cmu418_vec_float result;
	__cmu418_vec_float xpower;
	__cmu418_vec_float float_zero = _cmu418_vset_float(0.f);
	__cmu418_vec_float float_418 = _cmu418_vset_float(4.18f);
	__cmu418_vec_int int_zero = _cmu418_vset_int(0);
	__cmu418_vec_int y;
	__cmu418_vec_int yAndOne;
	__cmu418_vec_int One = _cmu418_vset_int(0x1);
	__cmu418_mask maskAll = _cmu418_init_ones(), maskWork, maskForY = _cmu418_init_ones(0), maskForResult = _cmu418_init_ones(0);
	__cmu418_mask maskYZero = _cmu418_init_ones(0), maskYNotZero;

	for (int i = 0; i < N; i += VECTOR_WIDTH)
	{
		if ((N - i) >= VECTOR_WIDTH)
			maskWork = maskAll;
		else
		{
			maskWork = _cmu418_init_ones(N - i);
		}

		_cmu418_vload_float(x, values + i, maskWork);  //float x = values[i];
		result = _cmu418_vset_float(1.f);			   //float result = 1.f;
		_cmu418_vload_int(y, exponents + i, maskWork); //int y = exponents[i];
		_cmu418_vmove_float(xpower, x, maskWork);	   //float xpower = x;
		_cmu418_vgt_int(maskForY, y, int_zero, maskWork);

		while (_cmu418_cntbits(maskForY) > 0) //while(y>0)
		{

			_cmu418_vbitand_int(yAndOne, y, One, maskWork); //if (y & 0x1)
			_cmu418_veq_int(maskYZero, yAndOne, int_zero, maskWork);
			maskYNotZero = _cmu418_mask_not(maskYZero);
			_cmu418_vmult_float(result, result, xpower, maskYNotZero); //result *= xpower;

			_cmu418_vmult_float(xpower, xpower, xpower, maskWork); //xpower = xpower * xpower;
			_cmu418_vshiftright_int(y, y, One, maskWork);		   //y >>= 1;

			_cmu418_vgt_int(maskForY, y, int_zero, maskWork);
		}

		_cmu418_vgt_float(maskForResult, result, float_418, maskWork); //if (result > 4.18f)
		_cmu418_vmove_float(result, float_418, maskForResult);		   //result = 4.18f;

		_cmu418_vstore_float(output + i, result, maskWork); //output[i] = result;
	}
}

float arraySumSerial(float *values, int N)
{
	float sum = 0;
	for (int i = 0; i < N; i++)
	{
		sum += values[i];
	}

	return sum;
}

// Assume N % VECTOR_WIDTH == 0
// Assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
	__cmu418_vec_float sum = _cmu418_vset_float(0.f);
	__cmu418_vec_float x;
	__cmu418_mask maskAll = _cmu418_init_ones();

	for (int i = 0; i < N; i += VECTOR_WIDTH)
	{
		_cmu418_vload_float(x, values + i, maskAll);
		_cmu418_vadd_float(sum, sum, x, maskAll);
	}

	for (int i = 0; pow(2.0, double(i)) < VECTOR_WIDTH; i++)
	{
		_cmu418_hadd_float(sum, sum);
		_cmu418_interleave_float(sum, sum);
	}

	float ans[VECTOR_WIDTH];
	_cmu418_vstore_float(ans, sum, maskAll);

	return ans[0];
}
