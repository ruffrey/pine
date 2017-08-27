package main

var _kernelSource = `
__kernel void square(
   __global float* input,
   __global float* output,
   const unsigned int count)
{
   int i = get_global_id(0);
   if(i < count)
       output[i] = input[i] * input[i];
}
`

// this function takes up about 91 - 98% of cpu burn.
func withValue(lastColIndex int, value float32, splitGroup []datarow) (count float32) {
	splitGroupLen := len(splitGroup)
	var prediction float32
	for i := 0; i < splitGroupLen; i++ {
		prediction = splitGroup[i][lastColIndex]
		if prediction == value {
			count++
		}
	}

	return count
}
