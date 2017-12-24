// +build !amd64,gc

package asm_examples

// this function takes up about 91 - 98% of cpu burn.
func withValue(lastColIndex int, value float32, splitGroup []datarow) (count float32) {
	splitGroupLen := len(splitGroup)
	var prediction float32
	for i := 0; i < splitGroupLen; i++ {
		prediction = splitGroup[i][lastColIndex]
		count += oneIfTrue(prediction, value)
	}

	return count
}
