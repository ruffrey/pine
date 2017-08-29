package main

// splitCache could just as well not exist, but it makes it effecient to reuse
// memory for these large arrays it carries around. It uses far fewer resources
// due to not invoking malloc and gc - instead we just reuse it in this little
// cache, but clear everything out each time.
type splitCache struct {
	left          []datarow
	right         []datarow
	leftLastCols  []float32
	rightLastCols []float32
}

/*
splitOnIndex splits a dataset based on an attribute and an attribute value

We also extract the last columns for the left and right splits, because they will
come in handy to speed up the really hot path in `withValue`.

This *modifies* the arrays passed in - saves hugely on memory allocation.

test_split
*/
func (sc *splitCache) splitOnIndex(index int32, value float32, dataSubset []datarow) {
	// keep garbage collection from cleaning up leftLastCols and rightLastCols
	// without this it is actually slower to just get rid of them entirely
	// just overwrite leftLastCols and rightLastCols values where needed
	//leftLastCols = leftLastCols[:cap(leftLastCols)]
	//rightLastCols = rightLastCols[:cap(rightLastCols)]
	sc.left = sc.left[:0]
	sc.right = sc.right[:0]
	sc.leftLastCols = sc.leftLastCols[:0]
	sc.rightLastCols = sc.rightLastCols[:0]

	for _, row := range dataSubset {
		// last column has same index as the original row
		if row[index] < value {
			sc.left = append(sc.left, row)
			sc.leftLastCols = append(sc.leftLastCols, row[lastColumnIndex])
		} else {
			sc.right = append(sc.right, row)
			sc.rightLastCols = append(sc.rightLastCols, row[lastColumnIndex])
		}
	}
}
