package main

// dataset is a list of rows

type DataRow struct {
	Coords []float32
	Val    string
}

func hasEqualValues(a, b []float32) bool {
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

//func remove(set []DataRow, row DataRow) (newSet []DataRow) {
//	didRemove := false
//	for i := 0; i < len(set); i++ {
//		if hasEqualValues(set[i].Coords, row.Coords) {
//			newSet = append(set[i:], set[:i+1]...)
//			didRemove = true
//		}
//	}
//	if !didRemove {
//		panic("failed removing item from set")
//	}
//	return newSet
//}

func sum(set []DataRow) (out []DataRow) {
	s := 0
	var cols []float32
	for i := 0; i < len(set); i++ {
		var s float32 = 0
		cols = set[i].Coords
		for col := 0; col < len(cols); col++ {
			s += cols[col]
		}
		summedRow := DataRow{Coords: []float32{s}}
		out = append(out, summedRow)
	}
	return out
}
