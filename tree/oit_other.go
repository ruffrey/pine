// +build !amd64,gc

package main

func oneIfTrue(x, y float32) float32 {
	if x == y {
		return 1
	}
	return 0
}
