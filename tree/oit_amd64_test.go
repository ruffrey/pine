package main

import (
	"math/rand"
	"testing"
)

func BenchmarkOit(b *testing.B) {
	var count float32
	var a float32
	var v float32
	for i := 0; i < b.N; i++ {
		a = rand.Float32()
		v = rand.Float32()
		count += oneIfTrue(a, v)
	}
}

func BenchmarkIf(b *testing.B) {
	var count float32
	var a float32
	var v float32
	for i := 0; i < b.N; i++ {
		a = rand.Float32()
		v = rand.Float32()
		if a == v {
			count++
		}
	}
}
