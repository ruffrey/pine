package asm_examples

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOit(t *testing.T) {
	t.Run("returns 1 when the same", func(t *testing.T) {
		var x float32 = 11
		var y float32 = 11
		var expected float32 = 1
		var actual float32 = 9999

		actual = oneIfTrue(x, y)
		assert.Equal(t, expected, actual)

		x = 54
		y = 54
		actual = oneIfTrue(x, y)
		assert.Equal(t, expected, actual)
	})
	t.Run("returns 0 when the different", func(t *testing.T) {
		var x float32 = 23
		var y float32 = 19
		var expected float32 = 0
		var actual float32 = 9999

		actual = oneIfTrue(x, y)
		assert.Equal(t, expected, actual)

		x = 0
		y = 3
		actual = oneIfTrue(x, y)
		assert.Equal(t, expected, actual)
	})
}

func BenchmarkOit(b *testing.B) {
	b.Run("oneIfTrue", func(b *testing.B) {
		var count float32
		var a float32
		var v float32
		for i := 0; i < b.N; i++ {
			a = rand.Float32()
			v = rand.Float32()
			count += oneIfTrue(a, v)
		}
	})
	b.Run("regular conditional", func(b *testing.B) {
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
	})
}
