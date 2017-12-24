package asm_examples

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestWithValue(t *testing.T) {
	t.Run("returns total rows with value in last column", func(t *testing.T) {
		rows := make([]datarow, 5)
		rows[0] = datarow{3, 4, 5, 3}
		rows[1] = datarow{3, 4, 5, 2}
		rows[2] = datarow{3, 4, 5, 9}
		rows[3] = datarow{3, 4, 5, 3}
		rows[4] = datarow{3, 4, 5, 3}

		var expected float32 = 3 // 3 instances of 3 in last row
		actual := withValue(4, 3, rows)
		assert.Equal(t, expected, actual)
	})
}

func BenchmarkWithValue(b *testing.B) {
	rows := make([]datarow, 5)
	rows[0] = datarow{3, 4, 5, 3}
	rows[1] = datarow{3, 4, 5, 2}
	rows[2] = datarow{3, 4, 5, 9}
	rows[3] = datarow{3, 4, 5, 3}
	rows[4] = datarow{3, 4, 5, 3}

	b.Run("asm withValue", func(b *testing.B) {
		var count float32
		for i := 0; i < b.N; i++ {
			count = withValue(4, 3, rows)
		}
		assert.Equal(b, 3, count)
	})
	b.Run("regular withValue", func(b *testing.B) {
		var count float32
		for i := 0; i < b.N; i++ {
			count = wv_other(4, 3, rows)
		}
		assert.Equal(b, 3, count)
	})
}

func wv_other(lastColIndex int, value float32, splitGroup []datarow) (count float32) {
	var prediction float32
	lenSpl := len(splitGroup)
	for i := 0; i < lenSpl; i++ {
		prediction = splitGroup[i][lastColIndex]
		count += oneIfTrue(prediction, value)
	}

	return count
}
