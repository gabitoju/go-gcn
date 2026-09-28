package data

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// SparseMatrix is a compressed sparse row matrix. It is intended for graph
// adjacency matrices, where the number of edges is much smaller than N².
type SparseMatrix struct {
	rows       int
	cols       int
	rowOffsets []int
	colIndices []int
	values     []float64
}

func NewSparseMatrix(rows, cols int, entries [][2]int) *SparseMatrix {
	rowValues := make([]map[int]float64, rows)
	for _, entry := range entries {
		r, c := entry[0], entry[1]
		if r < 0 || r >= rows || c < 0 || c >= cols {
			panic("sparse matrix entry out of bounds")
		}
		if rowValues[r] == nil {
			rowValues[r] = make(map[int]float64)
		}
		rowValues[r][c] = 1
	}
	return newSparseMatrix(rows, cols, rowValues)
}

func newSparseMatrix(rows, cols int, rowValues []map[int]float64) *SparseMatrix {
	rowOffsets := make([]int, rows+1)
	for r := 0; r < rows; r++ {
		rowOffsets[r+1] = rowOffsets[r] + len(rowValues[r])
	}
	colIndices := make([]int, rowOffsets[rows])
	values := make([]float64, rowOffsets[rows])
	for r := 0; r < rows; r++ {
		cols := make([]int, 0, len(rowValues[r]))
		for c := range rowValues[r] {
			cols = append(cols, c)
		}
		sortInts(cols)
		for i, c := range cols {
			idx := rowOffsets[r] + i
			colIndices[idx] = c
			values[idx] = rowValues[r][c]
		}
	}
	return &SparseMatrix{rows: rows, cols: cols, rowOffsets: rowOffsets, colIndices: colIndices, values: values}
}

func sortInts(values []int) {
	for i := 1; i < len(values); i++ {
		for j := i; j > 0 && values[j] < values[j-1]; j-- {
			values[j], values[j-1] = values[j-1], values[j]
		}
	}
}

func (s *SparseMatrix) Dims() (int, int) {
	return s.rows, s.cols
}

func (s *SparseMatrix) NonZeroCount() int {
	return len(s.values)
}

func (s *SparseMatrix) NormalizedWithSelfLoops() *SparseMatrix {
	if s.rows != s.cols {
		panic("adjacency matrix must be square")
	}
	degree := make([]float64, s.rows)
	for r := 0; r < s.rows; r++ {
		degree[r] = 1
		for i := s.rowOffsets[r]; i < s.rowOffsets[r+1]; i++ {
			degree[r] += s.values[i]
		}
	}
	rows := make([]map[int]float64, s.rows)
	for r := 0; r < s.rows; r++ {
		rows[r] = make(map[int]float64, s.rowOffsets[r+1]-s.rowOffsets[r]+1)
		for i := s.rowOffsets[r]; i < s.rowOffsets[r+1]; i++ {
			c := s.colIndices[i]
			rows[r][c] += s.values[i] / math.Sqrt(degree[r]*degree[c])
		}
		rows[r][r] += 1 / degree[r]
	}
	return newSparseMatrix(s.rows, s.cols, rows)
}

func (s *SparseMatrix) MulDense(dst, src *mat.Dense) {
	rows, cols := src.Dims()
	if rows != s.cols {
		panic("sparse matrix multiplication shape mismatch")
	}
	dstRows, dstCols := dst.Dims()
	if dstRows != s.rows || dstCols != cols {
		panic("sparse matrix destination shape mismatch")
	}
	dst.Zero()
	for r := 0; r < s.rows; r++ {
		out := dst.RawRowView(r)
		for i := s.rowOffsets[r]; i < s.rowOffsets[r+1]; i++ {
			value := s.values[i]
			in := src.RawRowView(s.colIndices[i])
			for c := range out {
				out[c] += value * in[c]
			}
		}
	}
}

func (s *SparseMatrix) TransposeMulDense(dst, src *mat.Dense) {
	rows, cols := src.Dims()
	if rows != s.rows {
		panic("sparse transpose multiplication shape mismatch")
	}
	dstRows, dstCols := dst.Dims()
	if dstRows != s.cols || dstCols != cols {
		panic("sparse transpose destination shape mismatch")
	}
	dst.Zero()
	for r := 0; r < s.rows; r++ {
		in := src.RawRowView(r)
		for i := s.rowOffsets[r]; i < s.rowOffsets[r+1]; i++ {
			out := dst.RawRowView(s.colIndices[i])
			for c := range out {
				out[c] += s.values[i] * in[c]
			}
		}
	}
}
