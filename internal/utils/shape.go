package utils

import (
	"reflect"
)

func Shape(data interface{}) []int {
	shape := getShape(data, []int{})
	return shape
}

func getShape(data interface{}, shape []int) []int {
	v := reflect.ValueOf(data)

	if v.Kind() == reflect.Slice {
		shape = append(shape, v.Len())
		if v.Len() > 0 {
			shape = getShape(v.Index(0).Interface(), shape)
		}
	}

	return shape
}
