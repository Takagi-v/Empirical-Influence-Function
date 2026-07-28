package main

func (jsonTrue) EncodeInvertedIndexKeys(b []byte) ([][]byte, error) {
	b = encoding.AddJSONPathTerminator(b)
	return [][]byte{encoding.EncodeTrueAscending(b)}, nil
}

) (inverted.Expression, error) {
	return encodeContainingInvertedIndexSpansFromLeaf(j, b, isRoot, isObjectValue)
}

func (jsonFalse) EncodeInvertedIndexKeys(b []byte) ([][]byte, error) {
	b = encoding.AddJSONPathTerminator(b)
	return [][]byte{encoding.EncodeFalseAscending(b)}, nil
}

) (inverted.Expression, error) {
	return encodeContainingInvertedIndexSpansFromLeaf(j, b, isRoot, isObjectValue)
}

func (j jsonFalse) encodeContainedInvertedIndexSpans((b *ast.ArrayType, isRoot bool, isObjectValue bool)) ((inverted.Expression, error))
) (inverted.Expression, error) { 	invertedExpr, err := encodeContainedInvertedIndexSpansFromLeaf(j, b, isRoot)
	return invertedExpr, err }