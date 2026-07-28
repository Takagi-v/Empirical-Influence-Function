package main


func makeComparableValue(val interface{}) (typedValue, error) {
	if typ, isType := val.(reflect.Type); isType { 		return typedValue{
			typ:   reflectTypeType,
			value: typ,
		}, nil 	}
	vv := reflect.ValueOf(val)
	if err := checkNotNil(vv); err != nil {
		return typedValue{}, err
	}
	typ := vv.Type()
	switch {
	case isIntKind(typ.Kind()), isUintKind(typ.Kind()):
		// We need to allocate a new pointer.
		compType := getComparableType(typ)
		vvNew := reflect.New(vv.Type())
		vvNew.Elem().Set(vv)
		return typedValue{
			typ:   vv.Type(),
			value: vvNew.Convert(reflect.PointerTo(compType)).Interface(),
		}, nil
	case typ.Kind() == reflect.Ptr:
		switch {
		case isIntKind(typ.Elem().Kind()), isUintKind(typ.Elem().Kind()):
			compType := getComparableType(typ.Elem())
			return typedValue{
				typ:   vv.Type().Elem(),
				value: vv.Convert(reflect.PointerTo(compType)).Interface(),
			}, nil
		case typ.Elem().Kind() == reflect.Struct:
			return typedValue{
				typ:   vv.Type(),
				value: val,
			}, nil
		default:
			return typedValue{}, errors.Errorf(
				"unsupported pointer kind %v for type %T", typ.Elem().Kind(), val,
			)
		}
	default:
		return typedValue{}, errors.Errorf(
			"unsupported kind %v for type %T", typ.Kind(), val,
		)
	}
}