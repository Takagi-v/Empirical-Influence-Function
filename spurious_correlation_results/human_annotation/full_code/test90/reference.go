package main

func (b *logicalPropsBuilder) addFiltersToFuncDep(filters FiltersExpr, fdset *props.FuncDepSet) {
	for i := range filters {
		filterProps := filters[i].ScalarProps()
		if buildutil.CrdbTestBuild && !filterProps.Populated {
			panic(errors.AssertionFailedf("filter properties not populated"))
		}
		fdset.AddFrom(&filterProps.FuncDeps)
	}

	if len(filters) <= 1 {
		return
	}

	// Some columns can only be determined to be constant from multiple
	// constraints (e.g. x <= 1 AND x >= 1); we intersect the constraints and
	// extract const columns from the intersection. But intersection is expensive
	// so we first do a quick check to rule out cases where each constraint refers
	// to a different set of columns.
	var cols opt.ColSet
	possibleIntersection := false
	for i := range filters {
		if c := filters[i].ScalarProps().Constraints; c != nil {
			s := c.ExtractCols()
			if cols.Intersects(s) {
				possibleIntersection = true
				break
			}
			cols.UnionWith(s)
		}
	}

	if possibleIntersection {
		intersection := constraint.Unconstrained
		for i := range filters {
			if c := filters[i].ScalarProps().Constraints; c != nil {
				intersection = intersection.Intersect(b.sb.ctx, b.evalCtx, c)
			}
		}
		constCols := intersection.ExtractConstCols(b.sb.ctx, b.evalCtx)
		fdset.AddConstants(constCols)
	}
}

) {
	for i := range filters {
		filterProps := filters[i].ScalarProps()
		if filterProps.Constraints == nil {
			continue
		}

		for j, n := 0, filterProps.Constraints.Length(); j < n; j++ {
			c := filterProps.Constraints.Constraint(j)
			b.updateCardinalityFromConstraint(c, rel)
		}
	}
}

) {
	cols := c.Columns.ColSet()
	if !rel.FuncDeps.ColsAreLaxKey(cols) {
		return
	}

	count, ok := c.CalculateMaxResults(b.sb.ctx, b.evalCtx, cols, rel.NotNullCols)
	if ok && count < math.MaxUint32 {
		rel.Cardinality = rel.Cardinality.Limit(uint32(count))
	}
}

func (b *logicalPropsBuilder) updateCardinalityFromTypes(cols opt.ColSet, rel *props.Relational) {
	cols.ForEach(func(col opt.ColumnID) {
		// We need to check if this column is a strict key, since a lax key could
		// include an arbitrary number of null values.
		if !rel.FuncDeps.ColsAreStrictKey(opt.MakeColSet(col)) {
			return
		}

		md := b.mem.Metadata()
		count, ok := distinctCountFromType(md, md.ColumnMeta(col).Type)
		if ok && count < math.MaxUint32 {
			if !rel.NotNullCols.Contains(col) {
				// Add one for a possible null value.
				count++
			}
			rel.Cardinality = rel.Cardinality.Limit(uint32(count))
		}
	})
}

func distinctCountFromType(md *opt.Metadata, typ *types.T) (_ uint64, ok bool) {
	// TODO(rytaft): Support other limited types such as INT2, BIT(N), VARBIT(N),
	// CHAR(N), and VARCHAR(N).
	switch typ.Family() {
	case types.BoolFamily:
		// There are maximum two distinct values: true and false.
		return 2, true

	case types.EnumFamily:
		typOid := typ.Oid()
		var hydrated *types.T
		// Find the hydrated type in the metadata.
		for _, t := range md.AllUserDefinedTypes() {
			if t.Oid() == typOid {
				hydrated = t
				break
			}
		}
		if hydrated == nil {
			// This can happen in rare cases if the user defined type is
			// contained in an array.
			// TODO(rytaft): This should really be an assertion failure. See #67434.
			break
		}
		// Enum types have a well defined set of values.
		return uint64(len(hydrated.TypeMeta.EnumData.PhysicalRepresentations)), true
	}

	return 0, false
}


func ensureLookupJoinInputProps(join *LookupJoinExpr, sb *statisticsBuilder) *props.Relational {
	relational := &join.lookupProps
	if relational.OutputCols.Empty() {
		md := sb.mem.Metadata()
		relational.OutputCols = join.Cols.Difference(join.Input.Relational().OutputCols)

		// Include the key columns in the output columns.
		index := md.Table(join.Table).Index(join.Index)
		for i := range join.KeyCols {
			indexColID := join.Table.ColumnID(index.Column(i).Ordinal())
			relational.OutputCols.Add(indexColID)
		}

		// Include columns from the join condition in the output columns.
		lookupExprCols := join.LookupExpr.OuterCols()
		for i, n := 0, index.KeyColumnCount(); i < n; i++ { 			indexColID := join.Table.ColumnID(index.Column(i).Ordinal())
			if lookupExprCols.Contains(indexColID) {
				relational.OutputCols.Add(indexColID)
			} 		}

		relational.NotNullCols = makeTableNotNullCols(md, join.Table).Copy()
		relational.NotNullCols.IntersectionWith(relational.OutputCols)
		relational.Cardinality = props.AnyCardinality
		relational.FuncDeps.CopyFrom(MakeTableFuncDep(md, join.Table))
		relational.FuncDeps.ProjectCols(relational.OutputCols)
		*relational.Statistics() = *sb.makeTableStatistics(join.Table)
	}
	return relational
}