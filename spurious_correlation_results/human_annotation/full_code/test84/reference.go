package main

) redact.RedactableString {
	fs := FmtSimple | FmtAlwaysQualifyTableNames | FmtMarkRedactionNode
	if !redactNamesInSQLStatementLog.Get(sv) {
		fs = fs | FmtOmitNameRedaction
	}
	return formatStmtKeyAsRedactableString(statement, annotations, fs)
}

func formatStmtKeyAsRedactableString((rootAST Statement, ann *Annotations, fs FmtFlags)) redact.RedactableString
) redact.RedactableString {
	f := NewFmtCtx(
		fs, 		FmtAnnotations(ann),
	)
	f.FormatNode(rootAST)
	formattedRedactableStatementString := f.CloseAndGetString() 	return redact.RedactableString(formattedRedactableStatementString)
}