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
		fs, 		InitialAnnotationString: ann.String(),
	).SetContentType(ContentTypeSQL) 	return redact.RedactableString(formattedRedactableStatementString)
}