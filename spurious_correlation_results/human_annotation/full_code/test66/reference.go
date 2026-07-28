package main

func (p *Parser) setComments(e Expr, comments CommentsExpr) {
	if len(comments) > 0 {
		p.exprComments[e] = comments
	}
}

func (p *Parser) hasComments() bool {
	return len(p.comments) > 0
}

func (p *Parser) appendComments() {
	comments := p.getComments()
	if len(comments) > 0 {
		p.exprs = append(p.exprs, &comments)
	}
}

func (p *Parser) parseRoot() *RootExpr {
	rootOp := &RootExpr{}

	// Ensure the scanner has been created over the first file.
	if p.s == nil {
		// If no files to parse, then return empty root expression.
		if len(p.files) == 0 {
			return rootOp
		}

		if !p.openScanner() {
			return nil
		}
	}

	for {
		var tags TagsExpr
		var comments CommentsExpr

		tok := p.scan()
		src := p.src

		switch tok {
		case EOF:
			return rootOp

		case LBRACKET:
			p.unscan()

			comments = p.getComments()
			tags = p.parseTags()
			if tags == nil {
				p.tryRecover()
				break
			}

			if p.scan() != IDENT {
				p.unscan()

				rule := p.parseRule(comments, tags, src)
				if rule == nil {
					p.tryRecover()
					break
				}
				p.setComments(rule, comments)

				rootOp.Rules = append(rootOp.Rules, rule)
				p.exprs = append(p.exprs, rule)
				break
			}

			fallthrough

		case IDENT:
			// Only define identifier is allowed at the top level.
			if !p.isDefineIdent() {
				p.addExpectedTokenErr("define statement")
				p.tryRecover()
				break
			}
			// If there was no tag, we need to check for comments.
			if len(comments) == 0 {
				comments = p.getComments()
			}

			p.unscan()

			define := p.parseDefine(comments, tags, src)
			if define == nil {
				p.tryRecover()
				break
			}
			p.setComments(define, comments)

			rootOp.Defines = append(rootOp.Defines, define)
			p.exprs = append(p.exprs, define)

		default:
			p.addExpectedTokenErr("define statement or rule")
			p.tryRecover()
		}
	}
}

func (p *Parser) parseDefine(comments CommentsExpr, tags TagsExpr, src SourceLoc) *DefineExpr {
	if !p.scanToken(IDENT, "define statement") || p.s.Literal() != "define" {
		return nil
	}

	if !p.scanToken(IDENT, "define name") {
		return nil
	}

	name := p.s.Literal()
	define := &DefineExpr{Src: &src, Comments: comments, Name: StringExpr(name), Tags: tags}

	if !p.scanToken(LBRACE, "'{'") {
		return nil
	}

	for {
		if p.scan() == RBRACE {
			if p.hasComments() {
				p.addErr(fmt.Sprintf("comments not allowed before closing }: %v", p.comments))
				return nil
			}
			return define
		}
		p.unscan()

		defineField := p.parseDefineField()
		if defineField == nil {
			return nil
		}

		define.Fields = append(define.Fields, defineField)
	}
}


func (p *Parser) parseDefineField() *DefineFieldExpr {
	if !p.scanToken(IDENT, "define field name") {
		return nil
	}

	src := p.src
	name := p.s.Literal()

	// Scan tokens until the first white space.
	var typ bytes.Buffer
	tok := p.scan()
	for {
		if tok != IDENT && tok != ASTERISK && tok != LBRACKET && tok != RBRACKET && tok != DOT { 			p.addExpectedTokenErr("define field type")
			return nil 		}
		typ.WriteString(p.s.Literal())
		tok = p.scanInternal(false /* skipWhitespace */)
		if tok == EOF || tok == WHITESPACE {
			break
		}
	}

	field := &DefineFieldExpr{
		Src:      &src,
		Name:     StringExpr(name),
		Comments: p.getComments(),
		Type:     StringExpr(typ.String()),
	}
	p.setComments(field, field.Comments)
	return field
}