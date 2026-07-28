package main

func catalog(ctx *context.Context, cfg config.SBOM, artifacts []*artifact.Artifact) error {
	for _, a := range artifacts {
		newArtifacts, err := catalogArtifact(ctx, cfg, a)
		if err != nil {
			return err
		}
		for _, newArtifact := range newArtifacts {
			ctx.Artifacts.Add(newArtifact)
		}
	}
	return nil
}

func subprocessDistPath(distDir string, pathRelativeToCwd string) (string, error) {
	distDir = filepath.Clean(distDir)
	pathRelativeToCwd = filepath.Clean(pathRelativeToCwd)
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	if !filepath.IsAbs(distDir) {
		distDir, err = filepath.Abs(distDir)
		if err != nil {
			return "", err
		}
	}
	relativePath, err := filepath.Rel(cwd, distDir)
	if err != nil {
		return "", err
	}
	return strings.TrimPrefix(pathRelativeToCwd, relativePath+string(filepath.Separator)), nil
}

func catalogArtifact(ctx *context.Context, cfg config.SBOM, a *artifact.Artifact) ([]*artifact.Artifact, error) {
	artifactDisplayName := "(any)"
	args, envs, paths, err := applyTemplate(ctx, cfg, a)
	if err != nil {
		return nil, fmt.Errorf("cataloging artifacts failed: %w", err)
	}

	if a != nil {
		artifactDisplayName = a.Path
	}

	var names []string
	for _, p := range paths {
		names = append(names, filepath.Base(p))
	}

	// The GoASTScanner flags this as a security risk.
	// However, this works as intended. The nosec annotation
	// tells the scanner to ignore this.
	// #nosec
	cmd := exec.CommandContext(ctx, cfg.Cmd, args...)
	cmd.Env = []string{}
	for _, key := range passthroughEnvVars {
		if value := os.Getenv(key); value != "" {
			cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", key, value))
		}
	}
	cmd.Env = append(cmd.Env, envs...)
	cmd.Dir = ctx.Config.Dist

	log.WithField("dir", cmd.Dir).
		WithField("cmd", cmd.Args).
		Debug("running")

	var b bytes.Buffer
	w := gio.Safe(&b)
	cmd.Stderr = io.MultiWriter(logext.NewWriter(), w)
	cmd.Stdout = io.MultiWriter(logext.NewWriter(), w)

	log.WithField("cmd", cfg.Cmd).
		WithField("artifact", artifactDisplayName).
		WithField("sbom", names).
		Info("cataloging")
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("cataloging artifacts: %s failed: %w: %s", cfg.Cmd, err, b.String())
	}

	var artifacts []*artifact.Artifact

	for _, path := range paths {
		if !filepath.IsAbs(path) {
			path = filepath.Join(ctx.Config.Dist, path)
		}

		matches, err := filepath.Glob(path)
		if err != nil {
			return nil, fmt.Errorf("cataloging artifacts: failed to find SBOM artifact %q: %w", path, err)
		}
		for _, match := range matches {
			artifacts = append(artifacts, &artifact.Artifact{
				Type: artifact.SBOM,
				Name: filepath.Base(path),
				Path: match,
				Extra: map[string]any{
					artifact.ExtraID: cfg.ID,
				},
			})
		}
	}

	if len(artifacts) == 0 {
		return nil, errors.New("cataloging artifacts: command did not write any files, check your configuration")
	}

	return artifacts, nil
}

func applyTemplate(ctx *context.Context, cfg config.SBOM, a *artifact.Artifact) ([]string, []string, []string, error) {
	env := ctx.Env.Copy()
	var extraEnvs []string
	templater := tmpl.New(ctx).WithEnv(env)

	if a != nil {
		procPath, err := subprocessDistPath(ctx.Config.Dist, a.Path)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("cataloging artifacts failed: cannot determine artifact path for %q: %w", a.Path, err)
		}
		extraEnvs = appendExtraEnv("artifact", procPath, extraEnvs, env)
		extraEnvs = appendExtraEnv("artifactID", a.ID(), extraEnvs, env)
		templater = templater.WithArtifact(a)
	}

	for _, keyValue := range cfg.Env {
		renderedKeyValue, err := templater.Apply(expand(keyValue, env))
		if err != nil {
			return nil, nil, nil, fmt.Errorf("env %q: invalid template: %w", keyValue, err)
		}
		extraEnvs = append(extraEnvs, renderedKeyValue)

		k, v, _ := strings.Cut(renderedKeyValue, "=")
		env[k] = v
	}

	var paths []string
	for idx, sbom := range cfg.Documents {
		input := expand(sbom, env)
		if !filepath.IsAbs(input) {
			// assume any absolute path is handled correctly and assume that any relative path is not already
			// adjusted to reference the dist path
			input = filepath.Join(ctx.Config.Dist, input)
		}

		path, err := templater.Apply(input)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("input %q: invalid template: %w", input, err)
		}

		path, err = filepath.Abs(path)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("unable to create artifact path %q: %w", sbom, err)
		}

		procPath, err := subprocessDistPath(ctx.Config.Dist, path)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("cannot determine document path for %q: %w", path, err)
		}

		extraEnvs = appendExtraEnv(fmt.Sprintf("document%d", idx), procPath, extraEnvs, env)
		if idx == 0 {
			extraEnvs = appendExtraEnv("document", procPath, extraEnvs, env)
		}

		paths = append(paths, procPath)
	}

	//nolint:prealloc
	var args []string
	for _, arg := range cfg.Args {
		renderedArg, err := templater.Apply(expand(arg, env))
		if err != nil {
			return nil, nil, nil, fmt.Errorf("arg %q: invalid template: %w", arg, err)
		}
		args = append(args, renderedArg)
	}

	return args, extraEnvs, paths, nil
}

func appendExtraEnv(key, value string, envs []string, env map[string]string) []string {
	env[key] = value
	return append(envs, fmt.Sprintf("%s=%s", key, value))
}


func expand(s string, env map[string]string) string { 	return os.Expand(s, func(key string) string {
		return env[key]
	}) }