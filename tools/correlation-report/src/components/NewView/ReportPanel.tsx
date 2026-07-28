import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styles from './NewView.module.css';

const TTAV_PREFS_KEY = 'eif:ttav-launch-prefs';
const TTAV_PREPARED_BUNDLES_KEY = 'eif:ttav-prepared-bundles';
const DEFAULT_TTAV_URL = 'http://1.94.115.154/';
const DEFAULT_TTAV_CONTENT_PATH_TEMPLATE = '/root/project/Dataset/eif_bundles/{sampleId}';
const DEFAULT_EIF_BUNDLE_CACHE_TEMPLATE = '/home/yilu/workspace/Empirical-Influence-Function/ttav_bundles/{sampleId}';
const DEFAULT_TTAV_METHOD = 'TimeVis';
const DEFAULT_TTAV_VIS_ID = '1';
function getDefaultEifApiUrl(): string {
    if (typeof window === 'undefined') {
        return 'http://127.0.0.1:8766/api/prepare-ttav-bundle';
    }

    const protocol = window.location.protocol || 'http:';
    const hostname = window.location.hostname || '127.0.0.1';
    const port = hostname === 'localhost' || hostname === '127.0.0.1' ? '8766' : '8765';
    return `${protocol}//${hostname}:${port}/api/prepare-ttav-bundle`;
}

const DEFAULT_EIF_API_URL = getDefaultEifApiUrl();

// ─── Types ────────────────────────────────────────────────────────────────────

interface TestCorrelation {
    source_token: string;
    source_token_index: number;
    target_token: string;
    target_token_index: number;
    saliency_score: number;
}

interface TrainCorrelation {
    source_token: string;
    source_token_index: number;
    target_token: string;
    target_token_index: number;
    saliency_score: number;
    response_token_offset: number;
}

interface CorrelationPair {
    id: string;
    cos_sim: number;
    coarse_cos_sim: number;
    train_sample_id: number;
    test_correlation: TestCorrelation;
    train_correlation: TrainCorrelation;
    train_context: { source_context: string[]; target_context: string[] };
    annotation: string | null;
}

interface TrainProbeFocusToken {
    tokenIndex: number;
    token: string;
    tokenDisplay: string;
    pointIndex?: number;
}

interface TrainProbeComparisonPair {
    leftIndex: number;
    leftToken: string;
    leftTokenDisplay: string;
    rightIndex: number;
    rightToken: string;
    rightTokenDisplay: string;
    cosine: number;
}

interface TrainProbeComparisonSummary {
    focusTokens: TrainProbeFocusToken[];
    pairwiseCosine: TrainProbeComparisonPair[];
}

interface PerTokenResult {
    target_token_index: number;
    target_token: string;
    top_correlations: TestCorrelation[];
    correlation_pairs: CorrelationPair[];
}

interface TrainSampleDetail {
    full_tokens: string[];
    answer_start_index: number;
    coarse_cos_sim: number;
    saliencies_by_token: Record<string, number[]>;
}

export interface AllTokensReport {
    experiment_meta: {
        test_sample_index: number;
        mode: 'all_tokens';
        tokens_analyzed: number;
        model_name?: string;
        model_path?: string | null;
        task_id?: string;
    };
    test_sample_baseline: {
        full_tokens: string[];
        correct_full_tokens: string[];
        prompt_len: number;
    };
    per_token_results: PerTokenResult[];
    train_sample_details: Record<string, TrainSampleDetail>;
}

export interface ImportedReport {
    report: AllTokensReport;
    meta: AllTokensExperimentMeta;
    format: 'all_tokens' | 'generic_saliency';
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function asFiniteNumber(value: unknown): number | null {
    if (typeof value !== 'number' || !Number.isFinite(value)) return null;
    return value;
}

function asString(value: unknown): string | null {
    return typeof value === 'string' ? value : null;
}

function asStringArray(value: unknown): string[] | null {
    if (!Array.isArray(value) || !value.every(item => typeof item === 'string')) return null;
    return value;
}

function asNumberArray(value: unknown): number[] | null {
    if (!Array.isArray(value)) return null;
    const values = value.map(asFiniteNumber);
    if (values.some(item => item === null)) return null;
    return values as number[];
}

function firstString(record: Record<string, unknown>, keys: string[]): string | null {
    for (const key of keys) {
        const value = asString(record[key]);
        if (value) return value;
    }
    return null;
}

function firstStringArray(record: Record<string, unknown>, keys: string[]): string[] | null {
    for (const key of keys) {
        const value = asStringArray(record[key]);
        if (value) return value;
    }
    return null;
}

function firstNumber(record: Record<string, unknown>, keys: string[]): number | null {
    for (const key of keys) {
        const value = asFiniteNumber(record[key]);
        if (value !== null) return value;
    }
    return null;
}

function stemFromSource(sourceName: string): string {
    const lastPart = sourceName.split(/[\\/]/).pop() || sourceName;
    return lastPart.replace(/\.json$/i, '') || 'uploaded-report';
}

function buildImportedMeta(report: AllTokensReport, sourceName: string): AllTokensExperimentMeta {
    const stem = stemFromSource(sourceName);
    const testIndex = report.experiment_meta.test_sample_index;
    const taskId = testIndex >= 0 ? `uploaded_test${testIndex}_${stem}` : `uploaded_${stem}`;
    return {
        taskId,
        label: `${stem} (uploaded)`,
        fileName: sourceName,
    };
}

export function isAllTokensReportLike(value: unknown): value is AllTokensReport {
    if (!isRecord(value)) return false;
    const experimentMeta = value.experiment_meta;
    const baseline = value.test_sample_baseline;
    if (!isRecord(experimentMeta) || !isRecord(baseline)) return false;
    return Array.isArray(value.per_token_results)
        && asStringArray(baseline.full_tokens) !== null
        && asFiniteNumber(baseline.prompt_len) !== null;
}

function normalizeTestCorrelation(value: unknown, tokens: string[], fallbackTargetIdx: number): TestCorrelation | null {
    if (!isRecord(value)) return null;
    const sourceIdx = asFiniteNumber(value.source_token_index);
    const targetIdx = asFiniteNumber(value.target_token_index) ?? fallbackTargetIdx;
    const score = asFiniteNumber(value.saliency_score);
    if (sourceIdx === null || targetIdx === null || score === null) return null;
    const sourceIndex = Math.trunc(sourceIdx);
    const targetIndex = Math.trunc(targetIdx);
    return {
        source_token: asString(value.source_token) ?? tokens[sourceIndex] ?? '',
        source_token_index: sourceIndex,
        target_token: asString(value.target_token) ?? tokens[targetIndex] ?? '',
        target_token_index: targetIndex,
        saliency_score: score,
    };
}

export function normalizeAllTokensReport(report: AllTokensReport): AllTokensReport {
    const baseline = report.test_sample_baseline;
    const fullTokens = asStringArray(baseline.full_tokens) ?? [];
    const promptLen = Math.max(0, Math.min(fullTokens.length, Math.trunc(asFiniteNumber(baseline.prompt_len) ?? 0)));
    const correctTokens = asStringArray(baseline.correct_full_tokens) ?? fullTokens;
    const perTokenResults = (Array.isArray(report.per_token_results) ? report.per_token_results : [])
        .map((item): PerTokenResult | null => {
            if (!isRecord(item)) return null;
            const targetIdxRaw = asFiniteNumber(item.target_token_index);
            if (targetIdxRaw === null) return null;
            const targetIdx = Math.trunc(targetIdxRaw);
            return {
                target_token_index: targetIdx,
                target_token: asString(item.target_token) ?? fullTokens[targetIdx] ?? '',
                top_correlations: (Array.isArray(item.top_correlations) ? item.top_correlations : [])
                    .map(c => normalizeTestCorrelation(c, fullTokens, targetIdx))
                    .filter((c): c is TestCorrelation => c !== null),
                correlation_pairs: Array.isArray(item.correlation_pairs)
                    ? (item.correlation_pairs as CorrelationPair[])
                    : [],
            };
        })
        .filter((item): item is PerTokenResult => item !== null);

    return {
        experiment_meta: {
            test_sample_index: Math.trunc(asFiniteNumber(report.experiment_meta.test_sample_index) ?? -1),
            mode: 'all_tokens',
            tokens_analyzed: Math.trunc(asFiniteNumber(report.experiment_meta.tokens_analyzed) ?? perTokenResults.length),
            model_name: asString(report.experiment_meta.model_name) ?? undefined,
            model_path: asString(report.experiment_meta.model_path) ?? undefined,
            task_id: asString(report.experiment_meta.task_id) ?? undefined,
        },
        test_sample_baseline: {
            full_tokens: fullTokens,
            correct_full_tokens: correctTokens,
            prompt_len: promptLen,
        },
        per_token_results: perTokenResults,
        train_sample_details: isRecord(report.train_sample_details)
            ? report.train_sample_details as Record<string, TrainSampleDetail>
            : {},
    };
}

interface GenericSaliencyItem {
    targetIdx: number;
    targetToken?: string;
    scores: number[];
}

function readGenericSaliencyItems(record: Record<string, unknown>, tokenCount: number): GenericSaliencyItem[] {
    const items: GenericSaliencyItem[] = [];

    const pushItem = (raw: unknown) => {
        if (!isRecord(raw)) return;
        const targetIdx = firstNumber(raw, ['target_token_index', 'target_index', 'index']);
        const scores = firstNumberArray(raw, ['scores', 'saliency', 'saliency_scores', 'source_scores']);
        if (targetIdx === null || !scores) return;
        const idx = Math.trunc(targetIdx);
        if (idx < 0 || idx >= tokenCount) return;
        items.push({
            targetIdx: idx,
            targetToken: firstString(raw, ['target_token', 'token']) ?? undefined,
            scores,
        });
    };

    for (const key of ['saliency', 'saliency_list', 'saliencies', 'targets']) {
        const value = record[key];
        if (Array.isArray(value)) value.forEach(pushItem);
    }

    for (const key of ['saliency_by_target', 'saliencyByTarget']) {
        const value = record[key];
        if (!isRecord(value)) continue;
        for (const [targetIdxRaw, scoresRaw] of Object.entries(value)) {
            const targetIdx = Number(targetIdxRaw);
            const scores = asNumberArray(scoresRaw);
            if (!Number.isFinite(targetIdx) || !scores) continue;
            const idx = Math.trunc(targetIdx);
            if (idx < 0 || idx >= tokenCount) continue;
            items.push({ targetIdx: idx, scores });
        }
    }

    return items;
}

function firstNumberArray(record: Record<string, unknown>, keys: string[]): number[] | null {
    for (const key of keys) {
        const value = asNumberArray(record[key]);
        if (value) return value;
    }
    return null;
}

function topCorrelationsFromScores(tokens: string[], targetIdx: number, targetToken: string, scores: number[]): TestCorrelation[] {
    return scores
        .map((score, idx) => ({ idx, score }))
        .filter(({ idx, score }) => idx !== targetIdx && idx < tokens.length && Number.isFinite(score) && score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 12)
        .map(({ idx, score }) => ({
            source_token: tokens[idx] ?? '',
            source_token_index: idx,
            target_token: targetToken,
            target_token_index: targetIdx,
            saliency_score: score,
        }));
}

function genericSaliencyToAllTokensReport(payload: unknown): AllTokensReport | null {
    if (!isRecord(payload)) return null;

    const targetTestSample = payload.target_test_sample;
    const legacyBefore = isRecord(targetTestSample) && isRecord(targetTestSample.before)
        ? targetTestSample.before
        : null;
    const source = legacyBefore ?? payload;

    const tokens = firstStringArray(source, ['full_tokens', 'tokens', 'token_list'])
        ?? firstStringArray(payload, ['full_tokens', 'tokens', 'token_list']);
    if (!tokens || tokens.length === 0) return null;

    const promptLenRaw = firstNumber(source, ['prompt_len', 'start_index', 'answer_start_index'])
        ?? firstNumber(payload, ['prompt_len', 'start_index', 'answer_start_index'])
        ?? tokens.length;
    const promptLen = Math.max(0, Math.min(tokens.length, Math.trunc(promptLenRaw)));
    const correctTokens = firstStringArray(source, ['correct_full_tokens', 'correct_tokens'])
        ?? firstStringArray(payload, ['correct_full_tokens', 'correct_tokens'])
        ?? tokens;
    const saliencyItems = readGenericSaliencyItems(source, tokens.length);
    if (saliencyItems.length === 0) return null;

    const experimentMeta = payload.experiment_meta;
    const sampleIndex = firstNumber(payload, ['test_sample_index', 'sample_index'])
        ?? (isRecord(experimentMeta) ? firstNumber(experimentMeta, ['test_sample_index']) : null)
        ?? -1;

    return {
        experiment_meta: {
            test_sample_index: Math.trunc(sampleIndex),
            mode: 'all_tokens',
            tokens_analyzed: saliencyItems.length,
        },
        test_sample_baseline: {
            full_tokens: tokens,
            correct_full_tokens: correctTokens,
            prompt_len: promptLen,
        },
        per_token_results: saliencyItems.map(item => {
            const targetToken = item.targetToken ?? tokens[item.targetIdx] ?? '';
            return {
                target_token_index: item.targetIdx,
                target_token: targetToken,
                top_correlations: topCorrelationsFromScores(tokens, item.targetIdx, targetToken, item.scores),
                correlation_pairs: [],
            };
        }),
        train_sample_details: {},
    };
}

export function normalizeImportedReport(payload: unknown, sourceName: string): ImportedReport {
    if (isAllTokensReportLike(payload)) {
        const report = normalizeAllTokensReport(payload);
        return {
            report,
            meta: buildImportedMeta(report, sourceName),
            format: 'all_tokens',
        };
    }

    const genericReport = genericSaliencyToAllTokensReport(payload);
    if (genericReport) {
        return {
            report: genericReport,
            meta: buildImportedMeta(genericReport, sourceName),
            format: 'generic_saliency',
        };
    }

    throw new Error('Unsupported JSON format. Expected an all-token report, or tokens + prompt_len + saliency_list/saliency_by_target.');
}

interface TtavLaunchPrefs {
    ttavUrl: string;
    contentPathTemplate: string;
    eifBundleCacheTemplate: string;
    visMethod: string;
    visId: string;
    eifApiUrl: string;
}

interface TtavJumpPayload {
    source: 'eif';
    sampleId: string;
    contentPath: string;
    visMethod: string;
    visId: string;
    dataType: 'Text';
    taskType: 'Alignment';
    selectedIndices: number[];
    targetIndex?: number;
    selectedSourceIndex?: number;
    promptLen: number;
}

interface TtavStaticBundlePayload {
    sample_id: string;
    vis_method: string;
    vis_id: string;
    overwrite: boolean;
    bundle: {
        model: string;
        classes: string[];
        sample_index: number;
        prompt_len: number;
        labels: number[];
        text_list: string[];
        text_data: string[];
        token_list: string[];
        index: { train: number[]; test: number[] };
        embeddings: number[][];
        projection: number[][];
    };
}

interface PreparedTtavBundleRecord {
    sampleId: string;
    contentPath: string;
    visMethod: string;
    visId: string;
    preparedAt: number;
}

interface TtavHighlightUpdateMessage {
    command: 'eifHighlightUpdate';
    data: TtavJumpPayload;
}

interface EifPrepareStatusPayload {
    status: 'success';
    sampleId: string;
    stage: string;
    message: string;
    active: boolean;
    error: boolean;
    updatedAt: number;
}

function loadPreparedTtavBundles(): Record<string, PreparedTtavBundleRecord> {
    if (typeof window === 'undefined') return {};

    try {
        const raw = window.localStorage.getItem(TTAV_PREPARED_BUNDLES_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw) as Record<string, PreparedTtavBundleRecord>;
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch {
        return {};
    }
}

function savePreparedTtavBundle(record: PreparedTtavBundleRecord) {
    if (typeof window === 'undefined') return;

    const current = loadPreparedTtavBundles();
    current[record.sampleId] = record;
    window.localStorage.setItem(TTAV_PREPARED_BUNDLES_KEY, JSON.stringify(current));
}

function getPreparedTtavBundle(sampleId: string): PreparedTtavBundleRecord | null {
    const current = loadPreparedTtavBundles();
    return current[sampleId] ?? null;
}

async function loadPrecomputedRealBundle(sampleId: string, visMethod: string, visId: string): Promise<TtavStaticBundlePayload> {
    const bundleResp = await fetch(`/data/real-bundles/${encodeURIComponent(sampleId)}/bundle_payload.json`, {
        cache: 'no-store',
    });

    if (!bundleResp.ok) {
        throw new Error(`Precomputed real bundle not found for ${sampleId} (HTTP ${bundleResp.status}).`);
    }

    const payload = await bundleResp.json() as TtavStaticBundlePayload;
    return {
        ...payload,
        sample_id: sampleId,
        vis_method: visMethod,
        vis_id: visId,
        overwrite: true,
    };
}

function buildEifApiUrl(apiUrl: string, pathname: string): string {
    const url = new URL(apiUrl.trim());
    url.pathname = pathname;
    url.search = '';
    return url.toString();
}

function buildEifPrepareStatusUrl(apiUrl: string, sampleId: string): string {
    const url = new URL(buildEifApiUrl(apiUrl, '/api/prepare-ttav-bundle-status'));
    url.searchParams.set('sampleId', sampleId);
    return url.toString();
}

async function fetchEifPrepareStatus(
    apiUrl: string,
    sampleId: string,
    timeoutMs = 3000,
): Promise<EifPrepareStatusPayload | null> {
    if (!apiUrl.trim() || !sampleId) return null;

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

    try {
        const resp = await fetch(buildEifPrepareStatusUrl(apiUrl, sampleId), {
            cache: 'no-store',
            signal: controller.signal,
        });
        if (!resp.ok) return null;

        const payload = await resp.json() as Partial<EifPrepareStatusPayload> & { status?: string };
    if (payload.status !== 'success' || typeof payload.message !== 'string' || typeof payload.sampleId !== 'string') {
        return null;
    }

        return {
            status: 'success',
            sampleId: payload.sampleId,
            stage: typeof payload.stage === 'string' ? payload.stage : 'unknown',
            message: payload.message,
            active: payload.active === true,
            error: payload.error === true,
            updatedAt: typeof payload.updatedAt === 'number' ? payload.updatedAt : Date.now(),
        };
    } catch {
        return null;
    } finally {
        window.clearTimeout(timeoutId);
    }
}

function loadTtavLaunchPrefs(): TtavLaunchPrefs {
    const fallbackPrefs: TtavLaunchPrefs = {
        ttavUrl: DEFAULT_TTAV_URL,
        contentPathTemplate: DEFAULT_TTAV_CONTENT_PATH_TEMPLATE,
        eifBundleCacheTemplate: DEFAULT_EIF_BUNDLE_CACHE_TEMPLATE,
        visMethod: DEFAULT_TTAV_METHOD,
        visId: DEFAULT_TTAV_VIS_ID,
        eifApiUrl: DEFAULT_EIF_API_URL,
    };

    if (typeof window === 'undefined') {
        return fallbackPrefs;
    }

    try {
        const raw = window.localStorage.getItem(TTAV_PREFS_KEY);
        if (!raw) throw new Error('missing prefs');
        const parsed = JSON.parse(raw) as Partial<TtavLaunchPrefs>;

        const legacyUrl = parsed.ttavUrl?.includes('localhost:5174');
        const legacyMethod = parsed.visMethod?.trim().toUpperCase() === 'UMAP';
        const legacyPath = parsed.contentPathTemplate?.includes('/root/project/time-travelling-visualizer/data/eif_bundles/');
        const legacyEifCachePath = parsed.eifBundleCacheTemplate?.includes('/root/project/time-travelling-visualizer/data/eif_bundles/')
            || parsed.eifBundleCacheTemplate?.includes('/root/project/Empirical-Influence-Function/ttav_bundles/');
        const defaultApiHost = new URL(DEFAULT_EIF_API_URL).host;
        const parsedApiHost = parsed.eifApiUrl ? (() => {
            try {
                return new URL(parsed.eifApiUrl).host;
            } catch {
                return '';
            }
        })() : '';
        const legacyApiUrl = parsed.eifApiUrl?.includes('124.70.161.19:8765')
            || parsed.eifApiUrl?.includes('0.0.0.0:8765')
            || parsed.eifApiUrl?.includes('127.0.0.1:8765')
            || parsed.eifApiUrl?.includes('localhost:8765')
            || (parsedApiHost !== '' && parsedApiHost !== defaultApiHost);

        return {
            ttavUrl: legacyUrl ? DEFAULT_TTAV_URL : (parsed.ttavUrl || DEFAULT_TTAV_URL),
            contentPathTemplate: legacyPath
                ? DEFAULT_TTAV_CONTENT_PATH_TEMPLATE
                : (parsed.contentPathTemplate || DEFAULT_TTAV_CONTENT_PATH_TEMPLATE),
            eifBundleCacheTemplate: legacyEifCachePath
                ? DEFAULT_EIF_BUNDLE_CACHE_TEMPLATE
                : (parsed.eifBundleCacheTemplate || DEFAULT_EIF_BUNDLE_CACHE_TEMPLATE),
            visMethod: legacyMethod ? DEFAULT_TTAV_METHOD : (parsed.visMethod || DEFAULT_TTAV_METHOD),
            visId: parsed.visId || DEFAULT_TTAV_VIS_ID,
            eifApiUrl: legacyApiUrl ? DEFAULT_EIF_API_URL : (parsed.eifApiUrl || DEFAULT_EIF_API_URL),
        };
    } catch {
        return fallbackPrefs;
    }
}

function resolveContentPath(template: string, sampleId: string): string {
    return template
        .replaceAll('{sampleId}', sampleId)
        .replaceAll('{taskId}', sampleId);
}

function inferSampleIdFromMeta(meta: AllTokensExperimentMeta, report?: AllTokensReport | null): string {
    if (report?.experiment_meta.task_id) {
        return report.experiment_meta.task_id;
    }
    const fileStem = meta.fileName.replace(/\.json$/i, '');
    const modelName = report?.experiment_meta.model_name;
    if (modelName) {
        const prefix = `correlation_matching_results_${modelName}_`;
        const suffix = '_all_tokens';
        if (fileStem.startsWith(prefix) && fileStem.endsWith(suffix)) {
            return fileStem.slice(prefix.length, -suffix.length);
        }
    }
    // Legacy: correlation_matching_results_{task}_all_tokens
    const matchLegacy = fileStem.match(/^correlation_matching_results_(.+)_all_tokens$/);
    if (matchLegacy) {
        return matchLegacy[1];
    }
    return meta.taskId || meta.label || (report ? `test${report.experiment_meta.test_sample_index}` : fileStem);
}

// ─── Token helpers ────────────────────────────────────────────────────────────

function decodeToken(t: string): string {
    return t.replaceAll('Ċ', '\n').replaceAll('Ġ', ' ').replaceAll('ĉ', '  ');
}

function decodeTokens(tokens: string[]): string[] {
    return tokens.map(decodeToken);
}

const TRIVIAL_STRIPPED = new Set(['{', '}', '(', ')', '[', ']', ',', ';']);

function isTrivialToken(t: string): boolean {
    const stripped = decodeToken(t).trim();
    if (!stripped) return true;
    if (TRIVIAL_STRIPPED.has(stripped)) return true;
    if (stripped.length === 1 && !/[a-zA-Z0-9_]/.test(stripped)) return true;
    return false;
}

function cosSimilarityColor(s: number): { bg: string; fg: string } {
    if (s > 0.6) return { bg: '#dcfce7', fg: '#15803d' };
    if (s > 0.3) return { bg: '#fef9c3', fg: '#854d0e' };
    return { bg: '#fee2e2', fg: '#b91c1c' };
}

function shouldFallbackToDirectPrepare(message: string): boolean {
    const lower = message.toLowerCase();
    return lower.includes('failed to fetch')
        || lower.includes('networkerror')
        || lower.includes('load failed')
        || lower.includes('not found')
        || lower.includes('http 404')
        || lower.includes('http 500')
        || lower.includes('http 502')
        || lower.includes('http 503')
        || lower.includes('http 504')
        || lower.includes('eif local bundle cache not found')
        || lower.includes('non-json response');
}


// ─── Export helpers ───────────────────────────────────────────────────────────

function formatContextForExport(tokens: string[]): string {
    return tokens.map(tok => {
        const marked = tok.startsWith('→[') && tok.endsWith(']←');
        const raw = decodeToken(marked ? tok.slice(2, -2) : tok);
        return marked ? `[${raw.trim() || '·'}]` : raw;
    }).join('');
}

interface ExportOptions {
    tokenRange: 'all' | 'selected' | 'range';
    selectedTokenIdx?: number;
    rangeFrom?: number;
    rangeTo?: number;
    cosSimThreshold: number;
    hideZero: boolean;
}

function generateExportMarkdown(report: AllTokensReport, options: ExportOptions): string {
    const { test_sample_baseline: baseline, per_token_results, train_sample_details, experiment_meta } = report;
    const promptLen = baseline.prompt_len;

    let tokensToExport: PerTokenResult[];
    if (options.tokenRange === 'selected' && options.selectedTokenIdx != null) {
        const r = per_token_results.find(r => r.target_token_index === options.selectedTokenIdx);
        tokensToExport = r ? [r] : [];
    } else if (options.tokenRange === 'range' && options.rangeFrom != null && options.rangeTo != null) {
        tokensToExport = per_token_results.filter(r =>
            r.target_token_index >= options.rangeFrom! && r.target_token_index <= options.rangeTo!
        );
    } else {
        tokensToExport = [...per_token_results];
    }

    const keepPair = (p: CorrelationPair) =>
        p.cos_sim >= options.cosSimThreshold && !(options.hideZero && p.cos_sim === 0);

    const L: string[] = [];

    // Header
    L.push(`# Attribution Report: Test Sample #${experiment_meta.test_sample_index}\n`);
    L.push(`- Tokens exported: ${tokensToExport.length} / ${per_token_results.length} analyzed`);
    L.push(`- Prompt length: ${promptLen}`);
    L.push(`- cos_sim filter: ≥${options.cosSimThreshold.toFixed(3)}${options.hideZero ? ', hiding cos=0' : ''}`);
    L.push('');

    // Test Code
    const promptText = decodeTokens(baseline.full_tokens.slice(0, promptLen)).join('');
    const modelResp = decodeTokens(baseline.full_tokens.slice(promptLen)).join('');
    const gtTokens = baseline.correct_full_tokens ?? [];
    const correctResp = decodeTokens(gtTokens.slice(promptLen)).join('');

    L.push('## Test Code\n');
    L.push('### Prompt\n' + '```');
    L.push(promptText.trimEnd());
    L.push('```' + '\n');
    L.push('### Model Output (response)\n' + '```');
    L.push(modelResp.trimEnd());
    L.push('```' + '\n');
    if (correctResp) {
        L.push('### Ground Truth (response)\n' + '```');
        L.push(correctResp.trimEnd());
        L.push('```' + '\n');
    }

    // Per-token analysis
    L.push('## Token Analysis\n');

    for (const tr of tokensToExport) {
        const tgt = decodeToken(tr.target_token).trim() || '·';
        const absIdx = tr.target_token_index;
        const correctTok = gtTokens[absIdx];
        const correctText = correctTok ? (decodeToken(correctTok).trim() || '·') : '?';
        const isCorrect = tr.target_token === correctTok;

        L.push(`### \`${tgt}\` @ pos ${absIdx} | GT: \`${correctText}\` | ${isCorrect ? '✓' : '✗'}\n`);

        // Feature Attribution
        L.push('Feature Attribution:');
        for (const c of tr.top_correlations) {
            const src = decodeToken(c.source_token).trim() || '·';
            L.push(`- \`${src}\`@${c.source_token_index} → \`${tgt}\` (saliency: ${c.saliency_score.toFixed(4)})`);
        }
        L.push('');

        // Data Attribution
        const pairs = tr.correlation_pairs.filter(keepPair);
        if (pairs.length === 0) {
            L.push('Data Attribution: No matches above threshold.\n');
            L.push('---\n');
            continue;
        }

        const byTrain = new Map<number, CorrelationPair[]>();
        for (const p of pairs) {
            if (!byTrain.has(p.train_sample_id)) byTrain.set(p.train_sample_id, []);
            byTrain.get(p.train_sample_id)!.push(p);
        }

        L.push('Data Attribution:');
        for (const [tid, tpairs] of byTrain) {
            const detail = train_sample_details[String(tid)];
            const coarse = detail?.coarse_cos_sim ?? tpairs[0]?.coarse_cos_sim ?? 0;
            L.push(`\n**Train #${tid}** (coarse: ${coarse.toFixed(4)}):`);
            for (const p of tpairs.sort((a, b) => b.cos_sim - a.cos_sim)) {
                const tSrc = decodeToken(p.test_correlation.source_token).trim() || '·';
                const tTgt = decodeToken(p.test_correlation.target_token).trim() || '·';
                const rSrc = decodeToken(p.train_correlation.source_token).trim() || '·';
                const rTgt = decodeToken(p.train_correlation.target_token).trim() || '·';
                const srcCtx = formatContextForExport(p.train_context.source_context);
                const tgtCtx = formatContextForExport(p.train_context.target_context);
                L.push(`- cos=${p.cos_sim.toFixed(4)} | test: \`${tSrc}\`→\`${tTgt}\` ⇔ train: \`${rSrc}\`→\`${rTgt}\``);
                L.push(`  src_ctx: ${srcCtx}`);
                L.push(`  tgt_ctx: ${tgtCtx}`);
            }
        }
        L.push('\n---\n');
    }

    // Training Samples 汇总（每个 train sample 只出现一次）
    const referencedTrainIds = new Set<number>();
    for (const tr of tokensToExport) {
        for (const p of tr.correlation_pairs.filter(keepPair)) {
            referencedTrainIds.add(p.train_sample_id);
        }
    }

    if (referencedTrainIds.size > 0) {
        L.push('## Training Samples\n');
        for (const tid of Array.from(referencedTrainIds).sort((a, b) => a - b)) {
            const detail = train_sample_details[String(tid)];
            if (!detail) continue;
            const tokens = decodeTokens(detail.full_tokens);
            const promptPart = tokens.slice(0, detail.answer_start_index).join('');
            const respPart = tokens.slice(detail.answer_start_index).join('');
            L.push(`### Train #${tid} (coarse: ${detail.coarse_cos_sim.toFixed(4)})\n`);
            L.push('Prompt:\n' + '```');
            L.push(promptPart.trimEnd());
            L.push('```' + '\n');
            L.push('Response:\n' + '```');
            L.push(respPart.trimEnd());
            L.push('```' + '\n');
        }
    }

    return L.join('\n');
}

function downloadMarkdown(text: string, filename: string) {
    const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ─── Token Renderer ───────────────────────────────────────────────────────────

type TokenState = 'normal' | 'response' | 'selected' | 'source-highlight' | 'analyzed';

function TokenSpan({
    token,
    state,
    onClick,
    title,
}: {
    token: string;
    state: TokenState;
    onClick?: () => void;
    title?: string;
}) {
    const display = token === '\n' ? '↵\n' : token === '  ' ? '→' : token;
    return (
        <span
            className={`${styles.token} ${styles[`token-${state}`]}`}
            onClick={onClick}
            title={title}
            style={{ cursor: onClick ? 'pointer' : 'default' }}
        >
            {display}
        </span>
    );
}

// ─── Code Panel (tokens display) ─────────────────────────────────────────────

function CodePanel({
    label,
    badge,
    badgeColor,
    tokens,
    promptLen,
    highlightSourceIndices,
    selectedTargetIndex,
    analyzedIndices,
    onTokenClick,
}: {
    label: string;
    badge: string;
    badgeColor: string;
    tokens: string[];
    promptLen: number;
    highlightSourceIndices?: Set<number>;
    selectedTargetIndex?: number;
    analyzedIndices?: Set<number>;
    onTokenClick?: (idx: number) => void;
}) {
    return (
        <div className={styles.codePanel}>
            <div className={styles.codePanelHeader}>
                <span className={styles.badge} style={{ background: badgeColor }}>{badge}</span>
                <span className={styles.codePanelLabel}>{label}</span>
            </div>
            <pre className={styles.codeBlock}>
                <code>
                    {tokens.map((tok, i) => {
                        const isResponse = i >= promptLen;
                        const isSelected = i === selectedTargetIndex;
                        const isSource   = highlightSourceIndices?.has(i) ?? false;
                        const isAnalyzed = analyzedIndices?.has(i) ?? false;

                        let state: TokenState = 'normal';
                        if (isSelected)  state = 'selected';
                        else if (isSource)   state = 'source-highlight';
                        else if (isAnalyzed && isResponse) state = 'analyzed';
                        else if (isResponse) state = 'response';

                        const clickable = isResponse && onTokenClick && !isTrivialToken(tok);
                        return (
                            <TokenSpan
                                key={i}
                                token={tok}
                                state={state}
                                onClick={clickable ? () => onTokenClick(i) : undefined}
                                title={clickable ? `Token ${i}: "${decodeToken(tok)}"` : undefined}
                            />
                        );
                    })}
                </code>
            </pre>
        </div>
    );
}

// ─── Context Chip ─────────────────────────────────────────────────────────────

function ContextChip({ tokens }: { tokens: string[] }) {
    return (
        <span className={styles.contextChip}>
            {tokens.map((tok, i) => {
                const marked = tok.startsWith('→[') && tok.endsWith(']←');
                const text   = (marked ? tok.slice(2, -2) : tok).replace(/\n/g, '↵').replace(/\t/g, '⇥') || '·';
                return (
                    <span key={i} className={marked ? styles.contextTokenMarked : styles.contextToken}>
                        {text}
                    </span>
                );
            })}
        </span>
    );
}

// ─── Train Sample Detail Viewer ───────────────────────────────────────────────

function TrainSampleViewer({
    detail,
    highlightPairs,
}: {
    detail: TrainSampleDetail;
    highlightPairs: CorrelationPair[];
}) {
    const tokens = useMemo(() => decodeTokens(detail.full_tokens), [detail]);

    // Collect all source + target indices to highlight from the pairs
    const sourceIndices = useMemo(() => new Set(highlightPairs.map(p => p.train_correlation.source_token_index)), [highlightPairs]);
    const targetIndices = useMemo(() => new Set(highlightPairs.map(p => p.train_correlation.target_token_index)), [highlightPairs]);

    return (
        <div className={styles.trainSampleViewer}>
            <pre className={styles.codeBlock} style={{ fontSize: '12px', maxHeight: '260px', overflow: 'auto' }}>
                <code>
                    {tokens.map((tok, i) => {
                        const isSrc = sourceIndices.has(i);
                        const isTgt = targetIndices.has(i);
                        let state: TokenState = i >= detail.answer_start_index ? 'response' : 'normal';
                        if (isTgt) state = 'selected';
                        else if (isSrc) state = 'source-highlight';
                        return <TokenSpan key={i} token={tok} state={state} />;
                    })}
                </code>
            </pre>
        </div>
    );
}

// ─── Correlation Pair Card ────────────────────────────────────────────────────

function PairCard({
    pair,
    detail,
    selected,
    onToggleSelect,
}: {
    pair: CorrelationPair;
    detail?: TrainSampleDetail;
    selected?: boolean;
    onToggleSelect?: () => void;
}) {
    const [expanded, setExpanded] = useState(false);
    const { bg, fg } = cosSimilarityColor(pair.cos_sim);

    return (
        <div
            className={styles.pairCard}
            style={selected ? { borderColor: '#7c3aed', boxShadow: '0 0 0 1px rgba(124,58,237,0.18)' } : undefined}
        >
            <div className={styles.pairCardHeader} onClick={() => setExpanded(e => !e)}>
                {onToggleSelect && (
                    <button
                        type="button"
                        onClick={(event) => {
                            event.stopPropagation();
                            onToggleSelect();
                        }}
                        aria-pressed={selected === true}
                        title={selected ? '取消选择这个 correlation pair' : '选择这个 correlation pair 参与 probe'}
                        style={{
                            border: selected ? '1px solid #7c3aed' : '1px solid #cbd5e1',
                            background: selected ? '#f5f3ff' : '#ffffff',
                            color: selected ? '#6d28d9' : '#64748b',
                            borderRadius: 999,
                            padding: '2px 8px',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: 'pointer',
                        }}
                    >
                        {selected ? '已选' : '选择'}
                    </button>
                )}

                <span className={styles.pairId}>{pair.id}</span>

                <span className={styles.cosSim} style={{ background: bg, color: fg }}>
                    {pair.cos_sim.toFixed(4)}
                </span>

                <span className={styles.corrTag} style={{ background: '#eff6ff', borderColor: '#bfdbfe', color: '#1d4ed8' }}>
                    <span className={styles.corrLabel}>test </span>
                    <strong>{pair.test_correlation.source_token.trim() || '·'}</strong>
                    <span className={styles.arrow}> → </span>
                    <strong>{pair.test_correlation.target_token.trim() || '·'}</strong>
                </span>

                <span className={styles.corrArrow}>⇔</span>

                <span className={styles.corrTag} style={{ background: '#fffbeb', borderColor: '#fde68a', color: '#92400e' }}>
                    <span className={styles.corrLabel}>train </span>
                    <strong>{pair.train_correlation.source_token.trim() || '·'}</strong>
                    <span className={styles.arrow}> → </span>
                    <strong>{pair.train_correlation.target_token.trim() || '·'}</strong>
                    <span className={styles.offset}>+{pair.train_correlation.response_token_offset}</span>
                </span>

                <span className={styles.trainBadge}>TRAIN #{pair.train_sample_id}</span>
                <span className={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
            </div>

            {expanded && (
                <div className={styles.pairCardBody}>
                    <div className={styles.contextRow}>
                        <div>
                            <div className={styles.contextRowLabel}>Source Context</div>
                            <ContextChip tokens={pair.train_context.source_context} />
                        </div>
                        <div className={styles.contextDivider} />
                        <div>
                            <div className={styles.contextRowLabel}>Target Context</div>
                            <ContextChip tokens={pair.train_context.target_context} />
                        </div>
                    </div>
                    {detail && (
                        <TrainSampleViewer detail={detail} highlightPairs={[pair]} />
                    )}
                </div>
            )}
        </div>
    );
}

// ─── Train Sample Group ───────────────────────────────────────────────────────

function TrainSampleGroup({
    trainIdx,
    pairs,
    detail,
    onProbeEmbeddings,
    probeBusy,
    selectedPairIds,
    onTogglePairSelection,
    comparisonSummary,
}: {
    trainIdx: number;
    pairs: CorrelationPair[];
    detail?: TrainSampleDetail;
    onProbeEmbeddings?: (trainIdx: number, pairs: CorrelationPair[]) => void;
    probeBusy?: boolean;
    selectedPairIds?: string[];
    onTogglePairSelection?: (trainIdx: number, pairId: string) => void;
    comparisonSummary?: TrainProbeComparisonSummary;
}) {
    const [collapsed, setCollapsed] = useState(false);
    const bestSim = Math.max(...pairs.map(p => p.cos_sim));
    const { bg, fg } = cosSimilarityColor(bestSim);
    const selectedPairIdSet = useMemo(() => new Set(selectedPairIds ?? []), [selectedPairIds]);
    const selectedPairCount = selectedPairIdSet.size;
    const comparisonTokens = comparisonSummary?.focusTokens ?? [];
    const comparisonPairs = comparisonSummary?.pairwiseCosine ?? [];

    return (
        <div className={styles.trainGroup}>
            <div className={styles.trainGroupHeader} onClick={() => setCollapsed(c => !c)}>
                <span className={styles.trainGroupId}>TRAIN #{trainIdx}</span>
                <span className={styles.trainGroupCoarse}>coarse {(detail?.coarse_cos_sim ?? pairs[0]?.coarse_cos_sim ?? 0).toFixed(4)}</span>
                <span className={styles.trainGroupCount}>{pairs.length} pairs</span>
                <span className={styles.cosSim} style={{ background: bg, color: fg }}>best {bestSim.toFixed(4)}</span>
                <span style={{ marginLeft: 8, fontSize: 11, color: '#6b7280', fontWeight: 600 }}>
                    已选 {selectedPairCount}
                </span>
                {detail && onProbeEmbeddings && (
                    <button
                        type="button"
                        onClick={event => {
                            event.stopPropagation();
                            onProbeEmbeddings(trainIdx, pairs);
                        }}
                        disabled={probeBusy}
                        style={{
                            marginLeft: 8,
                            padding: '4px 10px',
                            borderRadius: 999,
                            border: '1px solid #c4b5fd',
                            background: probeBusy ? '#ede9fe' : '#faf5ff',
                            color: '#6d28d9',
                            fontSize: 11,
                            fontWeight: 700,
                            cursor: probeBusy ? 'wait' : 'pointer',
                        }}
                    >
                        {probeBusy ? 'Probing…' : 'Open Full Probe'}
                    </button>
                )}
                <span className={styles.expandIcon} style={{ marginLeft: 'auto' }}>{collapsed ? '▶' : '▼'}</span>
            </div>
            {!collapsed && (
                <div className={styles.trainGroupBody}>
                    {detail && (
                        <div className={styles.trainFullView}>
                            <div className={styles.subLabel}>完整训练样本 — 全量 token 进入 probe，当前 correlation token 特殊标出</div>
                            <TrainSampleViewer detail={detail} highlightPairs={pairs} />
                        </div>
                    )}
                    {selectedPairCount > 0 && comparisonTokens.length > 0 && (
                        <div style={{ padding: '0 0 14px' }}>
                            <div className={styles.subLabel}>已选 token（用于 probe 高亮与数值比较）</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
                                {comparisonTokens.map(token => (
                                    <span
                                        key={`${token.tokenIndex}-${token.pointIndex ?? 'na'}`}
                                        style={{
                                            fontSize: 12,
                                            padding: '4px 8px',
                                            borderRadius: 999,
                                            background: '#f5f3ff',
                                            color: '#6d28d9',
                                            border: '1px solid #ddd6fe',
                                            fontFamily: 'monospace',
                                        }}
                                    >
                                        {token.tokenDisplay} @{token.tokenIndex}
                                    </span>
                                ))}
                            </div>
                            {comparisonPairs.length > 0 && (
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
                                    {comparisonPairs.map(pair => (
                                        <div
                                            key={`${pair.leftIndex}-${pair.rightIndex}`}
                                            style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: 10,
                                                fontSize: 12,
                                                color: '#475569',
                                            }}
                                        >
                                            <span style={{ fontFamily: 'monospace', color: '#7c2d12', minWidth: 200 }}>
                                                {pair.leftTokenDisplay} @{pair.leftIndex} ↔ {pair.rightTokenDisplay} @{pair.rightIndex}
                                            </span>
                                            <span style={{ fontWeight: 700, color: '#b91c1c' }}>{pair.cosine.toFixed(4)}</span>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                    <div className={styles.pairList}>
                        {pairs.map(pair => (
                            <PairCard
                                key={pair.id}
                                pair={pair}
                                detail={detail}
                                selected={selectedPairIdSet.has(pair.id)}
                                onToggleSelect={onTogglePairSelection ? () => onTogglePairSelection(trainIdx, pair.id) : undefined}
                            />
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

// ─── Main NewView Component ───────────────────────────────────────────────────

export interface AllTokensExperimentMeta {
    taskId: string;
    label: string;
    fileName: string;
}

export interface ReportPanelProps {
    report: AllTokensReport;
    meta: AllTokensExperimentMeta;
    modelLabel?: string;
    /** When provided with onSelectedTokIdxChange, token selection is controlled by parent (for dual-link). */
    selectedTokIdx?: number | null;
    onSelectedTokIdxChange?: (idx: number | null) => void;
    compact?: boolean;
}

export function ReportPanel({
    report,
    meta,
    modelLabel,
    selectedTokIdx: controlledTokIdx,
    onSelectedTokIdxChange,
    compact = false,
}: ReportPanelProps) {
    const importedReportActive = meta.fileName.startsWith('uploaded:') || meta.label.includes('(uploaded)');
    const selectedMeta = meta;

    // Selected output token (by absolute sequence index)
    const [internalTokIdx, setInternalTokIdx] = useState<number | null>(null);
    const selectedTokIdx = controlledTokIdx !== undefined ? controlledTokIdx : internalTokIdx;
    const setSelectedTokIdx = (updater: number | null | ((prev: number | null) => number | null)) => {
        const prev = selectedTokIdx;
        const next = typeof updater === 'function' ? updater(prev) : updater;
        if (onSelectedTokIdxChange) onSelectedTokIdxChange(next);
        else setInternalTokIdx(next);
    };
    // Selected test correlation (source_token_index)
    const [selectedTestCorrIdx, setSelectedTestCorrIdx] = useState<number | null>(null);
    // cos_sim filter threshold
    const [threshold, setThreshold] = useState(0.0);
    // Whether to hide pairs with cos_sim exactly 0
    const [hideZero, setHideZero] = useState(false);

    // Export state
    const [exportScope, setExportScope] = useState<'all' | 'selected' | 'range'>('all');
    const [rangeFrom, setRangeFrom] = useState(0);
    const [rangeTo, setRangeTo] = useState(0);
    const [ttavUrl, setTtavUrl] = useState(() => loadTtavLaunchPrefs().ttavUrl);
    const [ttavContentPathTemplate] = useState(() => loadTtavLaunchPrefs().contentPathTemplate);
    const [eifBundleCacheTemplate, setEifBundleCacheTemplate] = useState(() => loadTtavLaunchPrefs().eifBundleCacheTemplate);
    const [ttavVisMethod, setTtavVisMethod] = useState(() => loadTtavLaunchPrefs().visMethod);
    const [ttavVisId, setTtavVisId] = useState(() => loadTtavLaunchPrefs().visId);
    const [eifApiUrl] = useState(() => loadTtavLaunchPrefs().eifApiUrl);
    const [ttavLaunchError, setTtavLaunchError] = useState<string | null>(null);
    const [ttavLaunchStatus, setTtavLaunchStatus] = useState<string | null>(null);
    const [ttavPrepareDetail, setTtavPrepareDetail] = useState<string | null>(null);
    const [preparingTtavBundle, setPreparingTtavBundle] = useState(false);
    const [probingTrainSampleId, setProbingTrainSampleId] = useState<number | null>(null);
    const [selectedTrainPairIdsByGroup, setSelectedTrainPairIdsByGroup] = useState<Record<number, string[]>>({});
    const [trainProbeComparisons, setTrainProbeComparisons] = useState<Record<number, TrainProbeComparisonSummary>>({});
    const [showAdvancedTtav, setShowAdvancedTtav] = useState(false);
    const ttavWindowRef = useRef<Window | null>(null);
    const ttavWindowOriginRef = useRef<string | null>(null);
    const ttavWindowSampleIdRef = useRef<string | null>(null);

    const displayModelLabel = modelLabel
        || report.experiment_meta.model_name
        || meta.label
        || 'model';

    // Reset secondary selection when selected token changes
    useEffect(() => { setSelectedTestCorrIdx(null); }, [selectedTokIdx]);

    useEffect(() => {
        setSelectedTrainPairIdsByGroup({});
        setTrainProbeComparisons({});
    }, [selectedTokIdx, selectedTestCorrIdx]);

    useEffect(() => {
        if (typeof window === 'undefined') return;
        window.localStorage.setItem(TTAV_PREFS_KEY, JSON.stringify({
            ttavUrl,
            contentPathTemplate: ttavContentPathTemplate,
            eifBundleCacheTemplate,
            visMethod: ttavVisMethod,
            visId: ttavVisId,
            eifApiUrl,
        } satisfies TtavLaunchPrefs));
    }, [ttavUrl, ttavContentPathTemplate, eifBundleCacheTemplate, ttavVisMethod, ttavVisId, eifApiUrl]);

    const modelTokens   = useMemo(() => decodeTokens(report.test_sample_baseline.full_tokens), [report]);
    const correctTokens = useMemo(() => decodeTokens(report.test_sample_baseline.correct_full_tokens ?? []), [report]);
    const promptLen     = report?.test_sample_baseline.prompt_len ?? 0;
    const selectedSampleId = inferSampleIdFromMeta(selectedMeta, report);
    const resolvedEifBundleCachePath = selectedSampleId
        ? resolveContentPath(eifBundleCacheTemplate, selectedSampleId)
        : '';

    // Map from token index → PerTokenResult for quick lookup
    const perTokenMap = useMemo(() => {
        const m = new Map<number, PerTokenResult>();
        report?.per_token_results.forEach(r => m.set(r.target_token_index, r));
        return m;
    }, [report]);

    // Indices of analyzed output tokens (those with per_token_results)
    const analyzedIndices = useMemo(() => new Set(perTokenMap.keys()), [perTokenMap]);

    const selectedResult = selectedTokIdx !== null ? perTokenMap.get(selectedTokIdx) ?? null : null;

    // Source token highlights for the selected token
    const sourceHighlightIndices = useMemo(() => {
        if (!selectedResult) return new Set<number>();
        if (selectedTestCorrIdx !== null) return new Set([selectedTestCorrIdx]);
        return new Set(selectedResult.top_correlations.map(c => c.source_token_index));
    }, [selectedResult, selectedTestCorrIdx]);

    const ttavSelectedIndices = useMemo(() => {
        const selected = new Set<number>();
        if (selectedTokIdx !== null) selected.add(selectedTokIdx);
        sourceHighlightIndices.forEach(idx => selected.add(idx));
        return Array.from(selected).sort((a, b) => a - b);
    }, [selectedTokIdx, sourceHighlightIndices]);

    // Pairs to show in the right panel — only after the user picks one test (s→t) edge.
    const selectedTestCorr = useMemo(() => {
        if (!selectedResult || selectedTestCorrIdx === null) return null;
        return selectedResult.top_correlations.find(c => c.source_token_index === selectedTestCorrIdx) ?? null;
    }, [selectedResult, selectedTestCorrIdx]);

    const allDisplayPairs = useMemo(() => {
        const keep = (p: CorrelationPair) =>
            p.cos_sim >= threshold && !(hideZero && p.cos_sim === 0);

        // Require an explicit Top-Correlation click before showing train matches.
        if (!selectedResult || selectedTestCorrIdx === null) return [];

        return selectedResult.correlation_pairs
            .filter(p =>
                keep(p) && p.test_correlation.source_token_index === selectedTestCorrIdx
            )
            .sort((a, b) => b.cos_sim - a.cos_sim);
    }, [selectedResult, selectedTestCorrIdx, threshold, hideZero]);

    // Group pairs by train_sample_id; keep Top-10 trains by best pair cos for this edge.
    const trainGroups = useMemo(() => {
        const map = new Map<number, CorrelationPair[]>();
        allDisplayPairs.forEach(p => {
            if (!map.has(p.train_sample_id)) map.set(p.train_sample_id, []);
            map.get(p.train_sample_id)!.push(p);
        });
        return Array.from(map.entries())
            .map(([id, pairs]) => ({
                id,
                pairs: [...pairs].sort((a, b) => b.cos_sim - a.cos_sim),
                bestSim: Math.max(...pairs.map(p => p.cos_sim)),
            }))
            .sort((a, b) => b.bestSim - a.bestSim)
            .slice(0, 10);
    }, [allDisplayPairs]);

    const trainPanelEmptyHint = useMemo(() => {
        if (!selectedResult) {
            return 'Click an analyzed output token, then choose one Top Correlation (source→target) to retrieve related training samples.';
        }
        if (selectedTestCorrIdx === null) {
            return 'Select one Top Correlation on the left to show its Top-10 related training samples and matching pairs.';
        }
        if (importedReportActive && allDisplayPairs.length === 0) {
            return 'No training correlation pairs are included for this selected source→target edge.';
        }
        return 'No matching pairs for this source→target edge. Try lowering the cos_sim threshold.';
    }, [selectedResult, selectedTestCorrIdx, importedReportActive, allDisplayPairs.length]);

    // ── Export logic ─────────────────────────────────────────────────────────

    // Initialize range bounds when report changes
    useEffect(() => {
        if (report && report.per_token_results.length > 0) {
            const indices = report.per_token_results.map(r => r.target_token_index);
            setRangeFrom(Math.min(...indices));
            setRangeTo(Math.max(...indices));
        }
    }, [report]);

    const buildCurrentTtavPayload = useCallback((): TtavJumpPayload | null => {
        if (!report || !selectedMeta) return null;

        const sampleId = selectedSampleId;
        const preparedBundle = getPreparedTtavBundle(sampleId);
        const visMethod = preparedBundle?.visMethod || (ttavVisMethod.trim() || DEFAULT_TTAV_METHOD);
        const visId = preparedBundle?.visId || (ttavVisId.trim() || DEFAULT_TTAV_VIS_ID);
        const contentPath = preparedBundle?.contentPath || resolveContentPath(ttavContentPathTemplate, sampleId);

        return {
            source: 'eif',
            sampleId,
            contentPath,
            visMethod,
            visId,
            dataType: 'Text',
            taskType: 'Alignment',
            selectedIndices: ttavSelectedIndices,
            targetIndex: selectedTokIdx ?? undefined,
            selectedSourceIndex: selectedTestCorrIdx ?? undefined,
            promptLen,
        };
    }, [
        promptLen,
        report,
        selectedMeta,
        selectedSampleId,
        selectedTestCorrIdx,
        selectedTokIdx,
        ttavContentPathTemplate,
        ttavSelectedIndices,
        ttavVisId,
        ttavVisMethod,
    ]);

    const postTtavHighlightUpdate = useCallback((payload?: TtavJumpPayload | null) => {
        const ttavWindow = ttavWindowRef.current;
        const ttavOrigin = ttavWindowOriginRef.current;
        const nextPayload = payload ?? buildCurrentTtavPayload();

        if (!ttavWindow || !ttavOrigin || ttavWindow.closed || !nextPayload) {
            return;
        }

        if (ttavWindowSampleIdRef.current && ttavWindowSampleIdRef.current !== nextPayload.sampleId) {
            return;
        }

        const message: TtavHighlightUpdateMessage = {
            command: 'eifHighlightUpdate',
            data: nextPayload,
        };
        ttavWindow.postMessage(message, ttavOrigin);
    }, [buildCurrentTtavPayload]);

    const handleExport = async () => {
        const md = generateExportMarkdown(report, {
            tokenRange: exportScope,
            selectedTokenIdx: selectedTokIdx ?? undefined,
            rangeFrom, rangeTo,
            cosSimThreshold: threshold, hideZero,
        });
        const modelPart = report.experiment_meta.model_name ? `_${report.experiment_meta.model_name}` : '';
        downloadMarkdown(md, `attribution_report${modelPart}_test${report.experiment_meta.test_sample_index}.md`);
    };

    const buildTtavLaunchUrl = (payload: TtavJumpPayload): string => {
        const url = new URL(ttavUrl.trim());
        url.searchParams.set('eif_jump', JSON.stringify(payload));
        return url.toString();
    };

    const navigateTtavWindow = (targetWindow: Window, payload: TtavJumpPayload) => {
        targetWindow.location.href = buildTtavLaunchUrl(payload);
        ttavWindowRef.current = targetWindow;
        ttavWindowOriginRef.current = new URL(ttavUrl.trim()).origin;
        ttavWindowSampleIdRef.current = payload.sampleId;
        window.setTimeout(() => {
            postTtavHighlightUpdate(payload);
        }, 1200);
    };

    const openTtavWithPayload = (payload: TtavJumpPayload) => {
        const launchUrl = buildTtavLaunchUrl(payload);
        const openedWindow = window.open(launchUrl, '_blank');
        if (!openedWindow) return null;

        ttavWindowRef.current = openedWindow;
        ttavWindowOriginRef.current = new URL(ttavUrl.trim()).origin;
        ttavWindowSampleIdRef.current = payload.sampleId;

        window.setTimeout(() => {
            postTtavHighlightUpdate(payload);
        }, 1200);
        return openedWindow;
    };

    const callEifBundleApi = async (requireCached: boolean) => {
        if (!report || !selectedMeta) return null;

        const sampleId = selectedSampleId;
        const trimmedApiUrl = eifApiUrl.trim();
        const trimmedUrl = ttavUrl.trim();
        const visMethod = ttavVisMethod.trim() || DEFAULT_TTAV_METHOD;
        const visId = ttavVisId.trim() || DEFAULT_TTAV_VIS_ID;

        if (!trimmedApiUrl) {
            throw new Error('EIF API URL is required.');
        }

        const apiResp = await fetch(trimmedApiUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                reportFileName: selectedMeta.fileName,
                sampleId,
                testData: 'sft_test.jsonl',
                modelPath: null,
                ttavUploadUrl: new URL('/registerEIFBundle', trimmedUrl).toString(),
                ttavUrl: trimmedUrl,
                visMethod,
                visId,
                eifBundleCachePath: resolvedEifBundleCachePath,
                selectedIndices: ttavSelectedIndices,
                targetIndex: selectedTokIdx ?? undefined,
                requireCached,
            }),
        });

        const rawText = await apiResp.text();
        let parsedJson: Record<string, unknown> | null = null;
        if (rawText.trim()) {
            try {
                parsedJson = JSON.parse(rawText) as Record<string, unknown>;
            } catch {
                throw new Error(
                    `EIF API returned a non-JSON response (HTTP ${apiResp.status}). ` +
                    `${rawText.slice(0, 240)}`
                );
            }
        }

        const apiJson = parsedJson ?? {};
        if (!apiResp.ok || apiJson.status !== 'success') {
            const baseMessage = 'EIF bundle API failed (HTTP ' + apiResp.status + ')';
            const message = typeof apiJson.message === 'string'
                ? baseMessage + ': ' + apiJson.message
                : baseMessage;
            throw new Error(message);
        }

        return {
            sampleId: typeof apiJson.sampleId === 'string' ? apiJson.sampleId : sampleId,
            contentPath: typeof apiJson.contentPath === 'string'
                ? apiJson.contentPath
                : resolveContentPath(ttavContentPathTemplate, sampleId),
            visMethod: typeof apiJson.visMethod === 'string' ? apiJson.visMethod : visMethod,
            visId: typeof apiJson.visId === 'string' ? apiJson.visId : visId,
            eifBundleCachePath: typeof apiJson.eifBundleCachePath === 'string' ? apiJson.eifBundleCachePath : resolvedEifBundleCachePath,
            eifCacheHit: apiJson.eifCacheHit === true,
            ttavCached: typeof apiJson.uploadResult === 'object' && apiJson.uploadResult !== null && (apiJson.uploadResult as { cached?: boolean }).cached === true,
        };
    };

    const handleOpenInTtav = () => {
        if (!report || !selectedMeta) return;

        const sampleId = selectedSampleId;
        const trimmedUrl = ttavUrl.trim();
        const visMethod = ttavVisMethod.trim() || DEFAULT_TTAV_METHOD;
        const visId = ttavVisId.trim() || DEFAULT_TTAV_VIS_ID;
        if (!trimmedUrl) {
            setTtavLaunchError('TTAV URL is required.');
            return;
        }

        const optimisticPayload = buildCurrentTtavPayload();
        if (!optimisticPayload) {
            setTtavLaunchError('Unable to build TTAV jump payload.');
            return;
        }

        const openedWindow = openTtavWithPayload(optimisticPayload);
        if (!openedWindow) {
            setTtavLaunchError('Browser blocked the Visualizer window. Please allow pop-ups for this page.');
            return;
        }

        void (async () => {
            setTtavLaunchError(null);
            setTtavLaunchStatus(`Visualizer opening for ${optimisticPayload.sampleId}...`);
            try {
                new URL(trimmedUrl);
                const apiResult = await callEifBundleApi(true);
                if (!apiResult) return;
                savePreparedTtavBundle({
                    sampleId: apiResult.sampleId,
                    contentPath: apiResult.contentPath,
                    visMethod: apiResult.visMethod,
                    visId: apiResult.visId,
                    preparedAt: Date.now(),
                });
                const payload: TtavJumpPayload = {
                    source: 'eif',
                    sampleId: apiResult.sampleId,
                    contentPath: apiResult.contentPath,
                    visMethod: apiResult.visMethod,
                    visId: apiResult.visId,
                    dataType: 'Text',
                    taskType: 'Alignment',
                    selectedIndices: ttavSelectedIndices,
                    targetIndex: selectedTokIdx ?? undefined,
                    selectedSourceIndex: selectedTestCorrIdx ?? undefined,
                    promptLen,
                };
                if (!openedWindow.closed && buildTtavLaunchUrl(payload) !== buildTtavLaunchUrl(optimisticPayload)) {
                    navigateTtavWindow(openedWindow, payload);
                }
                setTtavLaunchStatus(`Visualizer opened for ${apiResult.sampleId}.`);
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Failed to open Visualizer';
                if (!shouldFallbackToDirectPrepare(msg)) {
                    setTtavLaunchError(msg);
                    setTtavLaunchStatus(null);
                    return;
                }

                const preparedBundle = getPreparedTtavBundle(sampleId);
                const fallbackContentPath = preparedBundle?.contentPath || resolveContentPath(ttavContentPathTemplate, sampleId);
                const fallbackVisMethod = preparedBundle?.visMethod || visMethod;
                const fallbackVisId = preparedBundle?.visId || visId;
                const payload: TtavJumpPayload = {
                    source: 'eif',
                    sampleId,
                    contentPath: fallbackContentPath,
                    visMethod: fallbackVisMethod,
                    visId: fallbackVisId,
                    dataType: 'Text',
                    taskType: 'Alignment',
                    selectedIndices: ttavSelectedIndices,
                    targetIndex: selectedTokIdx ?? undefined,
                    selectedSourceIndex: selectedTestCorrIdx ?? undefined,
                    promptLen,
                };
                if (!openedWindow.closed && buildTtavLaunchUrl(payload) !== buildTtavLaunchUrl(optimisticPayload)) {
                    navigateTtavWindow(openedWindow, payload);
                }
                setTtavLaunchError(null);
                setTtavLaunchStatus(preparedBundle
                    ? `Visualizer opened for ${sampleId} using the most recently prepared TTAV bundle.`
                    : `Visualizer opened for ${sampleId} using existing TTAV bundle.`);
            }
        })();
    };


    const toggleTrainPairSelection = (trainIdx: number, pairId: string) => {
        setSelectedTrainPairIdsByGroup(current => {
            const prev = new Set(current[trainIdx] ?? []);
            if (prev.has(pairId)) {
                prev.delete(pairId);
            } else {
                prev.add(pairId);
            }
            return {
                ...current,
                [trainIdx]: Array.from(prev),
            };
        });
        setTrainProbeComparisons(current => {
            if (!(trainIdx in current)) return current;
            const next = { ...current };
            delete next[trainIdx];
            return next;
        });
    };

    const handleOpenTrainProbe = (trainIdx: number, pairs: CorrelationPair[]) => {
        if (!report || !selectedMeta) return;

        const trimmedUrl = ttavUrl.trim();
        const visMethod = ttavVisMethod.trim() || DEFAULT_TTAV_METHOD;
        const visId = ttavVisId.trim() || DEFAULT_TTAV_VIS_ID;
        if (!trimmedUrl) {
            setTtavLaunchError('TTAV URL is required.');
            return;
        }

        const openedWindow = window.open(trimmedUrl, '_blank');
        if (!openedWindow) {
            setTtavLaunchError('Browser blocked the probe window. Please allow pop-ups for this page.');
            return;
        }

        const selectedPairIdSet = new Set(selectedTrainPairIdsByGroup[trainIdx] ?? []);
        const selectedPairs = selectedPairIdSet.size > 0
            ? pairs.filter(pair => selectedPairIdSet.has(pair.id))
            : [];
        const focusTrainIndices = Array.from(new Set(selectedPairs.flatMap(pair => [
            pair.train_correlation.source_token_index,
            pair.train_correlation.target_token_index,
        ]))).sort((a, b) => a - b);

        setProbingTrainSampleId(trainIdx);
        setTtavLaunchError(null);
        setTtavLaunchStatus(`Preparing full-train embedding probe for TRAIN #${trainIdx}...`);

        void (async () => {
            try {
                const probeResp = await fetch(buildEifApiUrl(eifApiUrl, '/api/prepare-ttav-train-probe'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        reportFileName: selectedMeta.fileName,
                        sampleId: selectedSampleId,
                        trainSampleId: trainIdx,
                        ttavUploadUrl: new URL('/registerEIFBundle', trimmedUrl).toString(),
                        ttavUrl: trimmedUrl,
                        visMethod,
                        visId,
                        contextRadius: 1,
                        includeFullTrain: true,
                        focusTrainIndices,
                        probePairs: pairs.map(pair => ({
                            id: pair.id,
                            trainSourceIndex: pair.train_correlation.source_token_index,
                            trainTargetIndex: pair.train_correlation.target_token_index,
                            testSourceIndex: pair.test_correlation.source_token_index,
                            testTargetIndex: pair.test_correlation.target_token_index,
                        })),
                    }),
                });

                const rawText = await probeResp.text();
                let parsedJson: Record<string, unknown> | null = null;
                if (rawText.trim()) {
                    try {
                        parsedJson = JSON.parse(rawText) as Record<string, unknown>;
                    } catch {
                        throw new Error(
                            `EIF train probe API returned a non-JSON response (HTTP ${probeResp.status}). ` +
                            `${rawText.slice(0, 240)}`
                        );
                    }
                }

                const apiJson = parsedJson ?? {};
                if (!probeResp.ok || apiJson.status !== 'success') {
                    const baseMessage = 'EIF train probe API failed (HTTP ' + probeResp.status + ')';
                    const message = typeof apiJson.message === 'string'
                        ? baseMessage + ': ' + apiJson.message
                        : baseMessage;
                    throw new Error(message);
                }

                const comparisonSummary = (typeof apiJson.comparisonSummary === 'object' && apiJson.comparisonSummary !== null)
                    ? apiJson.comparisonSummary as TrainProbeComparisonSummary
                    : { focusTokens: [], pairwiseCosine: [] };
                setTrainProbeComparisons(current => ({
                    ...current,
                    [trainIdx]: comparisonSummary,
                }));

                const browserUploadRequired = apiJson.browserUploadRequired === true;
                let resolvedSampleId = typeof apiJson.sampleId === 'string'
                    ? apiJson.sampleId
                    : `${selectedSampleId}_train${trainIdx}_probe`;
                let resolvedContentPath = typeof apiJson.contentPath === 'string' ? apiJson.contentPath : '';
                let resolvedVisMethod = typeof apiJson.visMethod === 'string' ? apiJson.visMethod : visMethod;
                let resolvedVisId = typeof apiJson.visId === 'string' ? apiJson.visId : visId;

                if (browserUploadRequired) {
                    setTtavLaunchStatus(`Probe computed on EIF; uploading TRAIN #${trainIdx} probe to TTAV from browser...`);
                    const bundlePayload = (typeof apiJson.bundlePayload === 'object' && apiJson.bundlePayload !== null)
                        ? apiJson.bundlePayload
                        : null;
                    if (!bundlePayload) {
                        throw new Error(typeof apiJson.uploadError === 'string'
                            ? apiJson.uploadError
                            : 'Probe bundle upload fallback payload is missing.');
                    }

                    const uploadUrl = new URL('/registerEIFBundle', trimmedUrl).toString();
                    const uploadResp = await fetch(uploadUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(bundlePayload),
                    });
                    const uploadRawText = await uploadResp.text();
                    let uploadJson: Record<string, unknown> | null = null;
                    if (uploadRawText.trim()) {
                        try {
                            uploadJson = JSON.parse(uploadRawText) as Record<string, unknown>;
                        } catch {
                            throw new Error(
                                `TTAV backend returned a non-JSON response (HTTP ${uploadResp.status}). ` +
                                `${uploadRawText.slice(0, 240)}`
                            );
                        }
                    }

                    const uploadApiJson = uploadJson ?? {};
                    if (!uploadResp.ok || uploadApiJson.status !== 'success') {
                        const message = typeof uploadApiJson.message === 'string'
                            ? uploadApiJson.message
                            : `Failed to register TTAV probe bundle (HTTP ${uploadResp.status})`;
                        throw new Error(message);
                    }

                    resolvedSampleId = typeof uploadApiJson.sample_id === 'string'
                        ? uploadApiJson.sample_id
                        : (typeof uploadApiJson.sampleId === 'string' ? uploadApiJson.sampleId : resolvedSampleId);
                    resolvedContentPath = typeof uploadApiJson.content_path === 'string'
                        ? uploadApiJson.content_path
                        : (typeof uploadApiJson.contentPath === 'string' ? uploadApiJson.contentPath : resolvedContentPath);
                    resolvedVisMethod = typeof uploadApiJson.vis_method === 'string'
                        ? uploadApiJson.vis_method
                        : (typeof uploadApiJson.visMethod === 'string' ? uploadApiJson.visMethod : resolvedVisMethod);
                    resolvedVisId = typeof uploadApiJson.vis_id === 'string'
                        ? uploadApiJson.vis_id
                        : (typeof uploadApiJson.visId === 'string' ? uploadApiJson.visId : resolvedVisId);
                }

                const payload: TtavJumpPayload = {
                    source: 'eif',
                    sampleId: resolvedSampleId,
                    contentPath: resolvedContentPath,
                    visMethod: resolvedVisMethod,
                    visId: resolvedVisId,
                    dataType: 'Text',
                    taskType: 'Alignment',
                    selectedIndices: Array.isArray(apiJson.selectedIndices)
                        ? apiJson.selectedIndices.filter((value): value is number => typeof value === 'number')
                        : [],
                    targetIndex: typeof apiJson.targetIndex === 'number' ? apiJson.targetIndex : undefined,
                    promptLen: typeof apiJson.promptLen === 'number' ? apiJson.promptLen : 0,
                };

                navigateTtavWindow(openedWindow, payload);
                setTtavLaunchStatus(browserUploadRequired
                    ? `Full-train probe opened for TRAIN #${trainIdx} using browser upload fallback.`
                    : `Full-train probe opened for TRAIN #${trainIdx}.`);
            } catch (error) {
                const msg = error instanceof Error ? error.message : 'Failed to open embedding probe';
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
                if (!openedWindow.closed) {
                    openedWindow.close();
                }
            } finally {
                setProbingTrainSampleId(current => (current === trainIdx ? null : current));
            }
        })();
    };

    useEffect(() => {
        if (!ttavWindowRef.current || ttavWindowRef.current.closed) return;
        if (!selectedSampleId || ttavWindowSampleIdRef.current !== selectedSampleId) return;

        postTtavHighlightUpdate();
    }, [postTtavHighlightUpdate, selectedSampleId]);

    const handlePrepareTtavBundle = async () => {
        if (!report || !selectedMeta) return;

        const sampleId = selectedSampleId;
        const trimmedUrl = ttavUrl.trim();
        const visMethod = ttavVisMethod.trim() || DEFAULT_TTAV_METHOD;
        const visId = ttavVisId.trim() || DEFAULT_TTAV_VIS_ID;

        setPreparingTtavBundle(true);
        setTtavLaunchError(null);
        setTtavLaunchStatus(null);

        const eifApiHost = (() => {
            try {
                return new URL(eifApiUrl).host;
            } catch {
                return eifApiUrl;
            }
        })();
        setTtavPrepareDetail(`Connecting to EIF API (${eifApiHost})...`);

        let statusPollTimer: number | null = null;
        let statusPollActive = true;
        const pollPrepareStatus = async () => {
            if (!statusPollActive) return;
            try {
                const statusPayload = await fetchEifPrepareStatus(eifApiUrl, sampleId);
                if (!statusPollActive || !statusPayload) return;
                setTtavPrepareDetail(statusPayload.message);
            } catch {
                // Ignore status polling failures and let the main request decide fallback behavior.
            }
        };

        try {
            const initialStatus = await fetchEifPrepareStatus(eifApiUrl, sampleId, 2500);
            if (!initialStatus) {
                throw new Error(`Unable to reach EIF API at ${eifApiHost}`);
            }

            setTtavPrepareDetail(`Submitting prepare request to EIF API (${eifApiHost})...`);
            void pollPrepareStatus();
            statusPollTimer = window.setInterval(() => {
                void pollPrepareStatus();
            }, 1000);

            const requestSubmittedTimer = window.setTimeout(() => {
                setTtavPrepareDetail('EIF API request submitted. Waiting for server-side embedding computation...');
            }, 1200);

            const apiResult = await callEifBundleApi(false);
            window.clearTimeout(requestSubmittedTimer);
            if (!apiResult) return;
            savePreparedTtavBundle({
                sampleId: apiResult.sampleId,
                contentPath: apiResult.contentPath,
                visMethod: apiResult.visMethod,
                visId: apiResult.visId,
                preparedAt: Date.now(),
            });
            const eifMsg = apiResult.eifCacheHit
                ? 'EIF cache reused'
                : 'EIF cache created';
            const ttavMsg = apiResult.ttavCached
                ? 'TTAV cache reused'
                : 'sent to TTAV';
            setTtavLaunchStatus(`${apiResult.sampleId}: ${eifMsg}; ${ttavMsg}.`);
        } catch (error) {
            const msg = error instanceof Error ? error.message : 'Failed to prepare TTAV bundle';
            if (!shouldFallbackToDirectPrepare(msg) && !msg.toLowerCase().includes('unable to reach eif api')) {
                setTtavLaunchError(msg);
                setTtavLaunchStatus(null);
            } else {
                try {
                    setTtavPrepareDetail(`EIF API unreachable; uploading precomputed real bundle to TTAV...`);
                    const uploadUrl = new URL('/registerEIFBundle', trimmedUrl).toString();
                    const bundlePayload = await loadPrecomputedRealBundle(sampleId, visMethod, visId);
                    const uploadResp = await fetch(uploadUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(bundlePayload),
                    });
                    const rawText = await uploadResp.text();
                    let parsedJson: Record<string, unknown> | null = null;
                    if (rawText.trim()) {
                        try {
                            parsedJson = JSON.parse(rawText) as Record<string, unknown>;
                        } catch {
                            throw new Error(
                                `TTAV backend returned a non-JSON response (HTTP ${uploadResp.status}). ` +
                                `${rawText.slice(0, 240)}`
                            );
                        }
                    }
                    const apiJson = parsedJson ?? {};
                    if (!uploadResp.ok || apiJson.status !== 'success') {
                        const message = typeof apiJson.message === 'string'
                            ? apiJson.message
                            : `Failed to register TTAV bundle (HTTP ${uploadResp.status})`;
                        throw new Error(message);
                    }
                    const returnedSampleId = typeof apiJson.sample_id === 'string'
                        ? apiJson.sample_id
                        : (typeof apiJson.sampleId === 'string' ? apiJson.sampleId : sampleId);
                    const returnedContentPath = typeof apiJson.content_path === 'string'
                        ? apiJson.content_path
                        : (typeof apiJson.contentPath === 'string'
                            ? apiJson.contentPath
                            : resolveContentPath(ttavContentPathTemplate, returnedSampleId));
                    savePreparedTtavBundle({
                        sampleId: returnedSampleId,
                        contentPath: returnedContentPath,
                        visMethod,
                        visId,
                        preparedAt: Date.now(),
                    });
                    const ttavMsg = apiJson.cached === true ? 'TTAV cache reused' : 'sent to TTAV';
                    setTtavLaunchError(null);
                    setTtavLaunchStatus(`${returnedSampleId}: precomputed real bundle used; ${ttavMsg}.`);
                } catch (fallbackError) {
                    const fallbackMsg = fallbackError instanceof Error
                        ? fallbackError.message
                        : 'Failed to prepare TTAV bundle';
                    setTtavLaunchError(fallbackMsg);
                    setTtavLaunchStatus(null);
                }
            }
        } finally {
            statusPollActive = false;
            if (statusPollTimer !== null) {
                window.clearInterval(statusPollTimer);
            }
            setTtavPrepareDetail(null);
            setPreparingTtavBundle(false);
        }
    };

    // ── Render ──

    return (
        <div className={`${styles.root}${compact ? ` ${styles.panelRootCompact}` : ''}`}>
            <div className={styles.modelBanner}>
                <span className={styles.modelBannerLabel}>Model</span>
                <span className={styles.modelBannerName}>{displayModelLabel}</span>
                <span className={styles.modelBannerMeta}>
                    test#{report.experiment_meta.test_sample_index}
                    {report.experiment_meta.task_id ? ` · ${report.experiment_meta.task_id}` : ''}
                    {` · ${report.per_token_results.length} tokens`}
                </span>
            </div>

                    {/* ── Top: Ground Truth (Full width, scrolls normally) ── */}
                    <div className={styles.topPanel}>
                        <CodePanel
                            label="Correct Output (Ground Truth)"
                            badge="GT"
                            badgeColor="#16a34a"
                            tokens={correctTokens}
                            promptLen={promptLen}
                        />
                    </div>

                    {/* ── Bottom Section: Left Sticky, Right Scroll ── */}
                    <div className={styles.bottomSection}>
                        {/* ── Left Column: Model Output & Correlations ── */}
                        <div className={styles.bottomLeft}>
                            <CodePanel
                                label="Model Output (Incorrect)"
                                badge="MODEL"
                                badgeColor="#dc2626"
                                tokens={modelTokens}
                                promptLen={promptLen}
                                highlightSourceIndices={sourceHighlightIndices}
                                selectedTargetIndex={selectedTokIdx ?? undefined}
                                analyzedIndices={analyzedIndices}
                                onTokenClick={idx => setSelectedTokIdx(prev => prev === idx ? null : idx)}
                            />

                            {selectedResult && (
                                <div className={styles.correlationList}>
                                    <div className={styles.correlationListTitle}>
                                        Top Correlations for "{decodeToken(selectedResult.target_token).trim()}" @ idx {selectedResult.target_token_index}
                                    </div>
                                    <div className={styles.correlationListHint}>
                                        Click one source→target edge to load its Top-10 training matches on the right.
                                    </div>
                                    <div className={styles.correlationListItems}>
                                        {selectedResult.top_correlations.slice(0, 4).map(c => (
                                            <button
                                                key={c.source_token_index}
                                                type="button"
                                                title={`Select ${decodeToken(c.source_token).trim() || '·'} → ${decodeToken(selectedResult.target_token).trim()} for train retrieval`}
                                                className={`${styles.corrBtn} ${c.source_token_index === selectedTestCorrIdx ? styles.corrBtnActive : ''}`}
                                                onClick={() => setSelectedTestCorrIdx(
                                                    prev => prev === c.source_token_index ? null : c.source_token_index
                                                )}
                                            >
                                                <div className={styles.corrBtnLeft}>
                                                    <span className={styles.corrLabel}>source → target</span>
                                                    <span className={styles.corrSourceTok}>
                                                        {(decodeToken(c.source_token).trim() || '·')}
                                                        <span className={styles.corrArrow}>→</span>
                                                        {decodeToken(selectedResult.target_token).trim() || '·'}
                                                    </span>
                                                </div>
                                                <div className={styles.corrBtnRight}>
                                                    <span className={styles.corrScoreLabel}>saliency</span>
                                                    <span className={styles.sourceChipSal}>{c.saliency_score.toFixed(3)}</span>
                                                </div>
                                            </button>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* ── Right Column: Training pairs ── */}
                        <div className={styles.bottomRight}>
                            <div className={styles.bottomPanel}>
                                <div className={styles.bottomPanelHeader}>
                                    <div className={styles.bottomPanelTitle}>
                                        {selectedTestCorr && selectedResult
                                            ? `Training matches for "${decodeToken(selectedTestCorr.source_token).trim() || '·'} → ${decodeToken(selectedResult.target_token).trim()}"`
                                            : 'Training Correlations'}
                                        <span className={styles.pairCount}>
                                            {selectedTestCorrIdx === null
                                                ? 'select a Top Correlation'
                                                : `${trainGroups.length} trains · ${allDisplayPairs.length} pairs (Top-10)`}
                                        </span>
                                    </div>
                                    <div className={styles.filterRow}>
                                        <span className={styles.filterLabel}>cos_sim ≥</span>
                                        <input
                                            type="range" min={0} max={0.2} step={0.001} value={threshold}
                                            onChange={e => setThreshold(parseFloat(e.target.value))}
                                            className={styles.thresholdSlider}
                                        />
                                        <span className={styles.thresholdVal}>{threshold.toFixed(3)}</span>
                                        <button
                                            onClick={() => setHideZero(v => !v)}
                                            style={{
                                                marginLeft: '12px',
                                                padding: '3px 10px',
                                                borderRadius: '6px',
                                                fontSize: '11px',
                                                cursor: 'pointer',
                                                border: `1px solid ${hideZero ? '#ef4444' : '#d1d5db'}`,
                                                background: hideZero ? '#fef2f2' : '#fff',
                                                color: hideZero ? '#b91c1c' : '#6b7280',
                                                fontWeight: hideZero ? 700 : 500,
                                                transition: 'all 0.12s',
                                            }}
                                        >
                                            {hideZero ? '✗ 已隐藏 cos=0' : '隐藏 cos_sim=0'}
                                        </button>
                                    </div>
                                    {/* ── Export row ── */}
                                    <div style={{
                                        marginTop: 8, padding: '8px 0',
                                        borderTop: '1px solid #e5e7eb',
                                        display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 12, fontSize: 13,
                                    }}>
                                        <span style={{ fontWeight: 600, color: '#374151' }}>Export:</span>
                                        <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                                            <input type="radio" name="exportScope" value="all"
                                                checked={exportScope === 'all'} onChange={() => setExportScope('all')} />
                                            All tokens
                                        </label>
                                        {selectedTokIdx !== null && (
                                            <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                                                <input type="radio" name="exportScope" value="selected"
                                                    checked={exportScope === 'selected'} onChange={() => setExportScope('selected')} />
                                                Selected only
                                            </label>
                                        )}
                                        <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: 'pointer' }}>
                                            <input type="radio" name="exportScope" value="range"
                                                checked={exportScope === 'range'} onChange={() => setExportScope('range')} />
                                            Range:
                                            <input type="number" value={rangeFrom}
                                                onChange={e => setRangeFrom(+e.target.value)}
                                                disabled={exportScope !== 'range'}
                                                style={{ width: 56, padding: '2px 4px', border: '1px solid #d1d5db', borderRadius: 4 }} />
                                            <span>—</span>
                                            <input type="number" value={rangeTo}
                                                onChange={e => setRangeTo(+e.target.value)}
                                                disabled={exportScope !== 'range'}
                                                style={{ width: 56, padding: '2px 4px', border: '1px solid #d1d5db', borderRadius: 4 }} />
                                        </label>
                                        <button
                                            onClick={handleExport}
                                            style={{
                                                marginLeft: 'auto', padding: '4px 16px', borderRadius: 6,
                                                border: '1px solid #6366f1', background: '#eef2ff', color: '#4338ca',
                                                fontWeight: 600, fontSize: 13, cursor: 'pointer',
                                            }}
                                        >
                                            Export Markdown
                                        </button>
                                    </div>
                                    {importedReportActive ? (
                                        <div className={styles.importedReportNotice}>
                                            TTAV bundle preparation is available only for bundled experiment files.
                                            Imported saliency JSON can still be inspected here and exported to Markdown.
                                        </div>
                                    ) : (
                                    <div style={{
                                        marginTop: 8, paddingTop: 8,
                                        borderTop: '1px solid #e5e7eb',
                                        display: 'grid', gap: 8,
                                    }}>
                                        <div style={{
                                            display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center',
                                            justifyContent: 'space-between',
                                            fontSize: 12,
                                        }}>
                                                <div style={{ display: 'grid', gap: 4 }}>
                                                    <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                                                        <span style={{ fontWeight: 700, color: '#374151' }}>TTAV Jump</span>
                                                        <span style={{
                                                            padding: '2px 8px',
                                                        borderRadius: 999,
                                                        background: '#eff6ff',
                                                        color: '#1d4ed8',
                                                        fontSize: 11,
                                                        fontWeight: 600,
                                                        }}>
                                                            {ttavVisMethod}
                                                        </span>
                                                    </div>
                                                    <div style={{ color: '#6b7280' }}>
                                                        TTAV URL: <span>{ttavUrl}</span>
                                                    </div>
                                                    <div style={{ color: '#374151' }}>
                                                        当前 sample: <code>{selectedSampleId || '未选择 sample'}</code>
                                                    </div>
                                                    {resolvedEifBundleCachePath && (
                                                        <div style={{ color: '#374151' }}>
                                                            EIF Bundle Cache Path: <code>{resolvedEifBundleCachePath}</code>
                                                        </div>
                                                    )}
                                                    <div style={{ color: '#6b7280' }}>
                                                        先准备当前 sample 的 TTAV bundle，再打开 TTAV 查看并高亮当前 target token 和 attribution token。
                                                    </div>
                                                </div>
                                            <button
                                                onClick={() => setShowAdvancedTtav(v => !v)}
                                                style={{
                                                    padding: '4px 10px', borderRadius: 6,
                                                    border: '1px solid #cbd5e1', background: '#fff', color: '#475569',
                                                    fontSize: 12, cursor: 'pointer',
                                                }}
                                            >
                                                {showAdvancedTtav ? 'Hide advanced' : 'Advanced settings'}
                                            </button>
                                        </div>
                                        {showAdvancedTtav && (
                                            <div style={{ display: 'grid', gap: 8, fontSize: 12 }}>
                                                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                                                    <label style={{ display: 'grid', gap: 4 }}>
                                                        <span style={{ fontWeight: 600, color: '#374151' }}>EIF Bundle Cache Path</span>
                                                        <input
                                                            value={eifBundleCacheTemplate}
                                                            onChange={e => setEifBundleCacheTemplate(e.target.value)}
                                                            placeholder={DEFAULT_EIF_BUNDLE_CACHE_TEMPLATE}
                                                            style={{
                                                                minWidth: 320, padding: '4px 8px',
                                                                border: '1px solid #d1d5db', borderRadius: 6,
                                                            }}
                                                        />
                                                        <span style={{ color: '#6b7280' }}>
                                                            EIF 服务器上的本地 bundle 缓存模板；会按当前 sample 自动展开。
                                                        </span>
                                                    </label>
                                                </div>
                                                <label style={{ display: 'grid', gap: 4 }}>
                                                    <span style={{ fontWeight: 600, color: '#374151' }}>TTAV URL</span>
                                                    <input
                                                        value={ttavUrl}
                                                        onChange={e => setTtavUrl(e.target.value)}
                                                        placeholder={DEFAULT_TTAV_URL}
                                                        style={{
                                                            minWidth: 220, padding: '4px 8px',
                                                            border: '1px solid #d1d5db', borderRadius: 6,
                                                        }}
                                                    />
                                                </label>
                                                <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                                                    <label style={{ display: 'grid', gap: 4 }}>
                                                        <span style={{ fontWeight: 600, color: '#374151' }}>Method</span>
                                                        <input
                                                            value={ttavVisMethod}
                                                            onChange={e => setTtavVisMethod(e.target.value)}
                                                            placeholder={DEFAULT_TTAV_METHOD}
                                                            style={{
                                                                width: 110, padding: '4px 8px',
                                                                border: '1px solid #d1d5db', borderRadius: 6,
                                                            }}
                                                        />
                                                    </label>
                                                    <label style={{ display: 'grid', gap: 4 }}>
                                                        <span style={{ fontWeight: 600, color: '#374151' }}>Vis ID</span>
                                                        <input
                                                            value={ttavVisId}
                                                            onChange={e => setTtavVisId(e.target.value)}
                                                            placeholder={DEFAULT_TTAV_VIS_ID}
                                                            style={{
                                                                width: 84, padding: '4px 8px',
                                                                border: '1px solid #d1d5db', borderRadius: 6,
                                                            }}
                                                        />
                                                    </label>
                                                </div>
                                            </div>
                                        )}
                                        <div style={{
                                            display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center',
                                            fontSize: 12,
                                        }}>
                                            <button
                                                onClick={handlePrepareTtavBundle}
                                                disabled={preparingTtavBundle}
                                                style={{
                                                    padding: '4px 16px', borderRadius: 6,
                                                    border: '1px solid #0f766e', background: '#ecfeff', color: '#0f766e',
                                                    fontWeight: 600, fontSize: 13, cursor: preparingTtavBundle ? 'wait' : 'pointer',
                                                }}
                                            >
                                                {preparingTtavBundle
                                                    ? 'Preparing sample...'
                                                    : 'Prepare sample'}
                                            </button>
                                            <button
                                                onClick={handleOpenInTtav}
                                                style={{
                                                    padding: '4px 12px', borderRadius: 6,
                                                    border: '1px solid #94a3b8', background: '#fff', color: '#475569',
                                                    fontWeight: 500, fontSize: 13, cursor: 'pointer',
                                                }}
                                            >
                                                Open Visualizer
                                            </button>
                                        </div>
                                        <div style={{ fontSize: 11, color: ttavLaunchError ? '#b91c1c' : '#6b7280' }}>
                                            {ttavLaunchError
                                                ? ttavLaunchError
                                                : (ttavLaunchStatus ?? '默认配置已经指向公网 TTAV。通常先点 “Prepare sample”，再点 “Open TTAV”。')}
                                        </div>
                                        {preparingTtavBundle && ttavPrepareDetail && (
                                            <div style={{ fontSize: 11, color: '#0f766e' }}>
                                                {ttavPrepareDetail}
                                            </div>
                                        )}
                                    </div>
                                    )}
                                </div>

                                {trainGroups.length === 0 ? (
                                    <div className={styles.emptyState} style={{ padding: '32px 0' }}>
                                        {trainPanelEmptyHint}
                                    </div>
                                ) : (
                                    <div className={styles.trainGroupList}>
                                        {trainGroups.map(({ id, pairs }) => (
                                            <TrainSampleGroup
                                                key={id}
                                                trainIdx={id}
                                                pairs={pairs}
                                                detail={report.train_sample_details[String(id)]}
                                                onProbeEmbeddings={importedReportActive ? undefined : handleOpenTrainProbe}
                                                probeBusy={probingTrainSampleId === id}
                                                selectedPairIds={selectedTrainPairIdsByGroup[id] ?? []}
                                                onTogglePairSelection={toggleTrainPairSelection}
                                                comparisonSummary={trainProbeComparisons[id]}
                                            />
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                
        </div>
    );
}
