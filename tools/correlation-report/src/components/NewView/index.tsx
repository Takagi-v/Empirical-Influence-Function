import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from 'react';
import styles from './NewView.module.css';
import {
    ReportPanel,
    normalizeAllTokensReport,
    normalizeImportedReport,
    type AllTokensExperimentMeta,
    type AllTokensReport,
} from './ReportPanel';

export type { AllTokensExperimentMeta };

interface Props {
    metas: AllTokensExperimentMeta[];
}

type SlotId = 'left' | 'right';

interface SlotState {
    report: AllTokensReport | null;
    meta: AllTokensExperimentMeta | null;
    status: string | null;
    error: string | null;
}

function emptySlot(): SlotState {
    return { report: null, meta: null, status: null, error: null };
}

function modelLabelFrom(report: AllTokensReport | null, meta: AllTokensExperimentMeta | null, fallback: string): string {
    return report?.experiment_meta.model_name
        || meta?.label
        || fallback;
}

function SlotImportCard({
    title,
    slot,
    metas,
    dragging,
    onDrag,
    onDrop,
    onFile,
    onPickMeta,
    onClear,
}: {
    title: string;
    slot: SlotState;
    metas: AllTokensExperimentMeta[];
    dragging: boolean;
    onDrag: (active: boolean) => void;
    onDrop: (file: File | null | undefined) => void;
    onFile: (file: File | null | undefined) => void;
    onPickMeta: (idx: number) => void;
    onClear: () => void;
}) {
    return (
        <div className={styles.slotImportCard}>
            <div className={styles.slotImportHeader}>
                <div>
                    <div className={styles.slotImportTitle}>{title}</div>
                    <div className={styles.slotImportDesc}>
                        {slot.report
                            ? modelLabelFrom(slot.report, slot.meta, 'loaded')
                            : 'Import JSON or pick a bundled experiment'}
                    </div>
                </div>
                {slot.report && (
                    <button type="button" className={styles.slotClearBtn} onClick={onClear}>
                        Clear
                    </button>
                )}
            </div>

            <label
                className={`${styles.importDropZone} ${dragging ? styles.importDropZoneActive : ''}`}
                onDragOver={event => {
                    event.preventDefault();
                    onDrag(true);
                }}
                onDragLeave={() => onDrag(false)}
                onDrop={event => {
                    event.preventDefault();
                    onDrag(false);
                    onDrop(event.dataTransfer.files?.[0]);
                }}
            >
                <input
                    type="file"
                    accept=".json,application/json"
                    className={styles.importFileInput}
                    onChange={(event: ChangeEvent<HTMLInputElement>) => {
                        onFile(event.target.files?.[0]);
                        event.target.value = '';
                    }}
                />
                <span className={styles.importDropMain}>Choose JSON</span>
                <span className={styles.importDropSub}>all-token report</span>
            </label>

            {metas.length > 0 && (
                <div className={styles.slotMetaList}>
                    {metas.map((m, i) => (
                        <button
                            key={`${m.fileName}-${i}`}
                            type="button"
                            className={`${styles.metaBtn} ${slot.meta?.fileName === m.fileName ? styles.metaBtnActive : ''}`}
                            onClick={() => onPickMeta(i)}
                        >
                            {m.label}
                        </button>
                    ))}
                </div>
            )}

            {slot.status && <div className={styles.importStatus}>{slot.status}</div>}
            {slot.error && <div className={styles.importError}>{slot.error}</div>}
        </div>
    );
}

export function NewView({ metas }: Props) {
    const [left, setLeft] = useState<SlotState>(() => emptySlot());
    const [right, setRight] = useState<SlotState>(() => emptySlot());
    const [draggingLeft, setDraggingLeft] = useState(false);
    const [draggingRight, setDraggingRight] = useState(false);
    const [linkTokenSelection, setLinkTokenSelection] = useState(false);
    const [linkedTokIdx, setLinkedTokIdx] = useState<number | null>(null);

    const dualMode = Boolean(left.report && right.report);

    const setSlot = useCallback((id: SlotId, next: SlotState) => {
        if (id === 'left') setLeft(next);
        else setRight(next);
    }, []);

    const activatePayload = useCallback((id: SlotId, payload: unknown, sourceName: string) => {
        try {
            const imported = normalizeImportedReport(payload, sourceName);
            const modelName = imported.report.experiment_meta.model_name;
            const meta: AllTokensExperimentMeta = {
                ...imported.meta,
                label: modelName
                    ? `${modelName} · ${imported.meta.label.replace(/ \(uploaded\)$/, '')}`
                    : imported.meta.label,
                fileName: `uploaded:${sourceName}`,
            };
            setSlot(id, {
                report: imported.report,
                meta,
                status: `Loaded ${imported.report.per_token_results.length} token(s) from ${sourceName}`
                    + (modelName ? ` [${modelName}]` : ''),
                error: null,
            });
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to parse JSON.';
            setSlot(id, {
                report: null,
                meta: null,
                status: null,
                error: `Failed to import ${sourceName}: ${message}`,
            });
        }
    }, [setSlot]);

    const loadMeta = useCallback(async (id: SlotId, meta: AllTokensExperimentMeta) => {
        setSlot(id, {
            report: null,
            meta,
            status: `Loading ${meta.fileName}...`,
            error: null,
        });
        try {
            const resp = await fetch(`/data/results/${meta.fileName}`);
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = normalizeAllTokensReport(await resp.json() as AllTokensReport);
            const modelName = data.experiment_meta.model_name;
            setSlot(id, {
                report: data,
                meta: {
                    ...meta,
                    label: modelName ? `${modelName} · ${meta.taskId}` : meta.label,
                },
                status: `Loaded ${meta.fileName}` + (modelName ? ` [${modelName}]` : ''),
                error: null,
            });
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to load.';
            setSlot(id, {
                report: null,
                meta: null,
                status: null,
                error: `Failed to load ${meta.fileName}: ${message}`,
            });
        }
    }, [setSlot]);

    const handleFile = useCallback(async (id: SlotId, file: File | null | undefined) => {
        if (!file) return;
        try {
            const text = await file.text();
            activatePayload(id, JSON.parse(text), file.name);
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Failed to parse JSON.';
            setSlot(id, {
                report: null,
                meta: null,
                status: null,
                error: `Failed to import ${file.name}: ${message}`,
            });
        }
    }, [activatePayload, setSlot]);

    // Optional URL query: ?leftUrl=...&rightUrl=...
    useEffect(() => {
        if (typeof window === 'undefined') return;
        const params = new URLSearchParams(window.location.search);
        const leftUrl = params.get('leftUrl') ?? params.get('reportUrl') ?? params.get('report_url');
        const rightUrl = params.get('rightUrl');

        const loadUrl = async (id: SlotId, raw: string) => {
            try {
                const url = new URL(raw, window.location.href);
                const resp = await fetch(url.toString(), { cache: 'no-store' });
                if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                activatePayload(id, await resp.json(), url.pathname.split('/').pop() || url.host);
            } catch (error) {
                const message = error instanceof Error ? error.message : 'Failed to load URL.';
                setSlot(id, { report: null, meta: null, status: null, error: message });
            }
        };

        if (leftUrl) void loadUrl('left', leftUrl);
        if (rightUrl) void loadUrl('right', rightUrl);
    }, [activatePayload, setSlot]);

    const onTokChange = useCallback((idx: number | null) => {
        setLinkedTokIdx(idx);
    }, []);

    const loadedCount = useMemo(() => Number(Boolean(left.report)) + Number(Boolean(right.report)), [left.report, right.report]);

    return (
        <div className={styles.root}>
            <div className={styles.dualImportRow}>
                <SlotImportCard
                    title="Model A (left)"
                    slot={left}
                    metas={metas}
                    dragging={draggingLeft}
                    onDrag={setDraggingLeft}
                    onDrop={file => void handleFile('left', file)}
                    onFile={file => void handleFile('left', file)}
                    onPickMeta={idx => { const m = metas[idx]; if (m) void loadMeta('left', m); }}
                    onClear={() => setLeft(emptySlot())}
                />
                <SlotImportCard
                    title="Model B (right)"
                    slot={right}
                    metas={metas}
                    dragging={draggingRight}
                    onDrag={setDraggingRight}
                    onDrop={file => void handleFile('right', file)}
                    onFile={file => void handleFile('right', file)}
                    onPickMeta={idx => { const m = metas[idx]; if (m) void loadMeta('right', m); }}
                    onClear={() => setRight(emptySlot())}
                />
            </div>

            <div className={styles.compareToolbar}>
                <label className={styles.linkToggle}>
                    <input
                        type="checkbox"
                        checked={linkTokenSelection}
                        onChange={e => setLinkTokenSelection(e.target.checked)}
                        disabled={!dualMode}
                    />
                    Link token selection across models
                </label>
                <span className={styles.compareHint}>
                    {loadedCount === 0
                        ? 'Load one JSON for single-model view, or two for side-by-side compare.'
                        : loadedCount === 1
                            ? 'Single-model mode — all attribution features available. Load a second JSON to compare.'
                            : 'Dual-model mode — panels scroll independently. Enable link only if you want shared token clicks.'}
                </span>
            </div>

            {loadedCount === 0 && (
                <div className={styles.emptyState}>
                    Import or select at least one correlation report to begin.
                </div>
            )}

            <div className={dualMode ? styles.dualGrid : styles.singleGrid}>
                {left.report && left.meta && (
                    <div className={styles.modelColumn}>
                        <ReportPanel
                            report={left.report}
                            meta={left.meta}
                            modelLabel={modelLabelFrom(left.report, left.meta, 'Model A')}
                            compact={dualMode}
                            selectedTokIdx={dualMode && linkTokenSelection ? linkedTokIdx : undefined}
                            onSelectedTokIdxChange={dualMode && linkTokenSelection ? onTokChange : undefined}
                        />
                    </div>
                )}
                {right.report && right.meta && (
                    <div className={styles.modelColumn}>
                        <ReportPanel
                            report={right.report}
                            meta={right.meta}
                            modelLabel={modelLabelFrom(right.report, right.meta, 'Model B')}
                            compact={dualMode}
                            selectedTokIdx={dualMode && linkTokenSelection ? linkedTokIdx : undefined}
                            onSelectedTokIdxChange={dualMode && linkTokenSelection ? onTokChange : undefined}
                        />
                    </div>
                )}
            </div>
        </div>
    );
}
