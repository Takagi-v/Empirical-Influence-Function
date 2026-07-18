import react from '@vitejs/plugin-react-swc'
import { defineConfig, type Plugin } from 'vite'
import { readdirSync, existsSync, readFileSync } from 'fs'
import { resolve, join } from 'path'
import type { IncomingMessage, ServerResponse } from 'http'

// ── Repo root where JSON experiment files live ────────────────────────────────
const DATA_ROOT          = resolve(__dirname, '../../')
const MODEL_COMPARE_DIR  = resolve(DATA_ROOT, 'legacy_by_model_sample')
const CORR_RESULTS_DIR   = resolve(DATA_ROOT, 'correlation_matching_results')
const REAL_BUNDLE_DIR    = resolve(DATA_ROOT, 'ttav_bundles_real')

interface ModelInfo { slug: string; name: string }
interface RawModelInfo { model_slug: string; model_name?: unknown }
interface ManifestData {
  allTokensExperiments: { taskId: string; label: string; fileName: string }[]
  modelCompare: { models: ModelInfo[]; sampleIds: string[]; oursSlug: string } | null
}
interface MiddlewareServer {
  middlewares: {
    use: (fn: (req: IncomingMessage, res: ServerResponse, next: () => void) => void) => void
  }
}

function isRawModelInfo(value: unknown): value is RawModelInfo {
  return typeof value === 'object'
    && value !== null
    && typeof (value as { model_slug?: unknown }).model_slug === 'string'
}

function isSafeSegment(value: string): boolean {
  return /^[\w-]+$/.test(value)
}

// ─────────────────────────────────────────────────────────────────────────────
// experimentDataPlugin
//
// Dev / Preview server:
//   GET /data/index.json                             → manifest
//   GET /data/results/<filename>.json                → correlation result files
//   GET /data/model-sample/<slug>/<sampleId>/latest_saliency.json
//
// Build:
//   emits dist/data/index.json + all data files
// ─────────────────────────────────────────────────────────────────────────────
function experimentDataPlugin(): Plugin {
  function buildManifest(): ManifestData {
    const allTokensExperiments: { taskId: string; label: string; fileName: string }[] = []

    // Read legacy_by_model_sample/manifest.json
    const modelCompare: { models: ModelInfo[]; sampleIds: string[]; oursSlug: string } | null = (() => {
      const mp = join(MODEL_COMPARE_DIR, 'manifest.json')
      if (!existsSync(mp)) return null
      try {
        const mm = JSON.parse(readFileSync(mp, 'utf-8')) as { models?: unknown[] }
        const models: ModelInfo[] = (mm.models ?? [])
          .filter((m): m is RawModelInfo => isRawModelInfo(m) && isSafeSegment(m.model_slug))
          .map(m => ({ slug: m.model_slug, name: typeof m.model_name === 'string' ? m.model_name : m.model_slug }))
        const oursSlug = models.find(m => m.slug === 'ours_graphsignal')?.slug ?? models[0]?.slug ?? ''
        const oursDir = join(MODEL_COMPARE_DIR, oursSlug)
        const sampleIds = existsSync(oursDir)
          ? readdirSync(oursDir).filter(isSafeSegment)
          : []
        return models.length > 0 && sampleIds.length > 0 ? { models, sampleIds, oursSlug } : null
      } catch { return null }
    })()

    let files: string[] = []
    try { files = readdirSync(CORR_RESULTS_DIR) } catch { /* directory not present */ }

    for (const f of files) {
      if (!f.endsWith('.json')) continue
      // Use stem as taskId; strip trailing _all_tokens if present for cleaner display
      const stem = f.slice(0, -5)
      const taskId = stem.endsWith('_all_tokens') ? stem.slice(0, -11) : stem
      allTokensExperiments.push({ taskId, label: taskId, fileName: f })
    }

    allTokensExperiments.sort((a, b) => a.taskId.localeCompare(b.taskId))

    return { allTokensExperiments, modelCompare }
  }

  function readDataFile(filename: string): string | null {
    const safe = filename.replace(/[/\\]/g, '').replace(/\.\./g, '')
    if (!safe.endsWith('.json') || safe.includes('/') || safe.includes('\\')) return null
    const filePath = join(CORR_RESULTS_DIR, safe)
    if (!existsSync(filePath)) return null
    return readFileSync(filePath, 'utf-8')
  }

  function readModelSaliency(slug: string, sampleId: string): string | null {
    if (!isSafeSegment(slug) || !isSafeSegment(sampleId)) return null
    const filePath = join(MODEL_COMPARE_DIR, slug, sampleId, 'latest_saliency.json')
    if (!existsSync(filePath)) return null
    return readFileSync(filePath, 'utf-8')
  }

  function readRealBundlePayload(sampleId: string): string | null {
    if (!isSafeSegment(sampleId)) return null
    const filePath = join(REAL_BUNDLE_DIR, sampleId, 'bundle_payload.json')
    if (!existsSync(filePath)) return null
    return readFileSync(filePath, 'utf-8')
  }

  function addMiddleware(server: MiddlewareServer) {
    server.middlewares.use((req, res, next) => {
      const reqUrl = req.url ?? ''
      if (reqUrl === '/data/index.json') {
        res.setHeader('Content-Type', 'application/json')
        res.setHeader('Cache-Control', 'no-cache')
        res.end(JSON.stringify(buildManifest()))
        return
      }
      // Model compare: /data/model-sample/<slug>/<sampleId>/latest_saliency.json
      const modelM = reqUrl.match(/^\/data\/model-sample\/([^/?]+)\/([^/?]+)\/latest_saliency\.json/)
      if (modelM) {
        const content = readModelSaliency(decodeURIComponent(modelM[1]), decodeURIComponent(modelM[2]))
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      // All-tokens correlation files: /data/results/<filename>.json
      const resultsM = reqUrl.match(/^\/data\/results\/([^/?]+\.json)/)
      if (resultsM) {
        const content = readDataFile(decodeURIComponent(resultsM[1]))
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      const realBundleM = (req.url as string)?.match(/^\/data\/real-bundles\/([^/?]+)\/bundle_payload\.json/)
      if (realBundleM) {
        const content = readRealBundlePayload(decodeURIComponent(realBundleM[1]))
        if (content !== null) {
          res.setHeader('Content-Type', 'application/json')
          res.setHeader('Cache-Control', 'no-cache')
          res.end(content)
          return
        }
      }
      next()
    })
  }

  return {
    name: 'experiment-data',

    configureServer:        addMiddleware,
    configurePreviewServer: addMiddleware,

    generateBundle() {
      const manifest = buildManifest()
      this.emitFile({
        type: 'asset',
        fileName: 'data/index.json',
        source: JSON.stringify(manifest, null, 2),
      })

      // All-tokens experiment files
      for (const exp of manifest.allTokensExperiments) {
        const content = readDataFile(exp.fileName)
        if (content) this.emitFile({ type: 'asset', fileName: `data/results/${exp.fileName}`, source: content })
      }

      const realBundleIds = existsSync(REAL_BUNDLE_DIR)
        ? readdirSync(REAL_BUNDLE_DIR).filter(isSafeSegment)
        : []
      for (const sampleId of realBundleIds) {
        const content = readRealBundlePayload(sampleId)
        if (content) {
          this.emitFile({
            type: 'asset',
            fileName: `data/real-bundles/${sampleId}/bundle_payload.json`,
            source: content,
          })
        }
      }

      // Model compare files
      if (manifest.modelCompare) {
        for (const model of manifest.modelCompare.models) {
          for (const sampleId of manifest.modelCompare.sampleIds) {
            const content = readModelSaliency(model.slug, sampleId)
            if (content) {
              this.emitFile({
                type: 'asset',
                fileName: `data/model-sample/${model.slug}/${sampleId}/latest_saliency.json`,
                source: content,
              })
            }
          }
        }
      }
    },
  }
}

// Reverse-proxy /api/* to the EIF bundle API so the frontend can call it
// same-origin (see getDefaultEifApiUrl). Works for both `vite` (dev) and
// `vite preview`. In VS Code forwarded-localhost setups this means you only
// need to forward the vite port — no separate 8766 forward, no port mismatch.
const EIF_API_PROXY = {
  '/api': {
    target: 'http://127.0.0.1:8766',
    changeOrigin: true,
  },
}

// Pin a dedicated port so this app never collides with TTAV's vite (5173).
// strictPort makes vite fail loudly instead of silently drifting to 5174,
// which was causing "am I on the report or on TTAV?" confusion.
const EIF_REPORT_PORT = 5273

export default defineConfig({
  plugins: [react(), experimentDataPlugin()],
  server: { port: EIF_REPORT_PORT, strictPort: true, proxy: EIF_API_PROXY },
  preview: { port: EIF_REPORT_PORT, strictPort: true, proxy: EIF_API_PROXY },
})
