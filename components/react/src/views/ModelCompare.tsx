import { useMemo, useState } from "react";
import { setComponentValue } from "../streamlit";

type Exp = { id: string; label: string; model: string; created_at: string; active: boolean; metrics: Record<string, number | null> };
type Args = { experiments: Exp[]; selected: string[]; metric: string; metrics: string[] };

export default function ModelCompare({ args }: { args: Args }) {
  const [metric, setMetric] = useState(args.metric);
  const [sel, setSel] = useState<string[]>(args.selected);

  const ranked = useMemo(
    () => [...args.experiments].sort((a, b) => (b.metrics[metric] ?? -1) - (a.metrics[metric] ?? -1)),
    [args.experiments, metric],
  );
  const emit = (nextSel: string[], nextMetric: string, extra: Record<string, unknown> = {}) =>
    setComponentValue({ selected: nextSel, metric: nextMetric, nonce: Date.now(), ...extra });

  const toggle = (id: string) => {
    const next = sel.includes(id) ? sel.filter((s) => s !== id) : [...sel, id];
    setSel(next);
    emit(next, metric);
  };

  return (
    <div className="rounded-2xl border border-line bg-white p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Rank by</span>
        {args.metrics.map((m) => (
          <button key={m} type="button" onClick={() => { setMetric(m); emit(sel, m); }}
            className={`rounded-full px-3 py-1 text-xs font-semibold ${m === metric ? "bg-forest text-white" : "border border-line bg-surface text-forest"}`}>
            {m.replace("_", "-")}
          </button>
        ))}
      </div>
      <div className="space-y-2">
        {ranked.map((e, i) => {
          const val = e.metrics[metric];
          const on = sel.includes(e.id);
          return (
            <div key={e.id} className={`rounded-xl border p-3 ${on ? "border-brand bg-surface" : "border-line"}`}>
              <div className="flex items-center gap-3">
                <input type="checkbox" className="h-4 w-4 accent-[#2f855a]" checked={on} onChange={() => toggle(e.id)} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 text-sm font-semibold text-forest">
                    <span className="text-slate-400">#{i + 1}</span>
                    <span className="truncate">{e.label}</span>
                    {e.active && <span className="rounded-full bg-accent/50 px-2 text-[10px]">ACTIVE</span>}
                  </div>
                  <div className="text-[11px] text-slate-500">{e.model} · {e.created_at}</div>
                </div>
                <div className="text-right text-base font-bold tabular-nums text-forest">{val == null ? "n/a" : val.toFixed(3)}</div>
                {!e.active && (
                  <button type="button" onClick={() => emit(sel, metric, { set_active: e.id })}
                    className="rounded-lg border border-brand px-2 py-1 text-[11px] font-semibold text-brand hover:bg-brand hover:text-white">
                    Set active
                  </button>
                )}
              </div>
              <div className="mt-2 h-1.5 rounded-full bg-surface">
                <div className="h-1.5 rounded-full bg-brand" style={{ width: `${Math.max(0, Math.min(1, val ?? 0)) * 100}%` }} />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
