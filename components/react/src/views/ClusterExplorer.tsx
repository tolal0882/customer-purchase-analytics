import { useState } from "react";
import { setComponentValue } from "../streamlit";

type Cluster = {
  id: number;
  name: string;
  description: string;
  size: number;
  share_pct: number;
  purchase_rate_pct: number | null;
  z: Record<string, number>;
};
type Args = { clusters: Cluster[]; selected: number | null; title?: string };

export default function ClusterExplorer({ args }: { args: Args }) {
  const [sel, setSel] = useState<number | null>(args.selected);
  const pick = (id: number) => {
    setSel(id);
    setComponentValue({ selected: id });
  };
  const maxAbs = Math.max(1, ...args.clusters.flatMap((c) => Object.values(c.z).map((v) => Math.abs(v))));
  const maxShare = Math.max(...args.clusters.map((c) => c.share_pct));

  return (
    <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
      {args.clusters.map((c) => {
        const active = sel === c.id;
        const top = Object.entries(c.z).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 5);
        return (
          <button key={c.id} type="button" onClick={() => pick(c.id)}
            className={`rounded-2xl border p-4 text-left shadow-sm transition ${active ? "border-brand bg-white ring-2 ring-brand/40" : "border-line bg-white hover:border-brand/60"}`}>
            <div className="flex items-start justify-between gap-2">
              <div>
                <div className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">Cluster {c.id}</div>
                <div className="text-sm font-bold leading-snug text-forest">{c.name}</div>
              </div>
              <span className="rounded-full bg-accent/40 px-2 py-0.5 text-[11px] font-semibold text-forest">{c.share_pct.toFixed(1)}%</span>
            </div>
            <div className="mt-2 h-1.5 w-full rounded-full bg-surface">
              <div className="h-1.5 rounded-full bg-brand" style={{ width: `${(c.share_pct / maxShare) * 100}%` }} />
            </div>
            <div className="mt-1 flex justify-between text-[11px] text-slate-500">
              <span>{c.size.toLocaleString()} members</span>
              <span>{c.purchase_rate_pct == null ? "" : `purchase rate ${c.purchase_rate_pct.toFixed(2)}%`}</span>
            </div>
            <div className="mt-3 space-y-1">
              {top.map(([f, z]) => (
                <div key={f} className="flex items-center gap-2 text-[11px]">
                  <span className="w-36 truncate text-slate-600" title={f}>{f.replace(/_/g, " ")}</span>
                  <div className="relative h-2 flex-1 rounded bg-surface">
                    <div className="absolute top-0 h-2 w-px bg-slate-400" style={{ left: "50%" }} />
                    <div className={`absolute top-0 h-2 rounded ${z >= 0 ? "bg-brand" : "bg-red-400"}`}
                      style={{ left: z >= 0 ? "50%" : `${50 - (Math.abs(z) / maxAbs) * 50}%`, width: `${(Math.abs(z) / maxAbs) * 50}%` }} />
                  </div>
                  <span className="w-10 text-right tabular-nums">{z.toFixed(2)}</span>
                </div>
              ))}
            </div>
            <p className="mt-2 text-[11px] text-slate-500">{c.description}</p>
          </button>
        );
      })}
    </div>
  );
}
