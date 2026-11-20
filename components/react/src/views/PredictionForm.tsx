import { useEffect, useMemo, useState } from "react";
import { setComponentValue } from "../streamlit";

type Values = {
  total_events: number;
  total_views: number;
  unique_items_viewed: number;
  add_to_cart_count: number;
  session_duration_sec: number;
  first_cart_position: number;
  ends_with_cart: boolean;
  start_hour: number;
  start_dayofweek: number;
  unique_categories_viewed: number;
  hist_prior_sessions: number;
  hist_prior_purchases: number;
  hist_hours_since_prev_session: number | null;
};

type Args = {
  features: string[];
  policy: "n_events" | "time";
  n_events: number;
  values: Values;
  reset_token: string;
  disabled?: boolean;
};

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// Same consistency rules as src/prediction.py (Python re-validates; this is for instant feedback).
function validate(v: Values, a: Args): string[] {
  const e: string[] = [];
  if (v.total_events < 1) e.push("Events so far must be at least 1.");
  if (v.total_views + v.add_to_cart_count > v.total_events)
    e.push("Views + add-to-cart cannot exceed events so far (no transactions before the snapshot).");
  if (v.unique_items_viewed > v.total_views) e.push("Unique products cannot exceed total views.");
  if (v.total_views > 0 && v.unique_items_viewed < 1) e.push("Unique products must be at least 1 when views > 0.");
  if (v.total_events === 1 && v.session_duration_sec > 0) e.push("A single-event session has zero duration.");
  if (v.add_to_cart_count === 0 && v.first_cart_position !== -1) e.push("First cart position must be -1 without add-to-cart events.");
  if (v.add_to_cart_count > 0 && !(v.first_cart_position >= 0 && v.first_cart_position < v.total_events))
    e.push("First add-to-cart position must be between 0 and events-1.");
  if (v.ends_with_cart && v.add_to_cart_count === 0) e.push("'Last event was add-to-cart' needs at least one add-to-cart.");
  if (a.policy === "n_events" && v.total_events !== a.n_events)
    e.push(`Model trained on snapshots after exactly ${a.n_events} events (warning only: outside training distribution).`);
  return e;
}

const EXAMPLES: Record<string, Partial<Values>> = {
  "Browsing only": { total_events: 3, total_views: 3, unique_items_viewed: 3, add_to_cart_count: 0, session_duration_sec: 60, first_cart_position: -1, ends_with_cart: false },
  "Cart already started": { total_events: 3, total_views: 2, unique_items_viewed: 2, add_to_cart_count: 1, session_duration_sec: 90, first_cart_position: 2, ends_with_cart: true },
  "Repeated product views": { total_events: 3, total_views: 3, unique_items_viewed: 1, add_to_cart_count: 0, session_duration_sec: 45, first_cart_position: -1, ends_with_cart: false },
};

function Num({ label, hint, value, onChange, min, max, step = 1, disabled }: { label: string; hint?: string; value: number; onChange: (n: number) => void; min?: number; max?: number; step?: number; disabled?: boolean }) {
  return (
    <label className="block">
      <span className="text-xs font-semibold text-ink">{label}</span>
      <input
        type="number"
        className="mt-1 w-full rounded-lg border border-line bg-white px-3 py-2 text-sm focus:border-brand focus:outline-none focus:ring-2 focus:ring-brand/30 disabled:bg-slate-100"
        value={Number.isFinite(value) ? value : 0}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value === "" ? 0 : Number(e.target.value))}
      />
      {hint && <span className="mt-0.5 block text-[11px] text-slate-500">{hint}</span>}
    </label>
  );
}

export default function PredictionForm({ args }: { args: Args }) {
  const [v, setV] = useState<Values>(args.values);
  // Re-sync with Python state only when Python explicitly changes the token (e.g. after "reset").
  useEffect(() => setV(args.values), [args.reset_token]); // eslint-disable-line react-hooks/exhaustive-deps
  const set = <K extends keyof Values>(k: K, val: Values[K]) => setV((p) => ({ ...p, [k]: val }));
  const issues = useMemo(() => validate(v, args), [v, args]);
  const hard = issues.filter((m) => !m.startsWith("Model trained"));
  const f = new Set(args.features);

  const avg = v.total_events > 1 ? v.session_duration_sec / (v.total_events - 1) : 0;
  const ratio = v.total_events > 0 ? v.add_to_cart_count / v.total_events : 0;
  const revisit = v.total_views > 0 ? 1 - v.unique_items_viewed / v.total_views : 0;

  const submit = () => setComponentValue({ action: "predict", values: v, nonce: Date.now() });

  return (
    <div className="rounded-2xl border border-line bg-white p-5 shadow-sm">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-slate-500">Example inputs</span>
        {Object.entries(EXAMPLES).map(([name, patch]) => (
          <button key={name} type="button" onClick={() => setV((p) => ({ ...p, ...patch }))}
            className="rounded-full border border-line bg-surface px-3 py-1 text-xs text-forest hover:bg-accent/30">
            {name}
          </button>
        ))}
        <span className="text-[11px] text-slate-400">(convenience values only — not dataset records)</span>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
        <Num label="Events so far" value={v.total_events} min={1} onChange={(n) => set("total_events", n)}
          hint={args.policy === "n_events" ? `Training snapshots used exactly ${args.n_events}` : undefined} />
        <Num label="Product views" value={v.total_views} min={0} onChange={(n) => set("total_views", n)} />
        <Num label="Unique products viewed" value={v.unique_items_viewed} min={0} onChange={(n) => set("unique_items_viewed", n)} />
        <Num label="Add-to-cart events" value={v.add_to_cart_count} min={0} onChange={(n) => set("add_to_cart_count", n)} />
        <Num label="Cart removals" value={0} disabled onChange={() => {}} hint="Unavailable: dataset has no remove-from-cart event" />
        <Num label="Session elapsed time (s)" value={v.session_duration_sec} min={0} step={5} onChange={(n) => set("session_duration_sec", n)} />
        <Num label="Position of first add-to-cart" value={v.first_cart_position} min={-1} onChange={(n) => set("first_cart_position", n)} hint="-1 if none yet" />
        <label className="flex items-end gap-2 pb-2 text-xs font-semibold">
          <input type="checkbox" className="h-4 w-4 accent-[#2f855a]" checked={v.ends_with_cart} onChange={(e) => set("ends_with_cart", e.target.checked)} />
          Last event was add-to-cart
        </label>
        <div className="grid grid-cols-2 gap-2">
          <Num label="Start hour (UTC)" value={v.start_hour} min={0} max={23} onChange={(n) => set("start_hour", n)} />
          <label className="block">
            <span className="text-xs font-semibold">Weekday</span>
            <select className="mt-1 w-full rounded-lg border border-line bg-white px-2 py-2 text-sm" value={v.start_dayofweek}
              onChange={(e) => set("start_dayofweek", Number(e.target.value))}>
              {DAYS.map((d, i) => <option key={d} value={i}>{d}</option>)}
            </select>
          </label>
        </div>
        {f.has("unique_categories_viewed") && <Num label="Categories explored" value={v.unique_categories_viewed} min={0} onChange={(n) => set("unique_categories_viewed", n)} />}
        {f.has("hist_prior_sessions") && <Num label="Visitor's earlier sessions" value={v.hist_prior_sessions} min={0} onChange={(n) => set("hist_prior_sessions", n)} />}
        {f.has("hist_prior_purchases") && <Num label="Visitor's earlier purchases" value={v.hist_prior_purchases} min={0} onChange={(n) => set("hist_prior_purchases", n)} />}
        {f.has("hist_hours_since_prev_session") && (
          <Num label="Hours since previous session" value={v.hist_hours_since_prev_session ?? 0} min={0} step={0.5}
            onChange={(n) => set("hist_hours_since_prev_session", n)} hint="0 is treated as a real value" />
        )}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-3 rounded-xl bg-surface p-3 text-center text-xs">
        <div><div className="text-slate-500">Avg interval (derived)</div><div className="text-base font-bold text-forest">{avg.toFixed(1)} s</div></div>
        <div><div className="text-slate-500">Cart ratio (derived)</div><div className="text-base font-bold text-forest">{(ratio * 100).toFixed(0)}%</div></div>
        <div><div className="text-slate-500">Revisit ratio (derived)</div><div className="text-base font-bold text-forest">{(revisit * 100).toFixed(0)}%</div></div>
      </div>

      {issues.length > 0 && (
        <ul className="mt-3 space-y-1 text-xs">
          {issues.map((m) => (
            <li key={m} className={m.startsWith("Model trained") ? "text-amber-700" : "text-red-700"}>• {m}</li>
          ))}
        </ul>
      )}

      <button type="button" onClick={submit} disabled={hard.length > 0 || args.disabled}
        className="mt-4 w-full rounded-xl bg-brand px-4 py-3 text-sm font-semibold text-white shadow hover:bg-forest disabled:cursor-not-allowed disabled:opacity-50">
        Predict purchase likelihood
      </button>
    </div>
  );
}
