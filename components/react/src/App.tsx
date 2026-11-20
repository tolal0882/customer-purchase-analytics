import { useAutoHeight, useStreamlitArgs } from "./streamlit";
import PredictionForm from "./views/PredictionForm";
import ClusterExplorer from "./views/ClusterExplorer";
import ModelCompare from "./views/ModelCompare";

export default function App() {
  const args = useStreamlitArgs();
  const ref = useAutoHeight();
  return (
    <div ref={ref} className="p-1">
      {!args && <div className="text-sm text-slate-500">Loading component…</div>}
      {args?.view === "prediction_form" && <PredictionForm args={args as any} />}
      {args?.view === "cluster_explorer" && <ClusterExplorer args={args as any} />}
      {args?.view === "model_compare" && <ModelCompare args={args as any} />}
    </div>
  );
}
