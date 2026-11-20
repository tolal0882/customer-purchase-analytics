// Minimal implementation of the Streamlit custom-component protocol (apiVersion 1).
// Using the raw postMessage protocol avoids an extra npm dependency and works with
// streamlit.components.v1.declare_component(path=...).
import { useEffect, useRef, useState } from "react";

type Json = unknown;

function post(message: Record<string, Json>) {
  window.parent.postMessage({ isStreamlitMessage: true, ...message }, "*");
}

export function setComponentValue(value: Json) {
  post({ type: "streamlit:setComponentValue", value, dataType: "json" });
}

export function setFrameHeight(height: number) {
  post({ type: "streamlit:setFrameHeight", height });
}

/** Subscribes to Streamlit render events and returns the latest args from Python. */
export function useStreamlitArgs<T = Record<string, any>>(): T | null {
  const [args, setArgs] = useState<T | null>(null);
  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      if (event.data?.type === "streamlit:render") setArgs(event.data.args as T);
    };
    window.addEventListener("message", onMessage);
    post({ type: "streamlit:componentReady", apiVersion: 1 });
    return () => window.removeEventListener("message", onMessage);
  }, []);
  return args;
}

/** Keeps the iframe height equal to the rendered content height. */
export function useAutoHeight() {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const el = ref.current;
    const report = () => setFrameHeight(Math.ceil(el.getBoundingClientRect().height) + 8);
    report();
    const ro = new ResizeObserver(report);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return ref;
}
